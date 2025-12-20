import torch
from torch import nn
from tqdm import tqdm
from muon import Muon
from math import sqrt
from torch.nn import functional as F
from utils import AttentionMask, ModelConfig, top_p


# Rotary positional encoding
# ---------------------------------------
def rope(embedding_dim: int, maxlen: int, base: int):
    freqs = 1 / (
        base
        ** (
            torch.arange(0, embedding_dim, 2)[: (embedding_dim // 2)].float()
            / embedding_dim
        )
    )
    t = torch.arange(0, maxlen, dtype=torch.float)
    freq_cis = torch.outer(t, freqs)
    freq_cis = torch.polar(torch.ones_like(freq_cis), freq_cis)
    freq_cis = torch.stack([freq_cis.real, freq_cis.imag], dim=-1)
    return freq_cis


def apply_rope(x: torch.Tensor, freq_cis: torch.Tensor):
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    freq_cis = freq_cis.view(1, x_.size(1), 1, x_.size(3), 2)
    x_out = torch.stack(
        [
            x_[..., 0] * freq_cis[..., 0] - x_[..., 1] * freq_cis[..., 1],
            x_[..., 1] * freq_cis[..., 0] + x_[..., 0] * freq_cis[..., 1],
        ],
        dim=-1,
    )
    x_out = x_out.flatten(3)
    return x_out.type_as(x)


# RMS Normalization
# ---------------------------------------
class RMS_Norm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim))
        self.eps = eps

    def norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.norm(x) * self.weight


# KVCache
# ---------------------------------------
class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        maxlen: int,
        num_heads: int,
        k_dim: int,
        v_dim: int,
        device: torch.device,
    ):
        super().__init__()
        k_cache_shape = (batch_size, maxlen, num_heads, k_dim)
        v_cache_shape = (batch_size, maxlen, num_heads, v_dim)
        self.register_buffer("k_cache", torch.zeros(k_cache_shape).to(device), False)
        self.register_buffer("v_cache", torch.zeros(v_cache_shape).to(device), False)

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        T = k.size(1)
        self.k_cache[:, start_pos : start_pos + T] = k
        self.v_cache[:, start_pos : start_pos + T] = v
        return self.k_cache[:, : start_pos + T], self.v_cache[:, : start_pos + T]


# Multi-Head Attention
# ---------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, conf: ModelConfig, type_):
        super().__init__()
        self.type_ = type_
        self.conf = conf
        self.head_dim = conf.embedding_dim // conf.num_heads

        assert self.head_dim * conf.num_heads == conf.embedding_dim, (
            "embedding_dim must be divisible by num_heads"
        )
        self.n_rep = conf.num_heads // conf.kv_heads

        self.query = nn.Linear(
            conf.embedding_dim, conf.num_heads * self.head_dim, bias=conf.atten_bias
        )
        self.key = nn.Linear(
            conf.embedding_dim, conf.kv_heads * self.head_dim, bias=conf.atten_bias
        )
        self.value = nn.Linear(
            conf.embedding_dim, conf.kv_heads * self.head_dim, bias=conf.atten_bias
        )

        self.proj = nn.Linear(
            conf.num_heads * self.head_dim, conf.embedding_dim, bias=conf.atten_bias
        )

        self.head_dp = nn.Dropout(conf.atten_dropout)
        self.atten_dp = nn.Dropout(conf.atten_dropout)

        self.cache: KVCache | None = None

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor,
        window: torch.Tensor,
    ):
        B, T, C = x.shape
        q, k, v = self.query.forward(x), self.key.forward(x), self.value.forward(x)

        q = q.view(B, T, self.conf.num_heads, self.head_dim)
        k = k.view(B, T, self.conf.kv_heads, self.head_dim)
        v = v.view(B, T, self.conf.kv_heads, self.head_dim)

        q = apply_rope(q, freq_cis)
        k = apply_rope(k, freq_cis)

        if self.cache is not None:
            k, v = self.cache.update(k, v, start_pos)

        k = torch.repeat_interleave(k, self.n_rep, dim=2)
        v = torch.repeat_interleave(v, self.n_rep, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.conf.flash:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask if self.type_ == AttentionMask.Global else window,
                dropout_p=self.conf.atten_dropout,
            )
        else:
            atte = q @ k.transpose(-2, -1) * (1 / sqrt(self.head_dim))
            if self.type_ == AttentionMask.Global:
                atte = atte + mask
            else:
                atte = atte + window
            atte = self.head_dp.forward(F.softmax(atte, -1))
            output = atte @ v

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.proj.forward(output)
        return self.atten_dp.forward(output)


# Multi-Head Latent Attention
# ---------------------------------------
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, conf: ModelConfig, type_):
        super().__init__()
        self.type_ = type_
        self.conf = conf
        self.qk_head_dim = conf.qk_nope_dim + conf.qk_rope_dim

        self.query = nn.Linear(
            conf.embedding_dim, conf.num_heads * self.qk_head_dim, bias=conf.atten_bias
        )

        # kv init
        self.compress_kv = nn.Linear(
            conf.embedding_dim,
            conf.kv_lora_rank + conf.qk_rope_dim,
            bias=conf.atten_bias,
        )
        self.kv_norm = RMS_Norm(conf.kv_lora_rank, conf.eps)
        self.decompress_kv = nn.Linear(
            conf.kv_lora_rank,
            conf.num_heads * (conf.qk_nope_dim + conf.v_dim),
            bias=conf.atten_bias,
        )

        self.proj = nn.Linear(
            conf.num_heads * conf.v_dim, conf.embedding_dim, bias=conf.atten_bias
        )

        self.head_dp = nn.Dropout(conf.atten_dropout)
        self.atten_dp = nn.Dropout(conf.atten_dropout)

        self.cache: KVCache | None = None

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor,
        window: torch.Tensor,
    ):
        B, T, C = x.shape
        # query rope part
        q = self.query.forward(x)
        q = q.view(B, T, self.conf.num_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.conf.qk_nope_dim, self.conf.qk_rope_dim], -1
        )
        q_rope = apply_rope(q_rope, freq_cis)

        # key rope part
        kv = self.compress_kv.forward(x)
        kv, k_rope = torch.split(
            kv, [self.conf.kv_lora_rank, self.conf.qk_rope_dim], -1
        )
        k_rope = k_rope.view(B, T, 1, self.conf.qk_rope_dim)
        k_rope = apply_rope(k_rope, freq_cis)

        # value and nope part
        kv = self.kv_norm.forward(kv)
        kv = self.decompress_kv.forward(kv)
        kv = kv.view(B, T, self.conf.num_heads, self.conf.qk_nope_dim + self.conf.v_dim)
        k_nope, v = torch.split(kv, [self.conf.qk_nope_dim, self.conf.v_dim], -1)

        # cat q and k
        q = torch.cat([q_nope, q_rope], -1)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.conf.num_heads, -1)], -1)

        # classic attention math
        if self.cache is not None:
            k, v = self.cache.update(k, v, start_pos)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.conf.flash:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask if self.type_ == AttentionMask.Global else window,
                dropout_p=self.conf.atten_dropout,
            )
        else:
            atten = q @ k.transpose(-2, -1) * (1 / sqrt(self.qk_head_dim))
            if self.type_ == AttentionMask.Global:
                atten = atten + mask
            else:
                atten = atten + window
            atten = self.head_dp.forward(F.softmax(atten, -1))
            output = atten @ v

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.proj.forward(output)
        return self.atten_dp.forward(output)


# Feed-Forward Network
# ---------------------------------------
class FFN(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.ln1 = nn.Linear(conf.embedding_dim, conf.inter_dim, bias=conf.ffn_bias)
        self.ln2 = nn.Linear(conf.embedding_dim, conf.inter_dim, bias=conf.ffn_bias)
        self.proj = nn.Linear(conf.inter_dim, conf.embedding_dim, bias=conf.ffn_bias)
        self.dropout = nn.Dropout(conf.ffn_dropout)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.ln1.forward(x)) * self.ln2.forward(x)
        x = self.dropout.forward(self.proj.forward(x))
        return x


# Expert Network
# ---------------------------------------
class Expert(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.ln1 = nn.Linear(
            conf.embedding_dim, conf.expert_inter_dim, bias=conf.ffn_bias
        )
        self.ln2 = nn.Linear(
            conf.embedding_dim, conf.expert_inter_dim, bias=conf.ffn_bias
        )
        self.proj = nn.Linear(
            conf.expert_inter_dim, conf.embedding_dim, bias=conf.ffn_bias
        )
        self.dropout = nn.Dropout(conf.ffn_dropout)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.ln1.forward(x)) * self.ln2.forward(x)
        x = self.dropout.forward(self.proj.forward(x))
        return x


# Mixture-Of-Experts
# ---------------------------------------
class MoE(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.conf = conf
        self.gate = nn.Linear(conf.embedding_dim, conf.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(conf) for _ in range(conf.n_experts)])
        self.shared_expert = FFN(conf)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        x = x.view(-1, C)
        y = torch.zeros_like(x)

        scores = self.gate.forward(x)
        if self.training:
            scores = scores + torch.randn_like(scores)
        scores_ = F.softmax(scores, dim=-1)
        indices = torch.topk(scores_, k=self.conf.active_experts, dim=-1)[1]
        weights = scores_.gather(1, indices)

        for i in range(0, self.conf.n_experts):
            idx, top = torch.where(indices == i)
            expert = self.experts[i]
            y[idx] += expert.forward(x[idx]) * weights[idx, top, None]

        z = self.shared_expert.forward(x)

        return (y + z).view(B, T, C), scores


# Load Balance Loss
# ---------------------------------------
def load_balance_loss(moe_logits: tuple[torch.Tensor], conf: ModelConfig):
    concat_logits = torch.cat([logit for logit in moe_logits])
    weights = F.softmax(concat_logits, dim=-1)
    expert = torch.topk(weights, k=conf.active_experts, dim=-1)[1]
    mask = F.one_hot(expert, conf.n_experts)
    token_per_expert = torch.mean(mask.float(), dim=0)
    prob_per_expert = torch.mean(weights, dim=0)
    overall_loss = torch.sum(token_per_expert * prob_per_expert.unsqueeze(0))
    return overall_loss * conf.n_experts


# Transformer Block
# ---------------------------------------
class Block(nn.Module):
    def __init__(self, conf: ModelConfig, atten_type):
        super().__init__()
        self.conf = conf
        self.norm1 = RMS_Norm(conf.embedding_dim, conf.eps)
        if conf.mla:
            self.atten = MultiHeadLatentAttention(conf, atten_type)
        else:
            self.atten = MultiHeadAttention(conf, atten_type)
        self.norm2 = RMS_Norm(conf.embedding_dim, conf.eps)
        if conf.use_moe:
            self.ffn = MoE(conf)
        else:
            self.ffn = FFN(conf)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor,
        window: torch.Tensor,
    ):
        x = x + self.atten.forward(
            self.norm1.forward(x), freq_cis, start_pos, mask, window
        )
        if self.conf.use_moe:
            rx, scores = self.ffn.forward(self.norm2.forward(x))
            x = x + rx
            return x, scores
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x


# Transformer Language Model
# ---------------------------------------
class TransformerLM(nn.Module):
    @staticmethod
    def get_transformer_config(
        maxlen: int,
        embedding_dim: int,
        num_heads: int,
        n_layers: int,
        inter_dim: int,
        window_size: int = 0,
        use_moe: bool = False,
        n_experts: int | None = None,
        expert_inter_dim: int | None = None,
        active_experts: int | None = None,
        kv_heads: int | None = None,
        mla: bool = False,
        kv_lora_rank: int | None = None,
        qk_rope_dim: int | None = None,
        qk_nope_dim: int | None = None,
        v_dim: int | None = None,
        base: int = 10000,
        eps: float = 1e-6,
        atten_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        flash: bool = False,
        atten_bias: bool = False,
        ffn_bias: bool = False,
        atten_types: list[AttentionMask] = [AttentionMask.Global],
    ):
        conf = ModelConfig(
            maxlen=maxlen,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            n_layers=n_layers,
            inter_dim=inter_dim,
            window_size=window_size,
            use_moe=use_moe,
            n_experts=n_experts,
            expert_inter_dim=expert_inter_dim,
            active_experts=active_experts,
            kv_heads=kv_heads,
            mla=mla,
            kv_lora_rank=kv_lora_rank,
            qk_rope_dim=qk_rope_dim,
            qk_nope_dim=qk_nope_dim,
            v_dim=v_dim,
            base=base,
            eps=eps,
            atten_dropout=atten_dropout,
            ffn_dropout=ffn_dropout,
            embedding_dropout=embedding_dropout,
            flash=flash,
            atten_bias=atten_bias,
            ffn_bias=ffn_bias,
            atten_types=atten_types,
        )
        return conf

    def __init__(self, conf: ModelConfig, vocab_size: int):
        super().__init__()
        self.conf = conf

        self.tokemb = nn.Embedding(vocab_size, conf.embedding_dim)

        self.dp = nn.Dropout(conf.embedding_dropout)

        self.blocks = nn.ModuleList()
        for i in range(conf.n_layers):
            at = conf.atten_types[i % len(conf.atten_types)]
            self.blocks.append(Block(conf, at))

        self.out_norm = RMS_Norm(conf.embedding_dim, conf.eps)
        self.logits = nn.Linear(conf.embedding_dim, vocab_size, False)

        self.tokemb.weight = self.logits.weight
        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / sqrt(2 * conf.n_layers), mean=0.0)

        if conf.mla:
            self.register_buffer(
                "freq_cis",
                rope(conf.qk_rope_dim, conf.maxlen * 2, conf.base),
                False,
            )
        else:
            self.register_buffer(
                "freq_cis",
                rope(conf.embedding_dim // conf.num_heads, conf.maxlen * 2, conf.base),
                False,
            )
        self.register_buffer(
            "mask", torch.full((conf.maxlen, conf.maxlen), -float("inf")).triu(1), False
        )
        self.register_buffer(
            "window",
            torch.full((conf.maxlen, conf.maxlen), -float("inf")).tril(
                -conf.window_size
            ),
            False,
        )

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02, mean=0.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02, mean=0.0)
        elif isinstance(module, RMS_Norm):
            nn.init.ones_(module.weight)

    def total_num_of_params(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        _, T = x.shape

        freq_cis = self.freq_cis[:T].to(x.device)
        emb = self.tokemb.forward(x)

        block = self.dp.forward(emb)

        mask = self.mask[:T, :T].to(x.device)
        window = self.window[:T, :T].to(x.device)

        if self.conf.use_moe:
            router_scores = None

        for b in self.blocks:
            if self.conf.use_moe:
                block, scores = b.forward(block, freq_cis, 0, mask, mask + window)
                router_scores = (
                    (scores,) if router_scores is None else (scores,) + router_scores
                )
            else:
                block = b.forward(block, freq_cis, 0, mask, mask + window)

        out_norm = self.out_norm.forward(block)
        logits = self.logits.forward(out_norm)

        r_logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = F.cross_entropy(r_logits, y)
        if self.conf.use_moe:
            aux_loss = load_balance_loss(router_scores, self.conf)
            loss += aux_loss
        return logits, loss

    def forward_infrence(self, x: torch.Tensor, start_pos: int = 0):
        _, T = x.shape
        freq_cis = self.freq_cis[start_pos : start_pos + T].to(x.device)
        emb = self.tokemb.forward(x)
        block = self.dp.forward(emb)

        mask = self.mask[start_pos : start_pos + T, start_pos : start_pos + T].to(
            x.device
        )
        window = self.window[start_pos : start_pos + T, start_pos : start_pos + T].to(
            x.device
        )
        if T > 1:
            mask = torch.hstack((torch.zeros((T, start_pos), device=x.device), mask))
            window = torch.hstack(
                (torch.zeros((T, start_pos), device=x.device), window)
            )

        for b in self.blocks:
            if self.conf.use_moe:
                block, _ = b.forward(block, freq_cis, start_pos, mask, mask + window)
            else:
                block = b.forward(block, freq_cis, start_pos, mask, mask + window)

        out_norm = self.out_norm.forward(block)
        logits = self.logits.forward(out_norm)
        return logits

    @torch.no_grad()
    def generate(
        self,
        device,
        pad_token: int,
        stop_tokens_: list[int],
        inp: list[list[int]],
        tokens_to_generate: int,
        temp: float,
        topp: float,
    ):
        B = len(inp)
        min_prompt_len = min(len(i) for i in inp)
        max_prompt_len = max(len(i) for i in inp)
        total_len = min(self.conf.maxlen, max_prompt_len + tokens_to_generate)
        tokens = torch.full((B, total_len), pad_token, dtype=torch.long, device=device)

        for block in self.blocks:
            if self.conf.mla is False:
                cache_device = block.atten.query.weight.device
                head_dim = block.atten.head_dim
                block.atten.cache = KVCache(
                    B,
                    self.conf.maxlen,
                    self.conf.kv_heads,
                    head_dim,
                    head_dim,
                    cache_device,
                )
            else:
                cache_device = block.atten.query.weight.device
                k_dim = block.atten.qk_head_dim
                block.atten.cache = KVCache(
                    B,
                    self.conf.maxlen,
                    self.conf.num_heads,
                    k_dim,
                    self.conf.v_dim,
                    cache_device,
                )

        for k, v in enumerate(inp):
            tokens[k, : len(v)] = torch.tensor(v, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * B, dtype=torch.bool, device=device)
        input_text_mask = tokens != pad_token
        if min_prompt_len == total_len:
            logits = self.forward_infrence(tokens)

        stop_tokens = torch.tensor(stop_tokens_, dtype=torch.long, device=device)
        for cur_pos in tqdm(range(min_prompt_len, total_len), "generating"):
            logits = self.forward_infrence(
                tokens[:, prev_pos:cur_pos], start_pos=prev_pos
            )
            if temp > 0:
                logits = logits[:, -1] / temp
                probs = F.softmax(logits, dim=-1)
                next_token = top_p(probs, topp)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & torch.isin(
                next_token, stop_tokens
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        return tokens.tolist()

    def get_optimizer(
        self,
        muon_lr: float = 0.02,
        adam_lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        betas=(0.9, 0.97),
        fused: bool = False,
    ):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        hidden_matrix_param = {
            p for n, p in param_dict.items() if p.ndim >= 2 and "tokemb" not in n
        }
        embed_param = [p for n, p in param_dict.items() if "tokemb" in n]
        scaler_param = [p for _, p in param_dict.items() if p.ndim < 2]
        head_param = [p for n, p in param_dict.items() if "logits" in n]

        adamw = torch.optim.AdamW(
            embed_param + scaler_param + head_param,
            lr=adam_lr,
            betas=betas,
            weight_decay=0,
            fused=fused,
        )
        muon = Muon(
            hidden_matrix_param,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        return [adamw, muon]
