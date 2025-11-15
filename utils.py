import os
import torch
from enum import Enum
from math import pi, cos
from dataclasses import dataclass


class TextDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, shard_file: str, maxlen: int, shard: str, rank: int, word_size: int
    ):
        super().__init__()
        self.rank = rank
        self.world_size = word_size
        self.shard_file = shard_file

        shards = os.listdir(shard_file)
        shards = sorted(shards)
        shards = [i for i in shards if shard in i]
        self.shards = shards
        self.maxlen = maxlen

    def __iter__(self):
        return TextDatasetIter(
            self.shard_file, self.shards, self.maxlen, self.rank, self.world_size
        )


class TextDatasetIter:
    def __init__(
        self,
        shard_file: str,
        shards: list[str],
        maxlen: int,
        rank: int,
        world_size: int,
    ):
        self.rank = rank
        self.world_size = world_size

        self.shard_file = shard_file
        self.shards = shards
        self.shard_i = 0

        self.maxlen = maxlen
        self.data = load(
            self.shard_file, self.shards[self.shard_i], weights_only=False
        ).astype("int32")

        self.idx = rank * maxlen

    def __iter__(self):
        return self

    def __next__(self):
        start = self.idx
        end = start + self.maxlen

        token = self.data[start : end + 1]
        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token, dtype=torch.long)

        x = token[:-1]
        y = token[1:]

        self.idx += self.maxlen * self.world_size
        if self.idx + (self.maxlen * self.world_size + 1) > len(self.data):
            print_master(self.idx)
            self.idx = self.rank * self.maxlen
            self.shard_i = (self.shard_i + 1) % len(self.shards)
            print_master(self.shard_i)
            self.data = load(
                self.shard_file, self.shards[self.shard_i], weights_only=False
            ).astype("int32")

        return x, y


@torch.no_grad()
def est_loss(
    model: torch.nn.Module,
    val_iter,
    eval_steps: int,
    device: str,
    device_type: str,
    is_cuda: bool,
    use_autocast: bool,
    dtype: torch.dtype,
):
    model.eval()
    losses = torch.zeros(eval_steps, device=device)
    for i in range(eval_steps):
        x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=is_cuda and use_autocast,
        ):
            _, loss = model.forward(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean()


def get_lr(iter: int, steps: int, lr: float, min_lr: float, warm_up: int):
    if iter < warm_up:
        return lr * iter / warm_up
    if iter > steps:
        return min_lr
    d_ratio = (iter - warm_up) / (steps - warm_up)
    coff = 0.5 * (1 + cos(pi * d_ratio))
    return min_lr + coff * (lr - min_lr)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def save(obj: object, file: str, name: str = "ckp.pt"):
    os.makedirs(file, exist_ok=True)
    pt = os.path.join(file, name)
    torch.save(obj, pt)


def load(file: str, name: str = "ckp.pt", weights_only: bool = True, map_location=None):
    pt = os.path.join(file, name)
    return torch.load(pt, weights_only=weights_only, map_location=map_location)


# This code is not used any where.
# I use it for calculating the total token len in the dataset so can set the epochs.
def get_encoded_data_token_len(file: str):
    shards = os.listdir(file)
    shards = sorted(shards)
    shards = [i for i in shards if "train" in i]
    print(shards)
    size = 0
    for shard in shards:
        print(shard)
        data = load(file, shard, False)
        size += len(data)
        print(size)
    return size


def top_p(probs: torch.Tensor, p: float):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs - sorted_probs > p
    sorted_probs[mask] = 0
    sorted_probs.div_(sorted_probs.sum(-1, keepdim=True))
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(sorted_indices, -1, next_token)
    return next_token


def print_master(inp):
    if int(os.environ.get("RANK", 0)) == 0:
        print(inp)


class AttentionMask(Enum):
    Local = 1
    Global = 2


@dataclass
class ModelConfig:
    # Common parameters
    maxlen: int
    embedding_dim: int
    num_heads: int
    n_layers: int
    inter_dim: int
    window_size: int
    # moe
    use_moe: bool
    n_experts: int | None
    expert_inter_dim: int | None
    active_experts: int | None
    # gqa
    kv_heads: int | None
    # mla
    mla: bool
    kv_lora_rank: int | None
    qk_rope_dim: int | None
    qk_nope_dim: int | None
    v_dim: int | None
    # rope
    base: int
    eps: float
    # dropout
    atten_dropout: float
    ffn_dropout: float
    embedding_dropout: float
    # flash attention
    flash: bool
    # use bias
    atten_bias: bool
    ffn_bias: bool
    # attention types
    atten_types: list[AttentionMask]
