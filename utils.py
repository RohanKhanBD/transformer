import os
import torch
from enum import Enum
from math import pi, cos
from dataclasses import dataclass


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        shard_file: str,
        maxlen: int,
        shard: str,
    ):
        self.shard_file = shard_file

        shards = os.listdir(shard_file)
        shards = sorted(shards)
        shards = [i for i in shards if shard in i]
        self.shards = shards
        self.shard_i = 0
        self.data = load(shard_file, self.shards[self.shard_i], False).astype("int32")

        self.maxlen = maxlen

    def __getitem__(self, idx):
        start_idx = idx * self.maxlen
        end_idx = start_idx + self.maxlen
        if end_idx + 1 > len(self.data):
            print("end of data")
            print(f"current pos:{idx}")
            self.shard_i = (self.shard_i + 1) % len(self.shards)
            print(f"current shard:{self.shard_i}")
            print(f"shards left:{len(self.shards) - self.shard_i}")
            self.data = load(self.shard_file, self.shards[self.shard_i], False).astype(
                "int32"
            )
            raise StopIteration
        tokens = self.data[start_idx : end_idx + 1]
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

    def __len__(self):
        return len(self.data) - self.maxlen


@torch.no_grad()
def est_loss(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    val_iter,
    e: int,
    device: torch.device,
):
    model.eval()
    losses = torch.zeros(e)
    for i in range(e):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dataloader)
            x, y = next(val_iter)
            x, y = x.to(device), y.to(device)
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
