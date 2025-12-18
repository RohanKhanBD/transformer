import os
import torch
from enum import Enum
from math import pi, cos
from dataclasses import dataclass


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
