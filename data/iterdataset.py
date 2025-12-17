import os
import torch
from utils import load


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
            self.idx = self.rank * self.maxlen
            self.shard_i = (self.shard_i + 1) % len(self.shards)
            self.data = load(
                self.shard_file, self.shards[self.shard_i], weights_only=False
            ).astype("int32")

        return x, y

    def state_dict(self):
        return {"shard_i": self.shard_i, "idx": self.idx}

    def load_state_dict(self, state_dict: dict):
        self.shard_i = state_dict["shard_i"]
        self.idx = state_dict["idx"]
        self.data = load(
            self.shard_file, self.shards[self.shard_i], weights_only=False
        ).astype("int32")
