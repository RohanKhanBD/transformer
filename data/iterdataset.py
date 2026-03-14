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

        self.data = self._load_shard(self.shard_i)
        self.idx = rank * maxlen

    def __iter__(self):
        return self

    def _load_shard(self, shard_i: int):
        return load(self.shard_file, self.shards[shard_i], weights_only=False).astype(
            "int64"
        )

    def _advance_shard(self):
        self.shard_i = (self.shard_i + 1) % len(self.shards)
        self.data = self._load_shard(self.shard_i)
        self.idx = self.rank * self.maxlen

    def __next__(self):
        while True:
            start = self.idx
            end = start + self.maxlen
            if end < len(self.data):
                break
            self._advance_shard()

        x = torch.from_numpy(self.data[start:end]).to(torch.long)
        y = torch.from_numpy(self.data[start + 1 : end + 1]).to(torch.long)

        return x, y

    def state_dict(self):
        return {"shard_i": self.shard_i, "idx": self.idx}

    def load_state_dict(self, state_dict: dict):
        self.shard_i = state_dict["shard_i"]
        self.idx = state_dict["idx"]
        self.data = load(
            self.shard_file, self.shards[self.shard_i], weights_only=False
        ).astype("int32")
