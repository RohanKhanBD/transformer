import os
import datasets
import numpy as np
from tqdm import tqdm

from utils import save
from tokenizer import Tokenizer

from multiprocessing import Pool
from configuration import (
    dataset_path_huggingface,
    dataset_sub_set,
    encoded_dataset_shard_size,
    non_encoded_dataset_shard_size,
)

train_data = datasets.load_dataset(
    dataset_path_huggingface, dataset_sub_set, split="train"
)

tok = Tokenizer()
tok.load()


def tokenize(token):
    data = [tok.special_token["<|endoftext|>"]]
    data.extend(tok.encode(token))
    return data


if __name__ == "__main__":
    nproc = max(1, os.cpu_count() // 2)
    print(nproc)
    with Pool(nproc) as p:
        precessed_size = 0
        shard_pos = 0
        bar = None
        c_data = np.empty((encoded_dataset_shard_size,), dtype=np.uint32)
        for i in range(non_encoded_dataset_shard_size):
            chunks = train_data.shard(non_encoded_dataset_shard_size, i)["text"]
            for token in p.imap(tokenize, chunks, chunksize=16):
                if precessed_size + len(token) < encoded_dataset_shard_size:
                    c_data[precessed_size : precessed_size + len(token)] = np.array(
                        token
                    )
                    precessed_size += len(token)
                    if bar is None:
                        bar = tqdm(
                            total=encoded_dataset_shard_size,
                            unit="tokens",
                            desc=f"Shard {shard_pos}",
                        )
                    bar.update(len(token))
                else:
                    shard_type = "val" if shard_pos == 0 else "train"
                    reminder = encoded_dataset_shard_size - precessed_size
                    bar.update(reminder)
                    c_data[precessed_size : precessed_size + reminder] = np.array(
                        token[:reminder]
                    )
                    save(c_data, "encoded_data", f"{shard_type}_{shard_pos}.pt")
                    shard_pos += 1
                    bar = None
                    c_data[0 : len(token) - reminder] = np.array(token[reminder:])
                    precessed_size = len(token) - reminder
        if precessed_size != 0:
            shard_type = "val" if shard_pos == 0 else "train"
            save(
                c_data[:precessed_size], "encoded_data", f"{shard_type}_{shard_pos}.pt"
            )
