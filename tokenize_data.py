import os
import datasets
import numpy as np
from tqdm import tqdm

from utils import save
from tokenizer import Tokenizer

from multiprocessing import Pool
from config import DataConfig

if __name__ == "__main__":
    cfg = DataConfig()
    train_data = datasets.load_dataset(
        cfg.dataset_path_huggingface, cfg.dataset_sub_set, split="train", streaming=True
    )

    tok = Tokenizer()
    tok.load()

    def tokenize(token):
        token = token["text"]
        data = [tok.special_token["<|endoftext|>"]]
        data.extend(tok.encode(token))
        return data

    nproc = max(1, os.cpu_count() // 2)
    print(nproc)
    with Pool(nproc) as p:
        precessed_size = 0
        shard_pos = 0
        bar = None
        c_data = np.empty((cfg.encoded_dataset_shard_size,), dtype=np.uint32)
        for token in p.imap(tokenize, train_data, chunksize=16):
            if precessed_size + len(token) < cfg.encoded_dataset_shard_size:
                c_data[precessed_size : precessed_size + len(token)] = np.array(token)
                precessed_size += len(token)
                if bar is None:
                    bar = tqdm(
                        total=cfg.encoded_dataset_shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_pos}",
                    )
                bar.update(len(token))
            else:
                shard_type = "val" if shard_pos == 0 else "train"
                reminder = cfg.encoded_dataset_shard_size - precessed_size
                bar.update(reminder)
                c_data[precessed_size : precessed_size + reminder] = np.array(
                    token[:reminder]
                )
                save(c_data, cfg.data_file_name, f"{shard_type}_{shard_pos}.pt")
                shard_pos += 1
                bar = None
                c_data[0 : len(token) - reminder] = np.array(token[reminder:])
                precessed_size = len(token) - reminder
    if precessed_size != 0:
        shard_type = "val" if shard_pos == 0 else "train"
        save(c_data[:precessed_size], cfg.data_file_name, f"{shard_type}_{shard_pos}.pt")
