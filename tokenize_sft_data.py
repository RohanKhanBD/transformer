import os
import datasets
import numpy as np

from tqdm import tqdm
from utils import save
from tokenizer import Tokenizer
from multiprocessing import Pool

from config_args import sft_args

if __name__ == "__main__":
    file_args = sft_args()
    sft_dataset_path_huggingface = file_args.sft_dataset_path_huggingface
    sft_dataset_sub_set = file_args.sft_dataset_sub_set
    tokenizer_file_name = file_args.tokenizer_file_name
    data_file_name = file_args.data_file_name
    encoded_dataset_shard_size = file_args.encoded_dataset_shard_size
    load_mistral_tokenizer = file_args.load_mistral_tokenizer

    tok = Tokenizer()
    if load_mistral_tokenizer:
        tok.load_mistral_tokenizer(tokenizer_file_name)
    else:
        tok.load(tokenizer_file_name)

    sft_dataset = datasets.load_dataset(
        sft_dataset_path_huggingface, sft_dataset_sub_set, split="train", streaming=True
    )
    print(sft_dataset)

    def tokenize_messages(messages):
        mes = "<s>"
        # TODO: Update tokenizer to have user start and end tokens and assistent start and end tokens.
        for tab in messages["messages"]:
            role = f"[INST]{tab['role']}[/INST]"
            content = f"{tab['content']}"
            full_mes = f"{role}\n\n{content}\n\n"
            mes += full_mes
        mes += "</s>"
        return tok.encode(mes)

    nproc = max(1, os.cpu_count() // 2)
    with Pool(nproc) as p:
        c_data = np.empty((encoded_dataset_shard_size,), dtype=np.uint32)
        processed_size = 0
        shard = 0
        bar = None
        for tokens in p.imap(tokenize_messages, sft_dataset, chunksize=16):
            if processed_size + len(tokens) < encoded_dataset_shard_size:
                c_data[processed_size : processed_size + len(tokens)] = np.array(tokens)
                processed_size += len(tokens)
                if bar is None:
                    bar = tqdm(
                        total=encoded_dataset_shard_size,
                        desc=f"Shard {shard}",
                        unit="tokens",
                    )
                bar.update(len(tokens))
            else:
                shard_type = "val" if shard == 0 else "train"
                remainder = encoded_dataset_shard_size - processed_size
                bar.update(remainder)
                c_data[processed_size : processed_size + remainder] = np.array(
                    tokens[:remainder]
                )
                save(c_data, data_file_name, f"{shard_type}_{shard}.pt")
                shard += 1
                bar = None
                c_data[0 : len(tokens) - remainder] = np.array(tokens[remainder:])
                processed_size = len(tokens) - remainder
    if processed_size != 0:
        shard_type = "val" if shard == 0 else "train"
        save(c_data[:processed_size], data_file_name, f"{shard_type}_{shard}.pt")
