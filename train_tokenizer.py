import datasets
from tokenizer import Tokenizer
from configuration import (
    dataset_path_huggingface,
    dataset_sub_set,
    tokenizer_train_shard_size,
    trust_remote_code,
)

if __name__ == "__main__":
    data_sets = datasets.load_dataset(
        dataset_path_huggingface,
        dataset_sub_set,
        split="train",
        trust_remote_code=trust_remote_code,
    )
    print(data_sets)

    inp = int(input("Vocab size: "))
    tok = Tokenizer(inp)

    trained = False
    for shard_i in range(tokenizer_train_shard_size):
        chunks = data_sets["text"].shard(tokenizer_train_shard_size, shard_i)
        chunk_str = "\n".join(chunks)
        trained = tok.train(chunk_str)
        print(f"Shard {shard_i} trained")
        if trained:
            break
    tok.regester_special_token({"<pad>": 0, "<|endoftext|>": 1})
    print(tok.vocab)
    print(tok.special_token)
    tok.save()
