import datasets
from tokenizer import Tokenizer
from configuration import (
    dataset_path_huggingface,
    dataset_sub_set,
    tokenizer_train_char_size,
    trust_remote_code,
)

if __name__ == "__main__":
    data_sets = datasets.load_dataset(
        dataset_path_huggingface,
        dataset_sub_set,
        split="train",
        trust_remote_code=trust_remote_code,
        streaming=True,
    )
    print(data_sets)

    inp = int(input("Vocab size: "))
    tok = Tokenizer(inp)

    idx = 0
    trained = False
    while not trained:
        chr_len = 0
        text_data = ""
        for text in data_sets:
            text_data += text["text"] + "\n"
            chr_len = len(text_data)
            print(chr_len)
            if chr_len >= tokenizer_train_char_size:
                break
        trained = tok.train(text_data)
        idx += 1
        print(f"trained on {tokenizer_train_char_size * idx} char")
    tok.regester_special_token({"<pad>": 0, "<|endoftext|>": 1})
    print(tok.vocab)
    print(tok.special_token)
    tok.save()
