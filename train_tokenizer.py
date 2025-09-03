import datasets
from tokenizer import Tokenizer
from config import DataConfig

if __name__ == "__main__":
    cfg = DataConfig()
    data_sets = datasets.load_dataset(
        cfg.dataset_path_huggingface,
        cfg.dataset_sub_set,
        split="train",
        trust_remote_code=cfg.trust_remote_code,
        streaming=True,
    )
    print(data_sets)

    inp = int(input("Vocab size: "))
    tok = Tokenizer(inp)

    trained = False
    while not trained:
        text_point = ""
        for text in data_sets:
            text_point += text["text"] + "\n"
            if len(text_point) > cfg.tokenizer_train_shard_size:
                break
        trained = tok.train(text_point)
        print(trained)
    tok.regester_special_token({"<pad>": 0, "<|endoftext|>": 1})
    print(tok.vocab)
    print(tok.special_token)
    tok.save()
