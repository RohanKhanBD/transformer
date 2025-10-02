import datasets
from tokenizer import Tokenizer
from config_args import tokenize_args

if __name__ == "__main__":
    file_args = tokenize_args(train_tokenizer=True)
    dataset_path_huggingface = file_args.dataset_path_huggingface
    dataset_sub_set = file_args.dataset_sub_set
    tokenizer_train_shard_size = file_args.tokenizer_train_shard_size
    trust_remote_code = file_args.trust_remote_code

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

    trained = False
    while not trained:
        text_point = ""
        for text in data_sets:
            text_point += text["text"] + "\n"
            if len(text_point) > tokenizer_train_shard_size:
                break
        trained = tok.train(text_point)
        print(trained)
    tok.regester_special_token({"<pad>": 0, "<|endoftext|>": 1})
    print(tok.vocab)
    print(tok.special_token)
    tok.save()
