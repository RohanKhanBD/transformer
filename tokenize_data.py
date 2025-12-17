import datasets

from data import pre_process
from tokenizer import Tokenizer
from config_args import tokenize_args

if __name__ == "__main__":
    file_args = tokenize_args()
    data_file_name = file_args.data_file_name
    dataset_path_huggingface = file_args.dataset_path_huggingface
    dataset_sub_set = file_args.dataset_sub_set
    encoded_dataset_shard_size = file_args.encoded_dataset_shard_size
    tokenizer_file_name = file_args.tokenizer_file_name
    load_mistral_tokenizer = file_args.load_mistral_tokenizer

    train_data = datasets.load_dataset(
        dataset_path_huggingface, dataset_sub_set, split="train", streaming=True
    )

    tok = Tokenizer()
    if load_mistral_tokenizer:
        tok.load_mistral_tokenizer(tokenizer_file_name)
    else:
        tok.load(tokenizer_file_name)

    def tokenize(token):
        token = token["text"]
        data = [tok.special_token["<s>"]]
        data.extend(tok.encode(token))
        data.append(tok.special_token["</s>"])
        return data

    pre_process(encoded_dataset_shard_size, train_data, data_file_name, tokenize)
