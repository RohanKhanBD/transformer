import datasets

from data import pre_process
from tokenizer import Tokenizer
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
        for tab in messages["messages"]:
            role = f"[INST]{tab['role']}[/INST]"
            content = f"{tab['content']}"
            full_mes = f"{role}\n\n{content}\n\n"
            mes += full_mes
        mes += "</s>"
        return tok.encode(mes)

    pre_process(
        encoded_dataset_shard_size, sft_dataset, data_file_name, tokenize_messages
    )
