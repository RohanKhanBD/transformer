from argparse import ArgumentParser


def tokenize_args(train_tokenizer=False):
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path_huggingface", type=str, default="HuggingFaceFW/fineweb-edu"
    )
    parser.add_argument("--dataset_sub_set", type=str, default="sample-10BT")

    if train_tokenizer:
        parser.add_argument("--tokenizer_train_shard_size", type=int, default=5_000_000)
        parser.add_argument("--trust_remote_code", type=bool, default=False)

    else:
        parser.add_argument("--data_file_name", type=str, default="encoded_data")
        parser.add_argument("--encoded_dataset_shard_size", type=int, default=int(1e8))

    args = parser.parse_args()
    return args


def generate_args():
    parser = ArgumentParser()
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--num_tokens_to_generate", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--save_file_name", type=str, default="nanogpt")
    args = parser.parse_args()
    return args


def train_args():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=18701)
    parser.add_argument("--eval_rate", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_rate", type=int, default=10)
    parser.add_argument("--warm_up", type=int, default=715)
    parser.add_argument("--total_batch_size", type=int, default=2**19)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min_lr", type=float, default=3e-3 * 0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.97)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--save_file_name", type=str, default="nanogpt")
    parser.add_argument("--data_file_name", type=str, default="encoded_data")
    parser.add_argument("--compile_model", type=bool, default=True)
    args = parser.parse_args()
    return args
