import os

import torch
import torch.distributed as dist

from torch import GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time

from tokenizer import Tokenizer
from model import TransformerLM

from utils import TextDataset, ModelConfig, save, load, est_loss, get_lr, print_master
from config_args import sft_train_args
from flops import transformer_flops


def main():
    file_args = sft_train_args()
    steps = file_args.steps
    eval_rate = file_args.eval_rate
    eval_steps = file_args.eval_steps
    save_rate = file_args.save_rate
    lr = file_args.lr
    min_lr = file_args.min_lr
    weight_decay = file_args.weight_decay
    betas = (file_args.beta1, file_args.beta2)
    warm_up = file_args.warm_up
    total_batch_size = file_args.total_batch_size
    batch_size = file_args.batch_size
    compile_model = file_args.compile_model
    backend = file_args.backend
    save_file_name = file_args.save_file_name
    data_file_name = file_args.data_file_name
    tokenizer_file_name = file_args.tokenizer_file_name
    use_autocast = file_args.use_autocast
    load_mistral_tokenizer = file_args.load_mistral_tokenizer
    promissed_flops = file_args.promissed_flops
    dtype = file_args.dtype
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]

    is_cuda = torch.cuda.is_available()

    # init multi-gpu training
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert is_cuda, "Need gpu for ddp training."
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        device_type = "cuda"
        master_process = rank == 0
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = f"cuda:{local_rank}" if is_cuda else "cpu"
        device_type = "cuda" if is_cuda else "cpu"
        master_process = True
        if is_cuda:
            torch.cuda.set_device(device)

    # loading tokenizer for the vocab_size
    tok = Tokenizer()
    if load_mistral_tokenizer:
        tok.load_mistral_tokenizer(tokenizer_file_name)
    else:
        tok.load(tokenizer_file_name)

    # Loading the model checkpoint and config
    checkpoint = load(save_file_name, weights_only=True)
    model_conf: ModelConfig = load(save_file_name, "model_config.pt", False)

    ## model flops
    flops_per_token = transformer_flops(
        vocab_size=tok.vocab_size,
        maxlen=model_conf.maxlen,
        embedding_dim=model_conf.embedding_dim,
        inter_dim=model_conf.inter_dim,
        num_heads=model_conf.num_heads,
        n_layers=model_conf.n_layers,
        qk_rope_dim=model_conf.qk_rope_dim,
        qk_nope_dim=model_conf.qk_nope_dim,
        kv_rank=model_conf.kv_lora_rank,
        v_dim=model_conf.v_dim,
    )

    # get grad accum
    assert total_batch_size % (batch_size * model_conf.maxlen * world_size) == 0, (
        "total_batch_size has divisible by batch_size * maxlen * world_size"
    )
    grad_accum = total_batch_size // (batch_size * model_conf.maxlen * world_size)

    # making the model
    model = TransformerLM(model_conf, tok.vocab_size)
    model.to(device)
    model: TransformerLM = torch.compile(
        model, backend=backend, disable=not compile_model
    )
    model.load_state_dict(checkpoint["model"])

    # init ddp
    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    model_params = raw_model.total_num_of_params()
    print_master(model)
    print_master(model_params)
    print_master(f"Number of devices:{world_size}")

    # making the optimizer
    optim = raw_model.get_optimizer(lr, weight_decay, betas)

    # datasets
    train_dataset = TextDataset(
        data_file_name, model_conf.maxlen, "train", rank, world_size
    )
    val_dataset = TextDataset(
        data_file_name, model_conf.maxlen, "val", rank, world_size
    )

    # dataloader
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_data_iter = iter(train_data)
    val_data_iter = iter(val_data)

    # amp scaler
    use_scaler = use_autocast and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    x, y = next(train_data_iter)
    x, y = x.to(device), y.to(device)
    t0 = time()

    # step loop
    for i in range(1, steps + 1):
        ploss = 0.0
        n_lr = get_lr(i, steps, lr, min_lr, warm_up)
        for param_group in optim.param_groups:
            param_group["lr"] = n_lr

        # Eval
        if i % eval_rate == 0 or i == steps:
            e_loss = est_loss(
                model,
                val_data_iter,
                eval_steps,
                device,
                device_type,
                is_cuda,
                use_autocast,
                dtype,
            )
            if ddp:
                dist.all_reduce(e_loss, dist.ReduceOp.AVG)
            print_master(f"step: {i}/{steps}, val_loss: {e_loss.item():.8f}")

        # sync after eval
        if is_cuda:
            torch.cuda.synchronize()

        # Train
        for grad_i in range(grad_accum):
            no_sync_enable = grad_i < grad_accum - 1
            if no_sync_enable and ddp:
                with model.no_sync():
                    with torch.autocast(
                        device_type=device_type,
                        dtype=dtype,
                        enabled=is_cuda and use_autocast,
                    ):
                        _, loss = model.forward(x, y)
                    loss = loss / grad_accum
                    scaler.scale(loss).backward()
            else:
                with torch.autocast(
                    device_type=device_type,
                    dtype=dtype,
                    enabled=is_cuda and use_autocast,
                ):
                    _, loss = model.forward(x, y)
                loss = loss / grad_accum
                scaler.scale(loss).backward()
            x, y = next(train_data_iter)
            x, y = x.to(device), y.to(device)
            ploss += loss.detach()
        if ddp:
            dist.all_reduce(ploss, dist.ReduceOp.AVG)

        # optim update
        scaler.unscale_(optim)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        # sync after train
        if is_cuda:
            torch.cuda.synchronize()

        t1 = time()
        dt = t1 - t0
        t0 = t1
        tok_per_sec = (batch_size * model_conf.maxlen * grad_accum * world_size) / dt
        flops_achived = flops_per_token * (batch_size * grad_accum * world_size) / dt
        mfu = (flops_achived / promissed_flops) * 100
        print_master(
            f"step: {i}/{steps} | lr: {n_lr:.8f} | loss: {ploss:.8f} | norm: {norm.item():.8f} | time: {dt:.2f}sec | tok/sec: {tok_per_sec:.2f} | mfu: {mfu:.2f}%"
        )
        if (save_rate % i == 0 or i == steps) and master_process:
            print_master("saving checkpoint...")
            checkpoint = {"model": raw_model.state_dict(), "optim": optim.state_dict()}

            save(checkpoint, f"{save_file_name}_inst")
            save(model_conf, f"{save_file_name}_inst", "model_config.pt")
            print_master("saved checkpoint")

    # end dist
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
