import os

import torch
import torch.distributed as dist

from torch import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time

from tokenizer import Tokenizer
from model import TransformerLM
from utils import TextDataset, AttentionMask, load, save, est_loss, get_lr, set_seed

from config_args import train_args


def print_master(inp, master):
    if master:
        print(inp)


def main():
    file_args = train_args()
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
    seed = file_args.seed
    compile_model = file_args.compile_model
    backend = file_args.backend
    save_file_name = file_args.save_file_name
    data_file_name = file_args.data_file_name
    tokenizer_file_name = file_args.tokenizer_file_name
    use_autocast = file_args.use_autocast
    load_mistral_tokenizer = file_args.load_mistral_tokenizer
    dtype = file_args.dtype
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]

    writer = SummaryWriter()
    tok = Tokenizer()
    if load_mistral_tokenizer:
        tok.load_mistral_tokenizer(tokenizer_file_name)
    else:
        tok.load(tokenizer_file_name)
    is_cuda = torch.cuda.is_available()

    # Distributed Data Parallel init
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
        else:
            torch.cpu.set_device(device)

    # model init
    set_seed(seed + rank)
    model_conf = TransformerLM.get_transformer_config(
        maxlen=512,
        embedding_dim=256,
        num_heads=8,
        n_layers=8,
        inter_dim=256 + 128,
        window_size=384,
        mla=True,
        kv_lora_rank=64,
        qk_rope_dim=64,
        qk_nope_dim=128,
        v_dim=128,
        flash=is_cuda,
        atten_types=[AttentionMask.Local],
    )
    assert total_batch_size % (batch_size * model_conf.maxlen * world_size) == 0, (
        "total_batch_size has divisible by batch_size * maxlen * world_size"
    )
    grad_ecum = total_batch_size // (batch_size * model_conf.maxlen * world_size)
    model = TransformerLM(model_conf, tok.vocab_size)
    model.to(device)
    model: TransformerLM = torch.compile(
        model, backend=backend, disable=not compile_model
    )
    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    print_master(model, master_process)
    print_master(raw_model.total_num_of_params(), master_process)
    print_master(f"Number of devices:{world_size}", master_process)
    # optimizer
    optim = raw_model.get_optimizer(lr, weight_decay, betas, fused=is_cuda)

    # dataset
    train_dataset = TextDataset(data_file_name, model_conf.maxlen, "train")
    val_dataset = TextDataset(data_file_name, model_conf.maxlen, "val")

    # sampler
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # data loader
    train_data = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler
    )
    val_data = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler
    )

    try:
        checkpoint = load(save_file_name)
        data_state = load(save_file_name, "data_state_and_training_info.pt", False)
        print_master(data_state, master_process)
        train_i = data_state["step"]
        val_i = data_state["val_i"]

        train_data = data_state["train_data"]
        val_data = data_state["val_data"]

        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])

        print_master("loaded: True", master_process)
    except FileNotFoundError:
        train_i = 1
        val_i = 1
        print_master("loaded: False", master_process)

    # amp scaler
    use_scaler = use_autocast and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    train_data_iter = iter(train_data)
    val_data_iter = iter(val_data)
    x, y = next(train_data_iter)
    x, y = x.to(device), y.to(device)
    t0 = time()

    for i in range(train_i, steps + 1):
        ploss = 0.0
        n_lr = get_lr(i, steps, lr, min_lr, warm_up)
        for param_group in optim.param_groups:
            param_group["lr"] = n_lr
        # ------- Eval -------
        if i % eval_rate == 0 or i == steps:
            e_loss = est_loss(
                model,
                val_data,
                val_data_iter,
                eval_steps,
                device,
                device_type,
                is_cuda,
                use_autocast,
                dtype,
            )
            if ddp:
                dist.all_reduce(e_loss, op=dist.ReduceOp.AVG)
            if master_process:
                writer.add_scalar("loss/val", e_loss, val_i)
            val_i += 1
            print_master(
                f"step: {i}/{steps}, val_loss: {e_loss.item():.8f}", master_process
            )
        # ------- Train -------
        for grad_i in range(grad_ecum):
            no_sync_enable = grad_i < grad_ecum - 1
            if no_sync_enable and ddp:
                with model.no_sync():
                    with torch.autocast(
                        device_type=device_type,
                        dtype=dtype,
                        enabled=is_cuda and use_autocast,
                    ):
                        _, loss = model.forward(x, y)
                    loss = loss / grad_ecum
                    scaler.scale(loss).backward()
            else:
                with torch.autocast(
                    device_type=device_type,
                    dtype=dtype,
                    enabled=is_cuda and use_autocast,
                ):
                    _, loss = model.forward(x, y)
                loss = loss / grad_ecum
                scaler.scale(loss).backward()
            try:
                x, y = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_data)
                x, y = next(train_data_iter)
            x, y = x.to(device), y.to(device)
            ploss += loss.detach()
        if ddp:
            dist.all_reduce(ploss, op=dist.ReduceOp.AVG)

        scaler.unscale_(optim)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        if is_cuda:
            torch.cuda.synchronize()
        else:
            torch.cpu.synchronize()

        t1 = time()
        dt = t1 - t0
        t0 = t1
        tok_per_sec = (batch_size * model_conf.maxlen * grad_ecum * world_size) / dt
        print_master(
            f"step: {i}/{steps} | lr: {n_lr:.8f} | loss: {ploss:.8f} | norm: {norm.item():.8f} | time: {dt:.2f}sec | tok/sec: {tok_per_sec:.2f}",
            master_process,
        )
        if master_process:
            writer.add_scalar("loss/train", ploss, i)
            writer.add_scalar("model/lr", n_lr, i)
            writer.add_scalar("model/norm", norm.item(), i)
            writer.flush()
        # ------- Save -------
        if (i % save_rate == 0 or i == steps) and master_process:
            print_master("saving checkpoint...", master_process)
            checkpoint = {"model": raw_model.state_dict(), "optim": optim.state_dict()}

            data_state = {
                "train_data": train_data,
                "val_data": val_data,
                "step": i + 1,
                "val_i": val_i,
            }
            save(checkpoint, save_file_name)
            save(model_conf, save_file_name, "model_config.pt")
            save(data_state, save_file_name, "data_state_and_training_info.pt")
            print_master("saved checkpoint", master_process)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
