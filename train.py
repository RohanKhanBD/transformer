import os

import torch

from torch import GradScaler
from torch.distributed import ReduceOp as rop
from torch.utils.tensorboard import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time

from tokenizer import Tokenizer
from model import TransformerLM
from utils import (
    AttentionMask,
    load,
    save,
    get_lr,
    set_seed,
    print_master,
    muon_momentum,
)

from data import TextDataset
from config_args import train_args
from flops import transformer_flops
from training import (
    dist_init,
    kill_dist,
    model_loss,
    est_loss,
    all_reduce,
    sync,
    barrier,
)


def main():
    file_args = train_args()
    steps = file_args.steps
    eval_rate = file_args.eval_rate
    eval_steps = file_args.eval_steps
    save_rate = file_args.save_rate
    muon_lr = file_args.muon_lr
    adamw_lr = file_args.adamw_lr
    momentum = file_args.momentum
    min_muon_lr = file_args.min_muon_lr
    min_adamw_lr = file_args.min_adamw_lr
    min_momentum = file_args.min_momentum
    weight_decay = file_args.weight_decay
    betas = (file_args.beta1, file_args.beta2)
    warm_up = file_args.warm_up
    muon_warm_up = file_args.muon_warm_up
    muon_cooldown = file_args.muon_cooldown
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
    promissed_flops = file_args.promissed_flops
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
    rank, local_rank, world_size, device, device_type, ddp, master_process = dist_init(
        is_cuda
    )

    # model init
    set_seed(seed + rank)
    model_conf = TransformerLM.get_transformer_config(
        # full model
        maxlen=512,
        embedding_dim=512,
        num_heads=8,
        n_layers=8,
        inter_dim=256 + 128,
        window_size=256,
        # attention
        kv_heads=4,
        flash=is_cuda,
        # rope
        base=100000,
        # drop-out
        atten_dropout=0.1,
        ffn_dropout=0.1,
        embedding_dropout=0.1,
        atten_types=[
            AttentionMask.Global,
            AttentionMask.Local,
            AttentionMask.Local,
            AttentionMask.Local,
        ],
    )

    ## model flops
    flops_per_token = transformer_flops(
        vocab_size=tok.vocab_size,
        maxlen=model_conf.maxlen,
        embedding_dim=model_conf.embedding_dim,
        inter_dim=model_conf.inter_dim,
        num_heads=model_conf.num_heads,
        kv_heads=model_conf.kv_heads,
        n_layers=model_conf.n_layers,
        use_mla=model_conf.mla,
        qk_rope_dim=model_conf.qk_rope_dim,
        qk_nope_dim=model_conf.qk_nope_dim,
        kv_rank=model_conf.kv_lora_rank,
        v_dim=model_conf.v_dim,
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
    if os.path.exists(save_file_name):
        checkpoint = load(save_file_name)
        model.load_state_dict(checkpoint["model"])

    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    model_params = raw_model.total_num_of_params()
    print_master(model)
    print_master(model_params)
    print_master(f"Number of devices:{world_size}")
    # optimizer
    optim = raw_model.get_optimizer(
        muon_lr, adamw_lr, momentum, weight_decay, betas, fused=is_cuda
    )
    if os.path.exists(save_file_name):
        optim[0].load_state_dict(checkpoint["adamw"])
        optim[1].load_state_dict(checkpoint["muon"])

    # dataset
    train_dataset = TextDataset(
        data_file_name, model_conf.maxlen, "train", rank, world_size
    )
    val_dataset = TextDataset(
        data_file_name, model_conf.maxlen, "val", rank, world_size
    )

    # data loader
    train_data = StatefulDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_data = StatefulDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if os.path.exists(save_file_name):
        data_state = load(save_file_name, "training_info.pt", False)
        dataset_state = load(save_file_name, f"dataset_info_rank{rank}.pt", False)
        print_master(data_state)
        train_i = data_state["step"]
        val_i = data_state["val_i"]

        train_data.load_state_dict(dataset_state["train_data"])
        val_data.load_state_dict(dataset_state["val_data"])

        print_master("loaded: True")
    else:
        train_i = 1
        val_i = 1
        print_master("loaded: False")

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
        n_muon_lr = get_lr(i, steps, muon_lr, min_muon_lr, warm_up)
        n_adamw_lr = get_lr(i, steps, adamw_lr, min_adamw_lr, warm_up)
        n_momentum = muon_momentum(
            i, steps, muon_warm_up, muon_cooldown, min_momentum, momentum
        )
        for param_group in optim[0].param_groups:
            param_group["lr"] = n_adamw_lr
        for param_group in optim[1].param_groups:
            param_group["lr"] = n_muon_lr
            param_group["momentum"] = n_momentum
        # ------- Eval -------
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
            all_reduce(e_loss, rop.AVG, ddp)
            if master_process:
                writer.add_scalar("loss/val", e_loss, val_i)
            val_i += 1
            print_master(f"step: {i}/{steps}, val_loss: {e_loss.item():.8f}")
        sync(is_cuda)
        # ------- Train -------
        for grad_i in range(grad_ecum):
            no_sync_enable = grad_i < grad_ecum - 1
            if no_sync_enable and ddp:
                with model.no_sync():
                    loss = model_loss(
                        model, x, y, dtype, device_type, is_cuda, use_autocast
                    )
                    loss = loss / grad_ecum
                    scaler.scale(loss).backward()
            else:
                loss = model_loss(
                    model, x, y, dtype, device_type, is_cuda, use_autocast
                )
                loss = loss / grad_ecum
                scaler.scale(loss).backward()
            x, y = next(train_data_iter)
            x, y = x.to(device), y.to(device)
            ploss += loss.detach()
        all_reduce(ploss, rop.AVG, ddp)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optim:
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        sync(is_cuda)

        t1 = time()
        dt = t1 - t0
        t0 = t1
        tok_per_sec = (batch_size * model_conf.maxlen * grad_ecum * world_size) / dt
        flops_achived = flops_per_token * (batch_size * grad_ecum * world_size) / dt
        mfu = (flops_achived / promissed_flops) * 100
        print_master(
            f"step: {i}/{steps} | loss: {ploss:.8f} | time: {dt:.2f}sec | tok/sec: {tok_per_sec:.2f} | mfu: {mfu:.2f}%"
        )
        if master_process:
            writer.add_scalar("loss/train", ploss, i)
            writer.add_scalar("model/muon/lr", n_muon_lr, i)
            writer.add_scalar("model/adamw/lr", n_adamw_lr, i)
            writer.add_scalar("model/muon/momentum", n_momentum, i)
            writer.add_scalar("model/norm", norm.item(), i)
            writer.flush()
        # ------- Save -------
        if (i % save_rate == 0 or i == steps) and master_process:
            print_master("saving checkpoint...")
            checkpoint = {
                "model": raw_model.state_dict(),
                "adamw": optim[0].state_dict(),
                "muon": optim[1].state_dict(),
            }

            data_state = {
                "step": i + 1,
                "val_i": val_i,
            }
            save(checkpoint, save_file_name)
            save(model_conf, save_file_name, "model_config.pt")
            save(data_state, save_file_name, "training_info.pt")
            print_master("saved checkpoint")

        barrier(ddp)
        if i % save_rate == 0 or i == steps:
            print(f"rank:{rank} saving dataset state dicts...")
            dataset_state = {
                "train_data": train_data.state_dict(),
                "val_data": val_data.state_dict(),
            }
            save(dataset_state, save_file_name, f"dataset_info_rank{rank}.pt")
            print(f"rank:{rank} saved dataset state dicts")
    kill_dist(ddp)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
