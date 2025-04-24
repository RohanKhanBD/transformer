import torch
from torch.utils.data import DataLoader

from time import time

from lightning import Fabric
from lightning.fabric.loggers.tensorboard import TensorBoardLogger

from tokenizer import Tokenizer
from model import TransformerLM
from utils import TextDataset, AttentionMask, load, save, est_loss, get_lr, set_seed

from configuration import (
    steps,
    eval_rate,
    eval_steps,
    save_rate,
    lr,
    min_lr,
    weight_decay,
    betas,
    warm_up,
    total_batch_size,
    batch_size,
    seed,
    compile_model,
    backend,
    save_fie_name,
    data_file_name,
)


def main(fabric: Fabric):
    set_seed(seed + fabric.global_rank)
    n_device = fabric.world_size
    tok = Tokenizer()
    tok.load()
    dev = fabric.device.type
    is_cuda = True if dev == "cuda" else False

    model_conf = TransformerLM.get_transformer_config(
        maxlen=1024,
        embedding_dim=512,
        num_heads=16,
        n_layers=8,
        inter_dim=512,
        window_size=512,
        mla=True,
        kv_lora_rank=32,
        flash=is_cuda,
        atten_types=[
            AttentionMask.Local,
            AttentionMask.Local,
            AttentionMask.Local,
            AttentionMask.Local,
            AttentionMask.Local,
            AttentionMask.Global,
        ],
    )
    grad_ecum = total_batch_size // (batch_size * model_conf.maxlen * n_device)
    model = TransformerLM(model_conf, tok.vocab_size)
    model: TransformerLM = torch.compile(
        model, backend=backend, disable=not compile_model
    )
    fabric.print(model)
    fabric.print(model.total_num_of_params())
    fabric.print(f"Number of devices:{n_device}")
    optim = model.get_optimizer(lr, weight_decay, betas, fused=is_cuda)
    model, optim = fabric.setup(model, optim)

    train_data = DataLoader(
        TextDataset(data_file_name, model_conf.maxlen, "train"),
        batch_size=batch_size,
        shuffle=False,
    )
    val_data = DataLoader(
        TextDataset(data_file_name, model_conf.maxlen, "val"),
        batch_size=batch_size,
        shuffle=False,
    )

    train_data, val_data = fabric.setup_dataloaders(train_data, val_data)

    try:
        checkpoint = load(save_fie_name)
        data_state = load(save_fie_name, "data_state_and_training_info.pt", False)
        fabric.print(data_state)
        train_i = data_state["step"]
        val_i = data_state["val_i"]

        train_data = data_state["train_data"]
        val_data = data_state["val_data"]

        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])

        fabric.print("loaded: True")
    except FileNotFoundError:
        train_i = 1
        val_i = 1
        fabric.print("loaded: False")

    train_data_iter = iter(train_data)
    val_data_iter = iter(val_data)
    x, y = next(train_data_iter)
    t0 = time()

    for i in range(train_i, steps + 1):
        ploss = 0.0
        n_lr = get_lr(i, steps, lr, min_lr, warm_up)
        fabric.log("learning rate", n_lr, step=i)
        for param_group in optim.param_groups:
            param_group["lr"] = n_lr

        if i % eval_rate == 0 or i == steps:
            e_loss = est_loss(model, val_data, val_data_iter, eval_steps)
            fabric.all_reduce(e_loss, reduce_op="mean")
            fabric.log("val loss", e_loss.item(), step=val_i)
            val_i += 1
            fabric.print(f"step: {i}/{steps}, val_loss: {e_loss.item():.8f}")

        if (i % save_rate == 0 or i == steps) and fabric.is_global_zero:
            fabric.print("saving checkpoint...")
            checkpoint = {"model": model.state_dict(), "optim": optim.state_dict()}

            data_state = {
                "train_data": train_data,
                "val_data": val_data,
                "step": i + 1,
                "val_i": val_i,
            }
            save(checkpoint, save_fie_name)
            save(model_conf, save_fie_name, "model_config.pt")
            save(data_state, save_fie_name, "data_state_and_training_info.pt")
            fabric.print("saved checkpoint")

        for grad_i in range(grad_ecum):
            no_sync_enable = grad_i < grad_ecum - 1
            with fabric.no_backward_sync(model, enabled=no_sync_enable):
                _, loss = model.forward(x, y)
                loss = loss / grad_ecum
                fabric.backward(loss)
            try:
                x, y = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_data)
                x, y = next(train_data_iter)
            ploss += loss.detach()

        fabric.all_reduce(ploss, reduce_op="mean")
        fabric.log("train loss", ploss, step=i)

        norm: torch.Tensor = fabric.clip_gradients(model, optim, max_norm=1.0)
        fabric.log("grad norm", norm.item(), step=i)
        optim.step()
        optim.zero_grad()

        if is_cuda:
            torch.cuda.synchronize()
        else:
            torch.cpu.synchronize()

        t1 = time()
        dt = t1 - t0
        t0 = t1
        tok_per_sec = (batch_size * model_conf.maxlen * grad_ecum * n_device) / dt
        fabric.print(
            f"step: {i}/{steps} | lr: {n_lr:.8f} | loss: {ploss:.8f} | norm: {norm.item():.8f} | time: {dt:.2f}sec | tok/sec: {tok_per_sec:.2f}"
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    logger = TensorBoardLogger("runs")
    fabric = Fabric(precision="bf16-mixed", loggers=[logger])
    fabric.launch()
    main(fabric)
