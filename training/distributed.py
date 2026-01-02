import os
import torch
import torch.distributed as dist


def dist_init(is_cuda: bool):
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
    return rank, local_rank, world_size, device, device_type, ddp, master_process


def kill_dist(ddp: bool):
    if ddp:
        dist.destroy_process_group()


def sync(is_cuda: bool):
    if is_cuda:
        torch.cuda.synchronize()


def barrier(ddp: bool):
    if ddp:
        dist.barrier()


def all_reduce(inp: torch.Tensor, op: dist.ReduceOp, ddp: bool):
    if ddp:
        dist.all_reduce(inp, op)
