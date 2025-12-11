import torch
from model import TransformerLM


def model_loss(
    model: TransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    dtype: torch.dtype,
    device_type: str,
    is_cuda: bool,
    use_autocast: bool,
):
    with torch.autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=is_cuda and use_autocast,
    ):
        _, loss = model.forward(x, y)
    return loss


@torch.no_grad()
def est_loss(
    model: torch.nn.Module,
    val_iter,
    eval_steps: int,
    device: str,
    device_type: str,
    is_cuda: bool,
    use_autocast: bool,
    dtype: torch.dtype,
):
    model.eval()
    losses = torch.zeros(eval_steps, device=device)
    for i in range(eval_steps):
        x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        loss = model_loss(model, x, y, dtype, device_type, is_cuda, use_autocast)
        losses[i] = loss.item()
    model.train()
    return losses.mean()
