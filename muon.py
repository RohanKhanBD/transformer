import torch
from torch import optim


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    if G.size(-2) >= G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) >= G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor, momentum: torch.Tensor, beta=0.95, ns_steps=5, nesterov=True
):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.1,
        momentum=0.5,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    param.grad == torch.zeros_like(param)
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)
                update = muon_update(
                    param.grad, state["momentum_buffer"], beta=group["momentum"]
                )
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update.reshape(param.shape), alpha=-group["lr"])

        return loss
