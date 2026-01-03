import torch
from torch import optim
from torch.optim.optimizer import ParamsT

coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    if G.size(-2) >= G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
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
        params: ParamsT,
        lr=0.02,
        weight_decay=0.1,
        momentum=0.95,
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
