"""Optimizer conventions for the nanoGPT reproduction configurations."""

import torch
import torch.distributed as dist


def cubic_polar(gradient):
    x = gradient.bfloat16()
    transpose = gradient.size(-2) > gradient.size(-1)
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        gram = x @ x.mT
        polynomial = -1.5 * gram + 0.5 * gram @ gram
        x = 2 * x + polynomial @ x
    return x.mT if transpose else x


@torch.compile
def update_direction(
    gradient, momentum_buffer, inner=None, outer=None, momentum=0.95, nesterov=True
):
    momentum_buffer.lerp_(gradient, 1 - momentum)
    update = gradient.lerp_(momentum_buffer, momentum) if nesterov else momentum_buffer
    if inner is not None:
        update = update.bfloat16() @ inner
    update = cubic_polar(update).bfloat16()
    if outer is not None:
        update = update @ outer
    return update.float() * max(1, gradient.size(-2) / gradient.size(-1)) ** 0.5


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, momentum=0.95, nesterov=True):
        super().__init__(
            sorted(list(params), key=lambda p: p.size(), reverse=True),
            dict(lr=lr, weight_decay=weight_decay, mu=momentum, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self):
        world, rank = dist.get_world_size(), dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            padded = params + [torch.empty_like(params[-1])] * (world - len(params) % world)
            for first in range(0, len(params), world):
                if first + rank < len(params):
                    p = params[first + rank]
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    update = update_direction(
                        p.grad, state["momentum"], None, None, group["mu"], group["nesterov"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(padded[first : first + world], padded[first + rank])

    def close(self):
        pass


def configure_backend(config):
    import backend

    root = config["root"]
    names = {
        "mode": "ROOT_ROOTS",
        "inner_strength": "ROOT_P_IN",
        "outer_strength": "ROOT_P_OUT",
        "decay_strength": "ROOT_P_DECAY",
        "damping": "ROOT_DAMPING",
        "iterations": "ROOT_ITERS",
        "refresh": "ROOT_REFRESH",
        "inline_steps": "ROOT_INLINE",
        "tolerance": "ROOT_TOL",
        "rho_max": "ROOT_RHO_MAX",
        "sqrt_iterations": "ROOT_SQRT_ITERS",
        "max_stale_steps": "ROOT_STALE_MAX",
        "workers": "ROOT_WORKERS",
        "wide_workers": "ROOT_WIDE_MAX",
        "graphs": "ROOT_GRAPH",
        "eigh_threads": "ROOT_EIGH_THREADS",
    }
    for key, name in names.items():
        if key in root:
            setattr(backend, name, root[key])
    backend.STATISTIC_MOMENTUM = config["statistics"]["momentum"]
    backend.GLOBAL_STATISTICS = config["statistics"]["global"]
    backend.STAT_MODULES.clear()
    backend._STAT["rows"] = None
    backend.ROOT_DECAY = "outer"
    backend._ROOT_D8 = abs(backend.ROOT_P_DECAY * 8 - round(backend.ROOT_P_DECAY * 8)) < 1e-9
    backend.ROOT_DEC_ROOT = backend.ROOT_P_DECAY if backend._ROOT_D8 else 2 * backend.ROOT_P_DECAY
    backend.ROOT_USE = sorted(
        {q for q in (backend.ROOT_P_IN, backend.ROOT_P_OUT, backend.ROOT_DEC_ROOT) if q}
    )
    backend.ROOT_POWERS = sorted({p for q in backend.ROOT_USE for p in backend.root_split(q)})
    return backend


def build_optimizers(model, config):
    opt = config["optimizer"]
    aux = config["auxiliary"]
    auxiliary = torch.optim.AdamW(
        [
            {"params": [model.embed.weight], "lr": aux["embedding_lr"]},
            {"params": [model.proj.weight], "lr": aux["head_lr"]},
            {"params": [p for p in model.parameters() if p.ndim < 2], "lr": aux["scalar_lr"]},
        ],
        betas=tuple(aux["betas"]),
        eps=aux["eps"],
        weight_decay=aux["weight_decay"],
        fused=True,
    )
    matrices = [p for p in model.blocks.parameters() if p.ndim >= 2]
    hooks = []
    if opt["name"] == "muon":
        body = Muon(matrices, opt["lr"], opt["weight_decay"], opt["momentum"], opt["nesterov"])
    elif opt["name"] == "muont":
        backend = configure_backend(config)
        rank, world = dist.get_rank(), dist.get_world_size()
        ordered = sorted(matrices, key=lambda p: p.size(), reverse=True)
        owned = set(ordered[rank::world])
        sources = {}
        head, stride = config["statistics"]["head"], config["statistics"]["stride"]

        def record(module, inputs, output):
            if module.training and torch.is_grad_enabled():
                with torch.no_grad():
                    rows = torch.cat(
                        [inputs[0].detach()[:, :head], inputs[0].detach()[:, head + 1 :: stride]],
                        dim=1,
                    ).flatten(0, 1)
                    module.stat_sum.add_(torch.mm(rows.T, rows, out_dtype=torch.float32))
                    module.stat_rows.add_(rows.size(0))

        for layer, block in enumerate(model.blocks):
            a, m = block.attn, block.mlp
            for kind, hooked, users in [
                ("attn_in", a.q, (a.q, a.k, a.v)),
                ("attn_out", a.proj, (a.proj,)),
                ("mlp_in", m.fc, (m.fc,)),
                ("mlp_hid", m.proj, (m.proj,)),
            ]:
                d = hooked.in_features
                device = hooked.weight.device
                hooked.register_buffer(
                    "stat_sum", torch.zeros(d, d, device=device), persistent=False
                )
                hooked.register_buffer(
                    "stat_rows", torch.zeros((), device=device), persistent=False
                )
                if any(u.weight in owned for u in users):
                    hooked.register_buffer(
                        "stat_cov", torch.zeros(d, d, device=device), persistent=False
                    )
                    hooked.register_buffer(
                        "stat_weight", torch.zeros((), device=device), persistent=False
                    )
                hooked.stat_name = f"{layer}.{kind}"
                hooks.append(hooked.register_forward_hook(record))
                backend.STAT_MODULES.append(hooked)
                sources.update({u.weight: hooked for u in users})
        body = backend.TonerRunOptimizer(
            matrices,
            lr=opt["lr"],
            weight_decay=opt["weight_decay"],
            mu=opt["momentum"],
            nesterov=opt["nesterov"],
            sources=sources,
        )
    else:
        raise ValueError("optimizer.name must be muon or muont")
    for optimizer in [auxiliary, body]:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return auxiliary, body, hooks
