"""Optimizer groups and the row-buffer interface to Toner's fast backend."""

from types import SimpleNamespace

import torch


def fast_arguments(config):
    o, s, r = config["optimizer"], config["statistics"], config["root"]
    return SimpleNamespace(
        toner_decay=s["momentum"],
        toner_iters=r["iterations"],
        toner_split=s["inner_strength"] != s["outer_strength"],
        toner_alpha=s["inner_strength"],
        toner_alpha_in=s["inner_strength"],
        toner_alpha_out=s["outer_strength"],
        toner_root="ref",
        toner_refresh=r["refresh"],
        toner_refresh_flagged=r["refresh_flagged"],
        toner_owner_weights="",
        toner_lr_mult=1.0,
        wd=o["weight_decay"],
        toner_overlap=r["overlap"],
        toner_own_pg=1,
        toner_stats_ctas=0,
        toner_offload="",
        toner_host_workers=r["host_workers"],
        toner_host_threads=r["host_threads"],
        toner_host_cores="",
        toner_tol=r["tolerance_kind"],
        toner_tol_frac=r["tolerance_fraction"],
        toner_tol_min=r["min_tolerance"],
        toner_eps=r["epsilon"],
        toner_rho_max=r["rho_max"],
        toner_graphs=r["graphs"],
        toner_inline=r["inline_steps"],
        toner_freeze=0,
        toner_repair_rounds=r["repair_rounds"],
        toner_check_every=0,
        toner_outer=1,
        beta1=o["betas"][0],
        beta2=o["betas"][1],
        eps=o["eps"],
        muon_momentum=o.get("momentum", 0.95),
        muon_nesterov=o.get("nesterov", True),
        muon_ns_steps=o.get("ns_steps", 10),
        muon_eps=o.get("ns_eps", 1e-5),
        muon_ns_mode=o.get("ns_mode", "tpu"),
    )


def build_optimizers(model, config, *, rows_total, world, rank, device):
    o = config["optimizer"]
    named = dict(model.named_parameters())
    matrices = [p for n, p in named.items() if n.startswith("body.blocks.") and p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    rest = [p for p in named.values() if id(p) not in matrix_ids]
    head = model.head.weight
    other = [p for p in rest if p is not head]
    fast = None
    if o["name"] in ("muont", "adamt"):
        from toner.fast import roots
        from toner.fast.engine import FastTonerCore

        if config["statistics"]["damping"] != roots.EPS:
            raise ValueError("This recorded fast backend requires damping=0.001")
        fast = FastTonerCore(
            list(model.body.blocks),
            named,
            o["name"],
            hid=config["model"]["hidden"],
            mlp=config["model"]["mlp"],
            rows_total=rows_total,
            args=fast_arguments(config),
            world=world,
            rank=rank,
            device=device,
        )
        optimizers = [fast]
    elif o["name"] == "muon":
        from toner.fast.muon import Muon

        optimizers = [
            Muon(
                matrices,
                lr=o["lr"],
                momentum=o["momentum"],
                nesterov=o["nesterov"],
                ns_steps=o["ns_steps"],
                eps=o["ns_eps"],
                weight_decay=o["weight_decay"],
                ns_mode=o["ns_mode"],
                batched=o.get("batched", True),
                distributed=o.get("distributed", True) and world > 1,
            )
        ]
    elif o["name"] == "adamw":
        optimizers = []
    else:
        raise ValueError(o["name"])
    groups = [
        {"params": [head], "weight_decay": o["head_weight_decay"]},
        {"params": other, "weight_decay": 0.0},
    ]
    if o["name"] == "adamw":
        groups.insert(0, {"params": matrices, "weight_decay": o["weight_decay"]})
    auxiliary = torch.optim.AdamW(
        groups, lr=o["auxiliary_lr"], betas=tuple(o["betas"]), eps=o["eps"], fused=True
    )
    optimizers.append(auxiliary)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["base_lr"] = o["auxiliary_lr"] if optimizer is auxiliary else o["lr"]
    clip_parameters = (
        rest
        if o["name"] in ("muon", "muont")
        else [
            p
            for optimizer in optimizers
            for group in optimizer.param_groups
            for p in group["params"]
        ]
    )
    return optimizers, fast, clip_parameters


@torch.no_grad()
def clip_gradients(parameters, max_norm):
    """The recorded global-norm clipping rule, with no additive epsilon."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not max_norm or not grads:
        return
    total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
    scale = torch.where(total < max_norm, 1.0, max_norm / total)
    torch._foreach_mul_(grads, scale)
