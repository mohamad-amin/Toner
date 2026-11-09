# Muon numerical routines adapted from Levanter (Apache-2.0).
"""Muon momentum and finite Newton–Schulz updates."""

import torch

ABC = (3.4445, -4.7750, 2.0315)


def _mm(x, y, mode):
    if mode == "tpu":
        return torch.mm(x.to(torch.bfloat16), y.to(torch.bfloat16), out_dtype=torch.float32)
    return x @ y


def newton_schulz(u, steps, eps, mode="tpu"):
    a, b, c = ABC
    x = u.float()
    x = x / (torch.linalg.matrix_norm(x) + eps)
    if mode == "bf16":
        x = x.to(torch.bfloat16)
    tr = x.size(0) > x.size(1)
    if tr:
        x = x.T
    for _ in range(steps):
        A = _mm(x, x.T, mode)
        x = a * x + _mm(b * A + c * _mm(A, A, mode), x, mode)
    return (x.T if tr else x).float()


def _bmm(x, y, mode):
    if mode == "tpu":
        return torch.bmm(x.to(torch.bfloat16), y.to(torch.bfloat16), out_dtype=torch.float32)
    return torch.bmm(x, y)


def newton_schulz_batched(U, steps, eps, mode="tpu"):
    """newton_schulz for a stack of same-shape matrices U [B, rows, cols]: the same math per matrix, a few batched
    GEMMs per iteration instead of one small GEMM per matrix (the per-matrix loop launched ~10^4 kernels a step)."""
    a, b, c = ABC
    x = U.float()
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + eps)
    if mode == "bf16":
        x = x.to(torch.bfloat16)
    tr = x.size(-2) > x.size(-1)
    if tr:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        A = _bmm(x, x.transpose(-2, -1), mode)
        x = a * x + _bmm(b * A + c * _bmm(A, A, mode), x, mode)
    return (x.transpose(-2, -1) if tr else x).float()


class Muon(torch.optim.Optimizer):
    """batched=True stacks same-shape matrices for Newton-Schulz (same math as the per-matrix loop, which
    m130_muon_1x ran with, at a fraction of the launch overhead). With torch.distributed initialized (DDP: every rank
    holds the same gradients), each rank orthogonalizes only its share of every stack and the results are
    all-gathered, so the Newton-Schulz work per GPU drops by the world size; every rank then applies all updates."""

    def __init__(
        self,
        params,
        lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=10,
        eps=1e-5,
        weight_decay=0.0,
        ns_mode="tpu",
        batched=True,
        distributed=True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                eps=eps,
                weight_decay=weight_decay,
                ns_mode=ns_mode,
                batched=batched,
            ),
        )
        import torch.distributed as dist

        self.dist = (
            dist
            if (
                distributed
                and dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
            else None
        )
        self.precond = None  # Newton-Muon (train.py --newton-muon): (p, grad) -> the preconditioned raw gradient

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            mu, lr, wd = g["momentum"], g["lr"], g["weight_decay"]
            if g.get("batched", False):
                self._step_batched(g, mu, lr, wd)
                continue
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                grad = p.grad if self.precond is None else self.precond(p, p.grad)
                buf.mul_(mu).add_(grad)
                u = grad.add(buf, alpha=mu) if g["nesterov"] else buf
                o = newton_schulz(u, g["ns_steps"], g["eps"], g["ns_mode"])
                p.mul_(1 - lr * wd)
                p.add_(o, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)

    def _step_batched(self, g, mu, lr, wd):
        ps = [p for p in g["params"] if p.grad is not None]
        if not ps:
            return
        bufs = []
        for p in ps:
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p)
            bufs.append(st["momentum_buffer"])
        grads = [p.grad if self.precond is None else self.precond(p, p.grad) for p in ps]
        torch._foreach_mul_(bufs, mu)
        torch._foreach_add_(bufs, grads)
        us = torch._foreach_add(grads, bufs, alpha=mu) if g["nesterov"] else bufs
        torch._foreach_mul_(ps, 1 - lr * wd)
        by_shape = {}
        for i, p in enumerate(ps):
            by_shape.setdefault(tuple(p.shape), []).append(i)
        for shape, idx in by_shape.items():
            U = torch.stack([us[i] for i in idx])
            if self.dist is None:
                o = newton_schulz_batched(U, g["ns_steps"], g["eps"], g["ns_mode"])
            else:  # rank r takes matrices r, r + W, ...; padded to equal shares for all_gather_into_tensor
                W, r = self.dist.get_world_size(), self.dist.get_rank()
                per = -(-len(idx) // W)
                Up = (
                    torch.cat((U, U.new_zeros((per * W - len(idx),) + U.shape[1:])))
                    if per * W > len(idx)
                    else U
                )
                mine = newton_schulz_batched(
                    Up[r::W], g["ns_steps"], g["eps"], g["ns_mode"]
                ).contiguous()
                gathered = torch.empty((W, per) + U.shape[1:], dtype=mine.dtype, device=mine.device)
                self.dist.all_gather_into_tensor(gathered, mine)
                o = gathered.transpose(0, 1).reshape((per * W,) + U.shape[1:])[
                    : len(idx)
                ]  # back to stack order
            torch._foreach_add_(
                [ps[i] for i in idx],
                list(o.unbind(0)),
                alpha=-lr * max(1.0, shape[0] / shape[1]) ** 0.5,
            )


if (
    __name__ == "__main__"
):  # the three modes agree; the result is near-orthogonal; batched == per-matrix
    import time

    torch.manual_seed(0)
    g = torch.randn(2048, 512, device="cuda")
    outs = {m: newton_schulz(g, 10, 1e-5, m) for m in ("tpu", "bf16", "f32")}
    for m, o in outs.items():
        s = torch.linalg.svdvals(o)
        print(
            f"{m:5s} singular values in [{s.min():.3f}, {s.max():.3f}], rel diff vs f32 "
            f"{(o - outs['f32']).norm() / outs['f32'].norm():.2e}"
        )
    # one Muon step on the 130M block-matrix shapes, per-matrix vs batched: same update, and the time per step
    shapes = [(512, 512)] * 128 + [(2048, 512)] * 64 + [(512, 2048)] * 32
    ref = [torch.randn(s, device="cuda") * 0.02 for s in shapes]
    for batched in (False, True):
        ps = [torch.nn.Parameter(r.clone()) for r in ref]
        for p, r in zip(ps, ref):
            p.grad = r.flip(0).clone()  # a deterministic stand-in gradient
        opt = Muon(ps, lr=0.016, momentum=0.95, weight_decay=0.1, batched=batched)
        opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            opt.step()
        torch.cuda.synchronize()
        print(f"batched={batched}: {(time.time() - t0) / 5 * 1e3:.1f} ms per step")
        if batched:
            d = max(((p - q).norm() / (q - r).norm()).item() for p, q, r in zip(ps, prev, ref))
            print(
                f"max relative difference of the parameter change, batched vs per-matrix: {d:.2e}"
            )
        prev = [p.detach().clone() for p in ps]
