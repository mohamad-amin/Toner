"""Current-step root refinement and distributed MuonT updates."""

import os
import time
from pathlib import Path
import torch
from torch import Tensor
import torch.distributed as dist
from optimizer import cubic_polar

STAT_MODULES = []
GLOBAL_STATISTICS = True
STATISTIC_MOMENTUM = 0.95
_STAT = {"rows": None}
ROOT_P_IN, ROOT_P_OUT = (0.5, 0.125)
ROOT_DAMPING = 0.001
ROOT_DECAY = "outer"
ROOT_P_DECAY = 0.125
ROOT_ROOTS = "fast"
ROOT_ITERS, ROOT_REFRESH, ROOT_INLINE = (4, 4, 20)
ROOT_WORKERS = int(os.environ.get("ROOT_WORKERS", "3"))
ROOT_WIDE_MAX = int(os.environ.get("ROOT_WIDE_MAX", "2"))
ROOT_EIGH_THREADS = int(os.environ.get("ROOT_EIGH_THREADS", "0"))
ROOT_SQRT_ITERS = 2
ROOT_TOL, ROOT_RHO_MAX = (0.001, 0.2)
ROOT_CHECK = int("0")
ROOT_DIAG = int("0")
ROOT_GRAPH = int(os.environ.get("ROOT_GRAPH", "1"))
ROOT_BASIC = (1.0, 0.5, 0.25, 0.125)


def root_split(p):
    out, r = ([], p)
    for b in ROOT_BASIC:
        if r >= b - 1e-12:
            out.append(b)
            r -= b
    assert abs(r) < 1e-12, f"power {p}: not a sum of distinct 1, 1/2, 1/4, 1/8"
    return tuple(out)


_ROOT_D8 = abs(ROOT_P_DECAY * 8 - round(ROOT_P_DECAY * 8)) < 1e-09
ROOT_DEC_ROOT = (
    0
    if ROOT_DECAY != "outer" or ROOT_P_DECAY == 0
    else ROOT_P_DECAY
    if _ROOT_D8
    else 2 * ROOT_P_DECAY
)
ROOT_USE = sorted({q for q in (ROOT_P_IN, ROOT_P_OUT, ROOT_DEC_ROOT) if q != 0})
ROOT_POWERS = sorted({b for q in ROOT_USE for b in root_split(q)})
assert ROOT_ROOTS in ("fast", "exact") and ROOT_DECAY in ("outer", "plain")
assert ROOT_ROOTS == "exact" or set(ROOT_POWERS) <= {0.125, 0.25, 0.5, 1.0}, (
    "fast roots: basic powers 1/8, 1/4, 1/2, 1 (else exact)"
)
assert not (ROOT_DEC_ROOT == 1.0 and ROOT_DAMPING < 2e-05), "Unsupported root combination"
import concurrent.futures as _root_futures
import torch._dynamo as _root_dynamo

if hasattr(_root_dynamo.config, "recompile_limit"):
    _root_dynamo.config.recompile_limit = max(_root_dynamo.config.recompile_limit, 64)
ROOT_LO = 4096.0
ROOT_LOWP = torch.float16


def _root_hi_lo(x):
    h = x.half()
    return (h, ((x - h.float()) * ROOT_LO).half())


def _root_cols_hi(x):
    h, lo = _root_hi_lo(x)
    return torch.cat([h, lo], -1)


def _root_cols_lo(x):
    h, lo = _root_hi_lo(x)
    return torch.cat([lo, h], -1)


def _root_rows_hi(x):
    h, lo = _root_hi_lo(x)
    return torch.cat([h, lo], -2)


def _root_rows_lo(x):
    h, lo = _root_hi_lo(x)
    return torch.cat([lo, h], -2)


def _root_scaled_cols_hi(C):
    mu = C.diagonal(dim1=-2, dim2=-1).sum(-1) / C.size(-1)
    return _root_cols_hi(C / mu[..., None, None])


_ROOT_EXACT = {"emulate_precision_casts": True}
root_cols_hi = torch.compile(_root_cols_hi, dynamic=False, options=_ROOT_EXACT)
root_cols_lo = torch.compile(_root_cols_lo, dynamic=False, options=_ROOT_EXACT)
root_rows_hi = torch.compile(_root_rows_hi, dynamic=False, options=_ROOT_EXACT)
root_rows_lo = torch.compile(_root_rows_lo, dynamic=False, options=_ROOT_EXACT)
root_scaled_cols_hi = torch.compile(_root_scaled_cols_hi, dynamic=False, options=_ROOT_EXACT)


def root_mm16(A2, B2, a_lo_first):
    k = A2.size(-1) // 2
    Ah, Bh = (A2[..., k:], B2[..., :k, :]) if a_lo_first else (A2[..., :k], B2[..., k:, :])
    C = torch.bmm(Ah, Bh, out_dtype=torch.float32)
    return torch.baddbmm(C, A2, B2, alpha=1.0 / ROOT_LO, out_dtype=torch.float32)


class RootBasis:
    def __init__(self, X):
        self.SB = root_rows_lo(X.contiguous())
        self.S2 = root_cols_hi(X.contiguous())

    def set_all(self, X):
        self.SB.copy_(root_rows_lo(X.contiguous()))
        self.S2.copy_(root_cols_hi(X.contiguous()))

    def install(self, j, SB, S2):
        self.SB[j].copy_(SB)
        self.S2[j].copy_(S2)


def root_transform(C, basis):
    Y = root_mm16(root_scaled_cols_hi(C), basis.SB, False)
    return root_mm16(basis.SB.mT, root_rows_hi(Y), True)


def _root_sym(R):
    return 0.5 * (R + R.mT)


root_sym = torch.compile(_root_sym, dynamic=False)


def root_form(basis, D):
    Xh = basis.S2[..., : D.size(-1)]
    return root_sym(root_bmm32(root_bmm32(Xh, D).half(), Xh.mT))


def _root_y(lam, O):
    return (torch.diag_embed(lam) + O).half()


def _root_y2(lam, O, OO):
    return (
        torch.diag_embed(lam.square()) + (lam[..., :, None] + lam[..., None, :]) * O + OO
    ).half()


def _root_y4(lam, P, PP):
    l2 = lam.square()
    D = torch.diag_embed(l2.square()) + (l2[..., :, None] + l2[..., None, :]) * P + PP
    s = torch.exp2(
        torch.ceil(torch.log2(D.abs().amax((-2, -1)).clamp_min(1e-30) / 16384.0))
    ).clamp_min(1.0)
    return ((D / s[..., None, None]).half(), s)


def _root_rescale(R, s):
    return R * s[:, None, None]


root_y = torch.compile(_root_y, dynamic=False)
root_y2 = torch.compile(_root_y2, dynamic=False)
root_y4 = torch.compile(_root_y4, dynamic=False)
root_rescale = torch.compile(_root_rescale, dynamic=False)


def _root_start(B, Bp, lamp, Op, warm):
    B = 0.5 * (B + B.mT)
    dB = B.diagonal(dim1=-2, dim2=-1)
    d = dB + ROOT_DAMPING
    E = B - torch.diag_embed(dB)
    rho = (E.abs() / (d[..., :, None] + d[..., None, :])).amax((-2, -1))
    lam0 = d.clamp_min(ROOT_DAMPING * 0.001).pow(-0.25)
    dd = d[..., :, None] - d[..., None, :]
    mid = 0.5 * (d[..., :, None] + d[..., None, :])
    L = torch.where(
        dd.abs() > 1e-06 * mid,
        (lam0[..., :, None] - lam0[..., None, :]) / torch.where(dd == 0, 1.0, dd),
        -0.25 * mid.pow(-1.25),
    )
    Dw = L * (B - Bp)
    Yw = torch.diag_embed(lamp) + Op + Dw
    lw = Yw.diagonal(dim1=-2, dim2=-1)
    lam = torch.where(warm[:, None], lw, lam0)
    O = torch.where(warm[:, None, None], Yw - torch.diag_embed(lw), L * E)
    dn0 = torch.where(warm, Dw.square().sum((-2, -1)), (L * E).square().sum((-2, -1))).sqrt()
    return (d, E, E.to(ROOT_LOWP), lam, O, O.to(ROOT_LOWP), d.sqrt(), rho, dn0, lam0, B)


def _root_iter_mid(lam, O, OO):
    P = (lam[..., :, None] + lam[..., None, :]) * O + OO
    return (P, P.to(ROOT_LOWP))


def _root_iter_w(P, d, EP):
    W = d[..., :, None] * P + EP
    s = torch.exp2(
        torch.ceil(torch.log2(W.abs().amax((-2, -1)).clamp_min(1e-30) / 16384.0))
    ).clamp_min(1.0)
    return ((W / s[..., None, None]).to(ROOT_LOWP), s)


def _root_iter_post(lam, O, d, E, P, EP, PW, sw, lam0, sq):
    PW = PW * sw[..., None, None]
    l2 = lam.square()
    F = (
        l2[..., :, None] * E * l2[..., None, :]
        + (l2 * d)[..., :, None] * P
        + P * (d * l2)[..., None, :]
        + l2[..., :, None] * EP
        + EP.mT * l2[..., None, :]
        + 0.5 * (PW + PW.mT)
    )
    G = torch.diag_embed(1.0 - l2 * d * l2) - F
    Dl = G / ((lam0[..., :, None] + lam0[..., None, :]) * (sq[..., :, None] + sq[..., None, :]))
    dg = Dl.diagonal(dim1=-2, dim2=-1)
    O = O + Dl - torch.diag_embed(dg)
    return (lam + dg, O, O.to(ROOT_LOWP), Dl.square().sum((-2, -1)).sqrt())


def _root_sqrt_start(lam, O):
    s = lam.clamp_min(0).sqrt()
    S0 = s[..., :, None] + s[..., None, :]
    Q = O / S0
    return (s, Q, Q.to(ROOT_LOWP), S0, Q.square().sum((-2, -1)).sqrt())


def _root_sqrt_post(lam, O, s, Q, QQ, S0):
    S = s[..., :, None] + s[..., None, :]
    G = torch.diag_embed(lam - s.square()) + O - S * Q - QQ
    Dl = G / S0
    dg = Dl.diagonal(dim1=-2, dim2=-1)
    Q = Q + Dl - torch.diag_embed(dg)
    return (s + dg, Q, Q.to(ROOT_LOWP), Dl.square().sum((-2, -1)).sqrt())


def _root_select(usable, Rn, R):
    R.copy_(torch.where(usable[:, None, None], Rn, R))


def _root_keep(usable, Bs, lam, O, Bp, lamp, Op, warm):
    Bp.copy_(Bs)
    lamp.copy_(lam)
    Op.copy_(O)
    warm.copy_(usable)


def _root_decay(R2, m2):
    return (R2 / m2[:, None, None]).half()


root_start = torch.compile(_root_start, dynamic=False)
root_iter_mid = torch.compile(_root_iter_mid, dynamic=False)
root_iter_w = torch.compile(_root_iter_w, dynamic=False)
root_iter_post = torch.compile(_root_iter_post, dynamic=False)
root_sqrt_start = torch.compile(_root_sqrt_start, dynamic=False)
root_sqrt_post = torch.compile(_root_sqrt_post, dynamic=False)
root_select = torch.compile(_root_select, dynamic=False)
root_keep = torch.compile(_root_keep, dynamic=False)
root_decay = torch.compile(_root_decay, dynamic=False)


def root_bmm32(a, b):
    return torch.bmm(a, b, out_dtype=torch.float32)


def root_iterate(d, E, Eb, lam, O, Ob, sq, lam0, iters):
    dn = []
    for _ in range(iters):
        P, Pb = root_iter_mid(lam, O, root_bmm32(Ob, Ob))
        EP = root_bmm32(Eb, Pb)
        W, sw = root_iter_w(P, d, EP)
        lam, O, Ob, upd = root_iter_post(lam, O, d, E, P, EP, root_bmm32(Pb, W), sw, lam0, sq)
        dn.append(upd)
    return (lam, O, Ob, dn)


def root_estimate(lam, dprev, dlast):
    r = (dlast / dprev.clamp_min(1e-30)).clamp_max(0.95)
    return r / (1 - r) * dlast / lam.square().sum(-1).sqrt()


def root_froot(B, Bp, lamp, Op, warm, iters):
    d, E, Eb, lam, O, Ob, sq, rho, dn0, lam0, Bs = root_start(B, Bp, lamp, Op, warm)
    lam, O, Ob, dn = root_iterate(d, E, Eb, lam, O, Ob, sq, lam0, iters)
    dn = [dn0] + dn
    return (lam, O, Ob, rho, root_estimate(lam, dn[-2], dn[-1]), Bs)


def root_sqrt(lam, O, iters):
    s, Q, Qb, S0, dn0 = root_sqrt_start(lam, O)
    dn = [dn0]
    for _ in range(iters):
        s, Q, Qb, upd = root_sqrt_post(lam, O, s, Q, root_bmm32(Qb, Qb), S0)
        dn.append(upd)
    return (s, Q, root_estimate(s, dn[-2], dn[-1]))


def root_eigh(sym):
    try:
        return torch.linalg.eigh(sym)
    except torch.linalg.LinAlgError:
        lam, V = torch.linalg.eigh(sym.detach().double().cpu())
        return (
            lam.to(device=sym.device, dtype=sym.dtype),
            V.to(device=sym.device, dtype=sym.dtype),
        )


def root_exact(C, powers):
    lam, V = root_eigh(0.5 * (C + C.mT))
    x = lam.clamp_min(0)
    x = x / x.mean(-1, keepdim=True).clamp_min(1e-30)
    out = {}
    for p in powers:
        R = V * (x + ROOT_DAMPING).pow(-p)[..., None, :] @ V.mT
        out[p] = 0.5 * (R + R.mT)
    return (V, out)


def root_exact64(C, p):
    lam, V = torch.linalg.eigh((0.5 * (C + C.mT)).double())
    x = lam.clamp_min(0)
    x = x / x.mean(-1, keepdim=True).clamp_min(1e-300)
    return V * (x + ROOT_DAMPING).pow(-p)[..., None, :] @ V.mT


def root_host_basis(snap, ready, device, side, consumer):
    with torch.cuda.device(device):
        t0 = time.perf_counter()
        host = torch.empty(snap.shape, dtype=snap.dtype, pin_memory=True)
        with torch.cuda.stream(side):
            side.wait_event(ready)
            snap.record_stream(side)
            host.copy_(snap, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(side)
        copied.synchronize()
        sym = 0.5 * (host + host.T)
        try:
            _, V = torch.linalg.eigh(sym)
        except torch.linalg.LinAlgError:
            _, V = torch.linalg.eigh(sym.double())
            V = V.float()
        V = V.pin_memory()
        with torch.cuda.stream(side):
            Vg = V.to(device, non_blocking=True).unsqueeze(0)
            SB, S2 = (_root_rows_lo(Vg)[0], _root_cols_hi(Vg)[0])
            done = torch.cuda.Event()
            done.record(side)
        SB.record_stream(consumer)
        S2.record_stream(consumer)
        return (SB, S2, done, time.perf_counter() - t0)


def _root_products(R):
    n = R.size(-1)
    nrm2 = R.square().sum((-2, -1))
    return (
        R.bfloat16(),
        (R * (n**0.5 / nrm2.sqrt().clamp_min(1e-30))[:, None, None]).bfloat16(),
        nrm2 / n,
    )


root_products = torch.compile(_root_products, dynamic=False)


@torch.compile
def root_update(grad, momentum, r_in, rhat_out, mu: float, nesterov: bool):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    if r_in is not None:
        update = update.bfloat16() @ r_in
    update = cubic_polar(update).bfloat16()
    if rhat_out is not None:
        update = update @ rhat_out
    update = update.float()
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


ROOT_STALE_MAX = int(os.environ.get("ROOT_STALE_MAX", "16"))


def root_flush(opt):
    if _STAT["rows"] is None:
        _STAT["rows"] = float(STAT_MODULES[0].stat_rows.item())
    rows = _STAT["rows"] * (dist.get_world_size() if GLOBAL_STATISTICS else 1)
    sums = [m.stat_sum for m in STAT_MODULES]
    if GLOBAL_STATISTICS:
        flat = torch.cat([s.flatten() for s in sums])
        dist.all_reduce(flat)
        sums = [f.view_as(s) for f, s in zip(flat.split([s.numel() for s in sums]), sums)]
    own = [(m, s) for m, s in zip(STAT_MODULES, sums) if hasattr(m, "stat_cov")]
    if own:
        covs, ws = ([m.stat_cov for m, _ in own], [m.stat_weight for m, _ in own])
        torch._foreach_mul_(covs, STATISTIC_MOMENTUM)
        torch._foreach_add_(
            covs, torch._foreach_div([s for _, s in own], rows), alpha=1 - STATISTIC_MOMENTUM
        )
        torch._foreach_mul_(ws, STATISTIC_MOMENTUM)
        torch._foreach_add_(ws, 1 - STATISTIC_MOMENTUM)
    torch._foreach_zero_([m.stat_sum for m in STAT_MODULES])
    torch._foreach_zero_([m.stat_rows for m in STAT_MODULES])


class RootGroup:
    def __init__(self, modules):
        self.modules = modules
        k, n = (len(modules), modules[0].stat_cov.size(0))
        dev = modules[0].stat_cov.device
        self.C = torch.zeros(k, n, n, device=dev, dtype=torch.float32)
        for j, m in enumerate(modules):
            self.C[j].copy_(m.stat_cov)
            m.stat_cov = self.C[j]
        self.Rs = {p: torch.zeros(k, n, n, device=dev, dtype=torch.float32) for p in ROOT_POWERS}
        if ROOT_ROOTS == "fast" and ROOT_POWERS:
            self.Bp, self.Op = (torch.zeros(k, n, n, device=dev), torch.zeros(k, n, n, device=dev))
            self.lamp, self.warm_ok = (
                torch.zeros(k, n, device=dev),
                torch.zeros(k, device=dev, dtype=torch.bool),
            )
        self.wide = n > 1000
        self.basis, self.pending, self.age, self.snap_step, self.flagged = (
            None,
            {},
            [0] * k,
            {},
            [],
        )
        self.Rb, self.Rhat, self.P = ({}, {}, None)
        self.graph, self.out, self.warm = (None, None, False)
        self.flags_host = torch.zeros(k, dtype=torch.uint8, pin_memory=True)
        self.flags_ev, self.unusable_run = (None, [0] * k)
        self.cnt = torch.zeros(3, k, device=dev, dtype=torch.int32)
        self.mx = torch.zeros(2, k, device=dev, dtype=torch.float32)


class TonerRunOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, nesterov=True, sources=None):
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sources, self.steps, self.groups, self.begun = (sources, 0, None, False)
        self.pool = (
            _root_futures.ThreadPoolExecutor(max_workers=ROOT_WORKERS) if ROOT_WORKERS > 0 else None
        )
        if ROOT_EIGH_THREADS > 0:
            torch.set_num_threads(ROOT_EIGH_THREADS)
        self.side, self.copy_stream = (torch.cuda.Stream(), torch.cuda.Stream())
        self._reset_diag()

    def _reset_diag(self):
        self.diag = dict(fallback=0, refresh=[], installed=0, host_ms=0.0, host_max=0.0, laps={})

    def _lap(self, key):
        t = time.perf_counter()
        self.diag["laps"][key] = self.diag["laps"].get(key, 0.0) + t - self._t0
        self._t0 = t

    def _setup(self, owned):
        mods = []
        for p in owned:
            m = self.sources[p]
            if m not in mods:
                mods.append(m)
        by_n = {}
        for m in mods:
            by_n.setdefault(m.stat_cov.size(0), []).append(m)
        self.groups = [RootGroup(ms) for n, ms in sorted(by_n.items(), reverse=True)]
        self.where = {m: (g, j) for g in self.groups for j, m in enumerate(g.modules)}

    def _refresh(self, g, j):
        snap = g.C[j].clone()
        ready = torch.cuda.Event()
        ready.record()
        g.pending[j] = self.pool.submit(
            root_host_basis, snap, ready, snap.device, self.copy_stream, self.side
        )
        g.snap_step[j] = self.steps

    def _schedule(self):
        free = ROOT_WORKERS - sum((len(g.pending) for g in self.groups))
        wide = sum((len(g.pending) for g in self.groups if g.wide))
        cand = []
        for gi, g in enumerate(self.groups):
            for j in range(len(g.modules)):
                age = self.steps - g.age[j]
                if j in g.pending:
                    continue
                if j in g.flagged:
                    cand.append((int(g.wide), -age, gi, j))
                elif age >= ROOT_REFRESH:
                    cand.append((2, -age, gi, j))
            g.flagged = []
        for _, _, gi, j in sorted(cand):
            g = self.groups[gi]
            if free <= 0:
                break
            if g.wide and wide >= ROOT_WIDE_MAX:
                continue
            self._refresh(g, j)
            free -= 1
            wide += int(g.wide)

    def _products(self, g):
        Rb, Rhat, m2 = ({}, {}, {})
        R = {}
        for q in ROOT_USE:
            parts = root_split(q)
            Rq = g.Rs[parts[0]]
            for b in parts[1:]:
                Rq = root_bmm32(Rq.half(), g.Rs[b].half())
            R[q] = Rq
            Rb[q], Rhat[q], m2[q] = root_products(Rq)
        P = None
        if ROOT_DEC_ROOT and _ROOT_D8:
            Rh = R[ROOT_DEC_ROOT].half()
            P = root_decay(root_bmm32(Rh, Rh), m2[ROOT_DEC_ROOT])
        elif ROOT_DEC_ROOT:
            P = root_decay(R[ROOT_DEC_ROOT], R[ROOT_DEC_ROOT].diagonal(dim1=-2, dim2=-1).mean(-1))
        return (Rb, Rhat, P)

    def _fast_body(self, g):
        B = root_transform(g.C, g.basis)
        lam, O, Ob, rho, est, Bs = root_froot(B, g.Bp, g.lamp, g.Op, g.warm_ok, ROOT_ITERS)
        fin = torch.isfinite(est) & torch.isfinite(lam).all(-1) & (rho <= ROOT_RHO_MAX)
        D, sc, ests = ({}, {}, [])
        for p in ROOT_POWERS:
            if p == 0.25:
                D[p] = root_y(lam, O)
                ests.append(est)
            elif p == 0.5:
                D[p] = root_y2(lam, O, root_bmm32(Ob, Ob))
                ests.append(2.0 * est)
            elif p == 1.0:
                P, Pb = root_iter_mid(lam, O, root_bmm32(Ob, Ob))
                D[p], sc[p] = root_y4(lam, P, root_bmm32(Pb, Pb))
                ests.append(4.0 * est)
            else:
                s, Q, est_s = root_sqrt(lam, O, ROOT_SQRT_ITERS)
                D[p] = root_y(s, Q)
                ests.append(0.5 * est + est_s)
        e = torch.stack(ests).amax(0)
        fin = fin & torch.isfinite(e)
        good, usable = (fin & (e <= ROOT_TOL), fin & (e <= 10 * ROOT_TOL))
        for p in ROOT_POWERS:
            Rn = root_form(g.basis, D[p])
            root_select(usable, root_rescale(Rn, sc[p]) if p in sc else Rn, g.Rs[p])
        root_keep(usable, Bs, lam, O, g.Bp, g.lamp, g.Op, g.warm_ok)
        g.cnt += torch.stack([~good, ~usable, ~torch.isfinite(e)]).int()
        torch.maximum(
            g.mx,
            torch.stack([rho, torch.where(torch.isfinite(e), e, 0.0)]).nan_to_num(1e30),
            out=g.mx,
        )
        flags = (~good).to(torch.uint8) + 2 * (~usable).to(torch.uint8)
        return self._products(g) + (flags,)

    def _fast(self, g, main):
        exact_now = []
        if g.flags_ev is not None and g.flags_ev.query():
            fl = g.flags_host.tolist()
            for j, f in enumerate(fl):
                g.unusable_run[j] = g.unusable_run[j] + 1 if f & 2 else 0
                if f:
                    g.flagged.append(j)
                if g.unusable_run[j] >= ROOT_STALE_MAX:
                    exact_now.append(j)
                    g.unusable_run[j] = 0
            g.flags_ev = None
        if exact_now:
            self.diag["fallback"] += len(exact_now)
            fidx = torch.tensor(exact_now, device=g.C.device)
            V, Rf = root_exact(g.C[fidx], ROOT_POWERS)
            SB, S2 = (_root_rows_lo(V), _root_cols_hi(V))
            for i, j in enumerate(exact_now):
                g.basis.install(j, SB[i], S2[i])
                g.warm_ok[j : j + 1].zero_()
                g.age[j] = self.steps
                for p in ROOT_POWERS:
                    g.Rs[p][j].copy_(Rf[p][i])
        if g.graph is not None:
            g.graph.replay()
        elif g.warm and ROOT_GRAPH:
            g.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g.graph, stream=self.side, capture_error_mode="thread_local"):
                g.out = self._fast_body(g)
            g.graph.replay()
        else:
            g.out = self._fast_body(g)
            g.warm = True
            for t in (
                list(g.out[0].values())
                + list(g.out[1].values())
                + [t for t in g.out[2:] if t is not None]
            ):
                t.record_stream(main)
        g.Rb, g.Rhat, g.P, flags = g.out
        g.flags_host.copy_(flags, non_blocking=True)
        g.flags_ev = torch.cuda.Event()
        g.flags_ev.record()

    @torch.no_grad()
    def roots(self, main):
        for g in self.groups:
            for j in [j for j, f in g.pending.items() if f.done()]:
                SB, S2, done, wall = g.pending.pop(j).result()
                torch.cuda.current_stream().wait_event(done)
                if g.basis is not None:
                    g.basis.install(j, SB, S2)
                    g.warm_ok[j : j + 1].zero_()
                    g.age[j] = g.snap_step.pop(j, self.steps)
                    self.diag["installed"] += 1
                self.diag["refresh"].append((wall, g.C.size(-1)))
            self._lap("install")
            if not ROOT_POWERS:
                g.P = None
                continue
            if ROOT_ROOTS == "exact" or self.steps < ROOT_INLINE or g.basis is None:
                V, R = root_exact(g.C, ROOT_POWERS)
                if ROOT_ROOTS == "fast":
                    if g.basis is None:
                        g.basis = RootBasis(V)
                    else:
                        g.basis.set_all(V)
                    g.age = [self.steps] * len(g.modules)
                for p in ROOT_POWERS:
                    g.Rs[p].copy_(R[p])
                g.Rb, g.Rhat, g.P = self._products(g)
                for t in (
                    list(g.Rb.values()) + list(g.Rhat.values()) + ([g.P] if g.P is not None else [])
                ):
                    t.record_stream(main)
            else:
                self._fast(g, main)
            self._lap("roots")
        self._lap("check")
        if (
            ROOT_ROOTS == "fast"
            and self.pool is not None
            and (self.steps >= ROOT_INLINE - 1)
            and ROOT_POWERS
        ):
            self._schedule()
        self._lap("schedule")

    @torch.no_grad()
    def begin(self):
        t0 = self._t0 = time.perf_counter()
        if self.groups is None:
            params = self.param_groups[0]["params"]
            self._setup(params[dist.get_rank() :: dist.get_world_size()])
        main = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev)
            self._lap("setup")
            root_flush(self)
            self._lap("flush")
            self.roots(main)
            self.done = torch.cuda.Event()
            self.done.record(self.side)
        self.begun = True
        dt = 1000.0 * (time.perf_counter() - t0)
        self.diag["host_ms"] += dt
        self.diag["host_max"] = max(self.diag["host_max"], dt)

    @torch.no_grad()
    def step(self):
        world, rank = dist.get_world_size(), dist.get_rank()
        if not self.begun:
            self.begin()
        self.begun = False
        torch.cuda.current_stream().wait_event(self.done)
        for group in self.param_groups:
            params = group["params"]
            padded = params + [torch.empty_like(params[-1])] * (world - len(params) % world)
            for first in range(0, len(params), world):
                if first + rank < len(params):
                    parameter = params[first + rank]
                    state = self.state[parameter]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(parameter)
                    roots, index = self.where[self.sources[parameter]]
                    inner = roots.Rb[ROOT_P_IN][index] if ROOT_P_IN else None
                    outer = roots.Rhat[ROOT_P_OUT][index] if ROOT_P_OUT else None
                    update = root_update(
                        parameter.grad,
                        state["momentum"],
                        inner,
                        outer,
                        group["mu"],
                        group["nesterov"],
                    )
                    if roots.P is not None:
                        parameter.sub_(
                            torch.mm(parameter.half(), roots.P[index], out_dtype=torch.float32),
                            alpha=group["lr"] * group["weight_decay"],
                        )
                    else:
                        parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                dist.all_gather(padded[first : first + world], padded[first + rank])
        self.steps += 1

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        self.side.synchronize()
        self.copy_stream.synchronize()
