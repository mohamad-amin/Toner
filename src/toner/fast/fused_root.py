"""Fused fixed-point kernels for inverse fourth-root refinement."""

import os
import torch
import triton
import triton.language as tl
from . import roots as PR


@triton.jit
def _symmetric(
    A,
    BB,
    Lam,
    O,
    D,
    E,
    P,
    EP,
    L0,
    Sq,
    Small,
    Out,
    OutHalf,
    Lout,
    Dparts,
    N: tl.constexpr,
    NT: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(0) // NT
    t = tl.program_id(0) % NT
    rem = t
    i = 0
    cnt = N // BN
    while rem >= cnt:
        rem -= cnt
        i += 1
        cnt = N // BN - i * BM // BN
    j = i * BM // BN + rem
    m = i * BM + tl.arange(0, BM)
    n = j * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    base = batch * N * N
    vb = batch * N
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, N, BK):
        aa = tl.load(A + base + m[:, None] * N + (kk + k)[None, :])
        bb = tl.load(BB + base + (kk + k)[:, None] * N + n[None, :])
        acc = tl.dot(aa, bb, acc)
    lm = tl.load(Lam + vb + m)
    ln = tl.load(Lam + vb + n)
    om = tl.load(O + base + m[:, None] * N + n[None, :])
    upper = m[:, None] <= n[None, :]
    if MODE == 0:
        val = (lm[:, None] + ln[None, :]) * om + acc
    else:
        dm = tl.load(D + vb + m)
        dn = tl.load(D + vb + n)
        l2m, l2n = lm * lm, ln * ln
        ee = tl.load(E + base + m[:, None] * N + n[None, :])
        pp = tl.load(P + base + m[:, None] * N + n[None, :])
        ep = tl.load(EP + base + m[:, None] * N + n[None, :])
        ept = tl.load(EP + base + n[None, :] * N + m[:, None])
        f = l2m[:, None] * ee * l2n[None, :] + (l2m * dm)[:, None] * pp
        f = f + pp * (dn * l2n)[None, :] + l2m[:, None] * ep + ept * l2n[None, :] + acc
        diag = m[:, None] == n[None, :]
        g = tl.where(diag, (1.0 - l2m * dm * l2m)[:, None], 0.0) - f
        l0m, l0n = tl.load(L0 + vb + m), tl.load(L0 + vb + n)
        sm, sn = tl.load(Sq + vb + m), tl.load(Sq + vb + n)
        frozen = tl.load(Small + vb + m)[:, None] & tl.load(Small + vb + n)[None, :]
        dl = tl.where(
            frozen, 0.0, g / ((l0m[:, None] + l0n[None, :]) * (sm[:, None] + sn[None, :]))
        )
        val = tl.where(diag, 0.0, om + dl)
        dl_diag = tl.sum(tl.where(diag, dl, 0.0), 1)
        tl.store(Lout + vb + m, lm + dl_diag, (m >= j * BN) & (m < (j + 1) * BN))
        mass = tl.where(upper, dl * dl * tl.where(diag, 1.0, 2.0), 0.0)
        total = tl.sum(tl.sum(mass, 1), 0)
        tl.store(Dparts + batch * NT + t, total)
    low = val.to(tl.float16)
    tl.store(Out + base + m[:, None] * N + n[None, :], val, upper)
    tl.store(Out + base + n[None, :] * N + m[:, None], val, upper)
    tl.store(OutHalf + base + m[:, None] * N + n[None, :], low, upper)
    tl.store(OutHalf + base + n[None, :] * N + m[:, None], low, upper)


@triton.jit
def _ep_w(
    Eb, Pb, P, D, EP, W, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    batch = tl.program_id(0)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    base = batch * N * N
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(0, N, BK):
        a = tl.load(Eb + base + m[:, None] * N + (kk + k)[None, :])
        b = tl.load(Pb + base + (kk + k)[:, None] * N + n[None, :])
        acc = tl.dot(a, b, acc)
    d = tl.load(D + batch * N + m)
    p = tl.load(P + base + m[:, None] * N + n[None, :])
    w = (d[:, None] * p + acc).to(tl.float16)
    tl.store(EP + base + m[:, None] * N + n[None, :], acc)
    tl.store(W + base + m[:, None] * N + n[None, :], w)


@torch.compile(dynamic=False)
def _norm(parts):
    return parts.sum(-1).sqrt()


def _iterate(d, e, eb, lam, o, ob, sq, small, lam0, iters):
    assert o.is_contiguous() and o.ndim == 3
    batch, n, _ = o.shape
    bm = int(os.environ.get("FUSED_ROOT_BM", 128))
    bn = int(os.environ.get("FUSED_ROOT_BN", 128))
    bk = 32
    assert n % bm == 0 and n % bn == 0 and n % bk == 0
    nt = sum(n // bn - i * bm // bn for i in range(n // bm))
    nw = 8 if bm * bn >= 16384 else 4
    fused = set(os.environ.get("FUSED_ROOT_PARTS", "mid,post").split(","))
    upd = []
    for _ in range(iters):
        p, ep, new_o = (torch.empty_like(o) for _ in range(3))
        pb, w, new_ob = (torch.empty_like(ob) for _ in range(3))
        new_lam = torch.empty_like(lam)
        parts = torch.empty(batch, nt, device=o.device)
        if "mid" in fused:
            _symmetric[(batch * nt,)](
                ob,
                ob,
                lam,
                o,
                d,
                e,
                p,
                ep,
                lam0,
                sq,
                small,
                p,
                pb,
                new_lam,
                parts,
                n,
                nt,
                0,
                bm,
                bn,
                bk,
                num_warps=nw,
                num_stages=3,
                enable_fp_fusion=False,
            )
        else:
            p, pb = PR.iter_mid(lam, o, PR.mm_sym(ob, ob))
        if "ep" in fused:
            _ep_w[(batch, n // bm, n // bn)](
                eb,
                pb,
                p,
                d,
                ep,
                w,
                n,
                bm,
                bn,
                bk,
                num_warps=nw,
                num_stages=3,
                enable_fp_fusion=False,
            )
        else:
            ep = PR._bmm32(eb, pb)
            w = PR.iter_w(p, d, ep)
        if "post" in fused:
            _symmetric[(batch * nt,)](
                pb,
                w,
                lam,
                o,
                d,
                e,
                p,
                ep,
                lam0,
                sq,
                small,
                new_o,
                new_ob,
                new_lam,
                parts,
                n,
                nt,
                1,
                bm,
                bn,
                bk,
                num_warps=nw,
                num_stages=3,
                enable_fp_fusion=False,
            )
            lam, o, ob = new_lam, new_o, new_ob
            upd.append(_norm(parts))
        else:
            pw = PR.mm_sym(pb, w)
            lam, o, ob, dn = PR.iter_post(lam, o, d, e, p, ep, pw, lam0, sq, small, True)
            upd.append(dn)
    return lam, o, ob, upd


def froot_state(b, iters, freeze=0.0):
    d, e, eb, lam, o, ob, sq, small, rho, dn0 = PR.start(b, freeze)
    lam, o, ob, dn = _iterate(d, e, eb, lam, o, ob, sq, small, lam, iters)
    dn = [dn0] + dn
    est = PR._estimate(lam, dn[-2], dn[-1]) if iters else torch.full_like(rho, float("inf"))
    return lam, o, rho, est, (lam, o, dn[-1])


def froot_more(b, state, iters, freeze=0.0):
    lam, o, dlast = state
    d, e, eb, lam0, _, _, sq, small, rho, _ = PR.start(b, freeze)
    lam, o, ob, dn = _iterate(d, e, eb, lam, o, o.half(), sq, small, lam0, iters)
    dn = [dlast] + dn
    return lam, o, rho, PR._estimate(lam, dn[-2], dn[-1]), dn[-1]
