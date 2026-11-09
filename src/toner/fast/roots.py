"""Inverse-power factors refined in a cached eigenbasis, with exact fallback."""

import os

import torch
import torch._dynamo

from .symm16 import mm16_sym, mm_sym

ALPHA, EPS = 0.25, 1e-3
QUARTER = 0.25  # the iteration's exponent for either root (alpha 1/2 squares its result)
U32 = 2.0**-24


def set_alpha(alpha):
    """The root's exponent, 1/4 (default) or 1/2: set once, before any root is computed."""
    global ALPHA
    assert alpha in (0.25, 0.5), f"roots: alpha 1/4 or 1/2 only (got {alpha})"
    ALPHA = float(alpha)


# the iteration's GEMMs multiply only off-diagonal (small) parts: FP16 inputs (11 significant bits, 8x bf16's; all
# entries are far inside FP16's range: |E|, |d P| <= n), FP32 accumulation
LOWP = torch.float16
LO = 4096.0  # the lo halves of the FP16 splits are stored times LO (no FP16 underflow); FP16 range needs |x| <= 2^14
# products with a symmetric result (B's and R's second products, the iteration's O O and P W) from their upper-triangle
# tiles (symm16.py): half their FLOPs, exactly symmetric results (TONER_SYM=0: full products, then symmetrized)
SYM = os.environ.get("TONER_SYM", "1") == "1"


def _sym_ok(n):
    return SYM and n % 256 == 0


class tf32_on:
    """TF32 tensor cores for this block only (the trainer keeps the global switch off because it changes the compiled
    model's autotune choices); restored on exit."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


def _hi_lo(x):
    h = x.half()
    return h, ((x - h.float()) * LO).half()


def _cols_hi(x):
    """A-side operand: [hi | lo] along the last dim."""
    h, lo = _hi_lo(x)
    return torch.cat([h, lo], -1)


def _cols_lo(x):
    """A-side operand: [lo | hi] along the last dim."""
    h, lo = _hi_lo(x)
    return torch.cat([lo, h], -1)


def _rows_hi(x):
    """B-side operand: [hi ; lo] along dim -2."""
    h, lo = _hi_lo(x)
    return torch.cat([h, lo], -2)


def _rows_lo(x):
    """B-side operand: [lo ; hi] along dim -2."""
    h, lo = _hi_lo(x)
    return torch.cat([lo, h], -2)


def _cols_hi_scaled(x):
    """[hi | lo] of x 2^k with k per matrix such that max |x 2^k| is in [2^13, 2^14) (any input scale), and 2^-k."""
    a = x.abs().amax((-2, -1), keepdim=True).clamp_min(1e-30)
    s = torch.exp2(13.0 - torch.floor(torch.log2(a)))
    h, lo = _hi_lo(x * s)
    return torch.cat([h, lo], -1), 1.0 / s


# Inductor keeps rounded intermediates in FP32 unless told to emulate the casts: lo = x - fp16(x) would be 0
_EXACT = {"emulate_precision_casts": True}
# static shapes: one compile per (batch, n, n) in use (groups, single statistics, the update's shape groups)
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
cols_hi = torch.compile(_cols_hi, dynamic=False, options=_EXACT)
cols_lo = torch.compile(_cols_lo, dynamic=False, options=_EXACT)
rows_hi = torch.compile(_rows_hi, dynamic=False, options=_EXACT)
rows_lo = torch.compile(_rows_lo, dynamic=False, options=_EXACT)
cols_hi_scaled = torch.compile(_cols_hi_scaled, dynamic=False, options=_EXACT)


def mm16(A2, B2, a_lo_first):
    """FP32-accurate A @ B from split operands in opposite orders: A2 = [lo | hi] and B2 = [hi ; lo] (a_lo_first), or
    A2 = [hi | lo] and B2 = [lo ; hi]. Strided / transposed views are used as they are (cuBLAS leading dimensions)."""
    k = A2.size(-1) // 2
    Ah, Bh = (A2[..., k:], B2[..., :k, :]) if a_lo_first else (A2[..., :k], B2[..., k:, :])
    if A2.dim() == 3:
        C = torch.bmm(Ah, Bh, out_dtype=torch.float32)
        return torch.baddbmm(C, A2, B2, alpha=1.0 / LO, out_dtype=torch.float32)
    C = torch.mm(Ah, Bh, out_dtype=torch.float32)
    return torch.addmm(C, A2, B2, alpha=1.0 / LO, out_dtype=torch.float32)


def apply16(A, R2):
    """A @ R, FP32-accurate, for any scale of A; R2 = rows_lo(R) (precomputed once per root)."""
    A2, inv = cols_hi_scaled(A)
    return mm16(A2, R2, False) * inv


def apply16_raw(A, R2):
    """apply16 for use inside a compiled function (the split is traced with it, so it fuses with its producer). Two
    bmm calls and an add (fused): torch 2.14's fake-tensor decomposition of baddbmm(out_dtype=) is broken."""
    A2, inv = _cols_hi_scaled(A)
    k = A2.size(-1) // 2
    C = torch.bmm(A2[..., :k], R2[..., k:, :], out_dtype=torch.float32)
    return (C + torch.bmm(A2, R2, out_dtype=torch.float32) * (1.0 / LO)) * inv


class Basis:
    """Orthonormal bases X (k, n, n), kept only as their FP16 splits in the two layouts the products need:
    SB = [lo ; hi] (rows) for M @ X, its transpose [lo | hi] for X^T @ M; S2 = [hi | lo] (cols) for X @ M, its
    transpose [hi ; lo] for M @ X^T."""

    def __init__(self, X):
        assert X.size(-1) <= 16384
        self.SB = rows_lo(X.contiguous())
        self.S2 = cols_hi(X.contiguous())

    @classmethod
    def of_splits(cls, S2):
        """A view for form() only (it reads S2): e.g. a subset of a group's bases."""
        b = cls.__new__(cls)
        b.S2 = S2
        return b

    def install(self, j, X):
        """Replace basis j (in place, keeping the tensors' storage)."""
        X = X.contiguous().unsqueeze(0)
        self.SB[j].copy_(rows_lo(X)[0])
        self.S2[j].copy_(cols_hi(X)[0])

    def install_many(self, X, mask):
        """Replace the bases where mask (k,) is set by those of X (k, n, n), in place: one call for a whole group."""
        install_masked(self.SB, self.S2, X, mask)


def _install_masked(SB, S2, X, mask):
    m = mask[:, None, None]
    SB.copy_(torch.where(m, _rows_lo(X), SB))
    S2.copy_(torch.where(m, _cols_hi(X), S2))


install_masked = torch.compile(_install_masked, dynamic=False, options=_EXACT)


def _scaled_cols_hi(C):
    n = C.size(-1)
    mu = C.diagonal(dim1=-2, dim2=-1).sum(-1) / n
    return _cols_hi(
        C / mu[..., None, None]
    ), mu  # entries of C / mu are <= n (its eigenvalues sum to n)


scaled_cols_hi = torch.compile(_scaled_cols_hi, dynamic=False, options=_EXACT)


def basis_transform(C, basis):
    """B = X^T (C / mu) X (FP32-accurate; EPS is added by froot), mu = tr(C) / n."""
    A2, mu = scaled_cols_hi(C)
    Y = mm16(A2, basis.SB, False)  # (C / mu) X
    if _sym_ok(C.size(-1)):
        return mm16_sym(basis.SB.mT, rows_hi(Y), True), mu  # X^T Y (symmetric)
    return mm16(basis.SB.mT, rows_hi(Y), True), mu  # X^T Y


def _square(lam, O, OO):
    """alpha 1/2: Y^2 = diag(lam^2) + (lam_i + lam_j) O + O O for Y = diag(lam) + O (O O: the small part's product)."""
    return lam.square(), (lam[..., :, None] + lam[..., None, :]) * O + OO


square = torch.compile(_square, dynamic=False)


def form(basis, lam, O):
    """R = X (diag(lam) + O) X^T, FP32-accurate, symmetrized; alpha 1/2: R = X Y^2 X^T for Y = diag(lam) + O."""
    if ALPHA == 0.5:
        Ob = O.to(LOWP)
        lam, O = square(
            lam, O, mm_sym(Ob, Ob) if _sym_ok(lam.size(-1)) and O.dim() == 3 else _bmm32(Ob, Ob)
        )
    XY = mm16(basis.S2, rows_lo(torch.diag_embed(lam) + O), False)
    if _sym_ok(lam.size(-1)):
        return mm16_sym(cols_lo(XY), basis.S2.mT, True)
    R = mm16(cols_lo(XY), basis.S2.mT, True)
    return 0.5 * (R + R.mT)


def _start(B, freeze):
    """Symmetrize B, add EPS; Daleckii-Krein start (one preconditioned step from the diagonal); (noise) pairs."""
    B = 0.5 * (B + B.mT)
    dB = B.diagonal(dim1=-2, dim2=-1)
    d = dB + EPS
    E = B - torch.diag_embed(dB)
    n = B.size(-1)
    tau = freeze * n**0.5 * U32 * d.amax(-1, keepdim=True)
    small = d < tau
    frozen = small[..., :, None] & small[..., None, :]
    rel = E.abs() / (d[..., :, None] + d[..., None, :])
    rho = rel.masked_fill(frozen, 0.0).amax((-2, -1))
    lam = d.clamp_min(EPS * 1e-3).pow(-QUARTER)
    dd = d[..., :, None] - d[..., None, :]
    mid = 0.5 * (d[..., :, None] + d[..., None, :])
    L = torch.where(
        dd.abs() > 1e-6 * mid,
        (lam[..., :, None] - lam[..., None, :]) / torch.where(dd == 0, 1.0, dd),
        -QUARTER * mid.pow(-QUARTER - 1),
    )
    O = L * E
    return (
        d,
        E,
        E.to(LOWP),
        lam,
        O,
        O.to(LOWP),
        d.sqrt(),
        small,
        rho,
        O.square().sum((-2, -1)).sqrt(),
    )


def _iter_mid(lam, O, OO):
    """P = Y^2 - diag(lam^2) = (lam_i + lam_j) O + O O."""
    P = (lam[..., :, None] + lam[..., None, :]) * O + OO
    return P, P.to(LOWP)


def _iter_w(P, d, EP):
    """W = D P + E P, so P W = P B P (the second-order part of Y^2 B Y^2)."""
    return (d[..., :, None] * P + EP).to(LOWP)


def _iter_post(lam, O, d, E, P, EP, PW, lam0, sq, small, pw_sym: bool = False):
    """G = I - Y^2 B Y^2 from its parts (first-order terms exact FP32 elementwise); preconditioned update. pw_sym: PW
    is exactly symmetric (symm16.mm_sym), so its symmetrization (a transposed read) is skipped."""
    l2 = lam.square()
    F = (
        l2[..., :, None] * E * l2[..., None, :]
        + (l2 * d)[..., :, None] * P
        + P * (d * l2)[..., None, :]
        + l2[..., :, None] * EP
        + EP.mT * l2[..., None, :]
        + (PW if pw_sym else 0.5 * (PW + PW.mT))
    )
    G = torch.diag_embed(1.0 - l2 * d * l2) - F
    pre = (lam0[..., :, None] + lam0[..., None, :]) * (sq[..., :, None] + sq[..., None, :])
    Dl = (G / pre).masked_fill(small[..., :, None] & small[..., None, :], 0.0)
    dg = Dl.diagonal(dim1=-2, dim2=-1)
    O = O + Dl - torch.diag_embed(dg)
    return lam + dg, O, O.to(LOWP), Dl.square().sum((-2, -1)).sqrt()


start = torch.compile(_start, dynamic=False)
iter_mid = torch.compile(_iter_mid, dynamic=False)
iter_w = torch.compile(_iter_w, dynamic=False)
iter_post = torch.compile(_iter_post, dynamic=False)


def _bmm32(a, b):
    """Low-precision (LOWP) product with FP32 output (no separate conversion pass)."""
    return (
        torch.bmm(a, b, out_dtype=torch.float32)
        if a.dim() == 3
        else torch.mm(a, b, out_dtype=torch.float32)
    )


def froot(B, iters, freeze=0.0):
    """Y = (B + EPS I)^(-1/4) = diag(lam) + O for near-diagonal B (k, n, n) (B without EPS). Returns lam (k, n),
    O (k, n, n), rho (k,) the relative off-diagonality over iterated pairs, est (k,) the estimated relative Frobenius
    error of Y (= of R) after the last update: linear convergence, e_m ~ r / (1 - r) |Delta_m| with
    r = |Delta_m| / |Delta_{m-1}| capped at 0.95 (an update at rounding level is accepted even when the ratio of two
    rounding-level updates is ~1); the start counts as the first update. alpha 1/2: est is doubled (form squares Y:
    Y^2's relative error is ~2x Y's)."""
    lam, O, rho, est, _ = froot_state(B, iters, freeze)
    return lam, O, rho, est


def _iterate(d, E, Eb, lam, O, Ob, sq, small, lam0, iters):
    """iters fixed-point updates from (lam, O); returns lam, O, Ob and the updates' norms."""
    sym = _sym_ok(E.size(-1)) and E.dim() == 3
    dn = []
    for _ in range(iters):
        P, Pb = iter_mid(
            lam, O, mm_sym(Ob, Ob) if sym else _bmm32(Ob, Ob)
        )  # O symmetric: O O symmetric
        EP = _bmm32(Eb, Pb)
        W = iter_w(P, d, EP)
        PW = mm_sym(Pb, W) if sym else _bmm32(Pb, W)  # P (D + E) P: symmetric
        lam, O, Ob, upd = iter_post(lam, O, d, E, P, EP, PW, lam0, sq, small, sym)
        dn.append(upd)
    return lam, O, Ob, dn


def _estimate(lam, dprev, dlast):
    r = (dlast / dprev.clamp_min(1e-30)).clamp_max(0.95)
    return r / (1 - r) * dlast / lam.square().sum(-1).sqrt()


def froot_state(B, iters, freeze=0.0):
    """froot, also returning the iteration's state for froot_more: (lam, O, the last update's norm)."""
    d, E, Eb, lam, O, Ob, sq, small, rho, dn0 = start(B, freeze)
    lam0 = lam
    lam, O, Ob, dn = _iterate(d, E, Eb, lam, O, Ob, sq, small, lam0, iters)
    dn = [dn0] + dn
    est = _estimate(lam, dn[-2], dn[-1]) if iters >= 1 else torch.full_like(rho, float("inf"))
    if ALPHA == 0.5:
        est = 2 * est
    return lam, O, rho, est, (lam, O, dn[-1])


def froot_more(B, state, iters, freeze=0.0):
    """Continue froot_state's iteration for iters more updates (B and state of a subset of statistics, indexed alike):
    the same iterates as froot with the total number of iterations. The start's elementwise quantities are recomputed
    from B. Returns lam, O, rho, est."""
    lam, O, dlast = state
    d, E, Eb, _, _, _, sq, small, rho, _ = start(B, freeze)
    lam0 = d.clamp_min(EPS * 1e-3).pow(-QUARTER)
    lam, O, Ob, dn = _iterate(d, E, Eb, lam, O, O.to(LOWP), sq, small, lam0, iters)
    dn = [dlast] + dn
    est = _estimate(lam, dn[-2], dn[-1])
    return lam, O, rho, (2 * est if ALPHA == 0.5 else est), dn[-1]


U32_FLOOR = 2.0**-24


def floor_base(lam):
    """Heuristic FP32 root-error scale from the normalized statistic eigenvalues."""
    x = lam.double().clamp_min(0)
    x = x / x.mean(-1, keepdim=True).clamp_min(1e-30)
    f = (x + EPS).pow(-ALPHA)
    fp = ALPHA * (x + EPS).pow(-ALPHA - 1)
    return (U32_FLOOR * x.amax(-1) * fp.norm(dim=-1) / f.norm(dim=-1)).float()


def exact_eigh(C, with_eigenvalues=False):
    """Compute eigenvectors, inverse-power factors and eigenvalues by FP32 eigendecomposition."""
    lam, V = torch.linalg.eigh(0.5 * (C + C.mT))
    if with_eigenvalues:
        return (V, _exact_root(lam, V, C)), lam
    return V, _exact_root(lam, V, C)


def _exact_root(lam, V, C):
    x = lam.clamp_min(0)
    x = x / x.mean(-1, keepdim=True)
    f = (x + EPS).pow(-ALPHA)
    if _sym_ok(C.size(-1)) and C.dim() == 3:
        return mm16_sym(cols_hi(V * f[..., None, :]), rows_lo(V.mT.contiguous()), False)
    R = mm16(cols_hi(V * f[..., None, :]), rows_lo(V.mT.contiguous()), False)
    return 0.5 * (R + R.mT)


# ---------------------------------------------------------------------------------------------- split inner / outer
# MuonT with separate inner and outer strengths (train.py --toner-alpha-in / --toner-alpha-out, engine.py): both roots
# come from the same quarter iteration Y = B^(-1/4) = diag(lam) + O. R_1/4 = X Y X^T; R_1/2 = X Y^2 X^T (as form() at
# alpha 1/2); R_1/8 = X Z X^T with Z = Y^(1/2) = diag(s) + Q by a Daleckii-Krein start (s = sqrt(lam),
# Q = O / (s_i + s_j)) and SQRT_ITERS fixed-point updates Z += (Y - Z^2) / (s0_i + s0_j) (first-order terms exact FP32
# elementwise, Q Q an FP16 product of the small part, as in the quarter iteration). The iteration and its estimate
# run at 1/4 (set_powers sets ALPHA = 1/4); engine.py holds each root to its own tolerance (floor_base_p): R_1/2's
# error ~2 est(Y), R_1/4's est(Y), R_1/8's ~est(Y) / 2 + est(sqrt).
POWERS = None  # the split rule's root exponents (sorted, distinct); None: ALPHA's single root (form / exact_eigh)
SQRT_ITERS = (
    2  # the 1/8 root's square-root iterations (4 changed nothing in the 60-step end-to-end test)
)


def floor_base_p(lam, p):
    """floor_base for exponent p (the split rule's per-root tolerances; floor_base(lam) = floor_base_p(lam, ALPHA))."""
    x = lam.double().clamp_min(0)
    x = x / x.mean(-1, keepdim=True).clamp_min(1e-30)
    f = (x + EPS).pow(-p)
    fp = p * (x + EPS).pow(-p - 1)
    return (U32_FLOOR * x.amax(-1) * fp.norm(dim=-1) / f.norm(dim=-1)).float()


def set_powers(powers):
    """Split rule: the exponents to form (1/8, 1/4, 1/2), once, before any root; the iteration runs at 1/4."""
    global POWERS
    powers = tuple(sorted(set(float(p) for p in powers)))
    assert set(powers) <= {0.125, 0.25, 0.5}, (
        f"roots: split powers 1/8, 1/4, 1/2 only (got {powers})"
    )
    set_alpha(0.25)
    POWERS = powers


def _sqrt_start(lam, O):
    """Z = Y^(1/2) for Y = diag(lam) + O (O off-diagonal): Daleckii-Krein start s = sqrt(lam), Q = O / (s_i + s_j)."""
    s = lam.clamp_min(0).sqrt()
    S0 = s[..., :, None] + s[..., None, :]
    Q = O / S0
    return s, Q, Q.to(LOWP), S0, Q.square().sum((-2, -1)).sqrt()


def _sqrt_post(lam, O, s, Q, QQ, S0):
    """G = Y - Z^2 for Z = diag(s) + Q (first-order terms exact FP32 elementwise; QQ the small part's product);
    preconditioned update Z += G / (s0_i + s0_j)."""
    S = s[..., :, None] + s[..., None, :]
    G = torch.diag_embed(lam - s.square()) + O - S * Q - QQ
    Dl = G / S0
    dg = Dl.diagonal(dim1=-2, dim2=-1)
    Q = Q + Dl - torch.diag_embed(dg)
    return s + dg, Q, Q.to(LOWP), Dl.square().sum((-2, -1)).sqrt()


sqrt_start = torch.compile(_sqrt_start, dynamic=False)
sqrt_post = torch.compile(_sqrt_post, dynamic=False)


def ysqrt(lam, O, iters=None):
    """Z = Y^(1/2) = diag(s) + Q for the quarter root Y = diag(lam) + O; returns s, Q and the square-root iteration's
    estimated relative error (relative to Z)."""
    s, Q, Qb, S0, dn0 = sqrt_start(lam, O)
    sym = _sym_ok(lam.size(-1)) and O.dim() == 3
    dn = [dn0]
    for _ in range(SQRT_ITERS if iters is None else iters):
        s, Q, Qb, upd = sqrt_post(lam, O, s, Q, mm_sym(Qb, Qb) if sym else _bmm32(Qb, Qb), S0)
        dn.append(upd)
    return s, Q, _estimate(s, dn[-2], dn[-1])


def _formation(basis, lam, O):
    """R = X (diag(lam) + O) X^T, FP32-accurate, symmetrized (form()'s formation)."""
    XY = mm16(basis.S2, rows_lo(torch.diag_embed(lam) + O), False)
    if _sym_ok(lam.size(-1)):
        return mm16_sym(cols_lo(XY), basis.S2.mT, True)
    R = mm16(cols_lo(XY), basis.S2.mT, True)
    return 0.5 * (R + R.mT)


def form_powers(basis, lam, O):
    """Split rule: {p: R_p} for p in POWERS from the quarter root Y = diag(lam) + O, and the 1/8 root's square-root
    estimate (None without 1/8)."""
    out, est_s = {}, None
    for p in POWERS:
        if p == 0.25:
            out[p] = _formation(basis, lam, O)
        elif p == 0.5:
            Ob = O.to(LOWP)
            l2, P = square(
                lam, O, mm_sym(Ob, Ob) if _sym_ok(lam.size(-1)) and O.dim() == 3 else _bmm32(Ob, Ob)
            )
            out[p] = _formation(basis, l2, P)
        else:
            s, Q, est_s = ysqrt(lam, O)
            out[p] = _formation(basis, s, Q)
    return out, est_s


def exact_eigh_powers(C, with_eigenvalues=False):
    """Compute inverse factors for several strengths from one eigendecomposition."""
    lam, V = torch.linalg.eigh(0.5 * (C + C.mT))
    Rs = {p: _exact_root_p(lam, V, C, p) for p in POWERS}
    if with_eigenvalues:
        return (V, Rs), lam
    return V, Rs


def _exact_root_p(lam, V, C, p):
    x = lam.clamp_min(0)
    x = x / x.mean(-1, keepdim=True)
    f = (x + EPS).pow(-p)
    if _sym_ok(C.size(-1)) and C.dim() == 3:
        return mm16_sym(cols_hi(V * f[..., None, :]), rows_lo(V.mT.contiguous()), False)
    R = mm16(cols_hi(V * f[..., None, :]), rows_lo(V.mT.contiguous()), False)
    return 0.5 * (R + R.mT)
