"""Products whose result is known to be symmetric, computing only the tiles that meet the upper triangle and writing each
upper element also to its mirror (half the FLOPs of a full product; the result is exactly symmetric):
- mm16_sym: FP32-accurate products from roots.py's FP16 hi/lo splits (B = X^T (C X), R = (X Y) X^T), in its mm16
  convention A B = Ah Bh + (Ah Bl + Al Bh) / LO, the second sum as one product over the concatenated halves (K doubled);
- mm_sym: plain FP16 products with FP32 accumulation (the root iteration's O O and P W, roots.froot).
One accumulator per tile: the cross terms are accumulated first, scaled by 1 / LO (exact), and the hi products are
accumulated onto them (the same FP32 sum as hi + cross / LO up to the order of the additions)."""

import torch
import triton
import triton.language as tl

LO = 4096.0


@triton.jit
def _sym_kernel(
    A,
    B,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    NT: tl.constexpr,
    sab,
    sam,
    sak,
    sbb,
    sbk,
    sbn,
    SPLIT: tl.constexpr,
    A_LO_FIRST: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INV_LO: tl.constexpr,
):
    """C[b] (N x N, FP32) = A[b] B[b], the result being symmetric. SPLIT: A[b] is (N, 2K) split along K and B[b] (2K, N)
    (roots.mm16's layouts); else A[b] (N, K), B[b] (K, N). Tile (i, j) covers rows i BM .., columns j BN ..
    (j BN + BN > i BM) and stores its elements with row <= col at (row, col) and (col, row)."""
    b = tl.program_id(0) // NT
    t = tl.program_id(0) % NT
    ntn = N // BN
    i = 0
    rem = t
    cnt = ntn
    while rem >= cnt:
        rem -= cnt
        i += 1
        cnt = ntn - (i * BM) // BN
    j = (i * BM) // BN + rem
    m = i * BM + tl.arange(0, BM)
    n = j * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a_b = A + b * sab
    b_b = B + b * sbb
    acc = tl.zeros((BM, BN), tl.float32)
    if SPLIT:  # cross terms Ah Bl + Al Bh: one product over the concatenated halves (2K), then scaled by 1 / LO
        for kk in range(0, 2 * K, BK):
            a = tl.load(a_b + m[:, None] * sam + (kk + k)[None, :] * sak)
            c = tl.load(b_b + (kk + k)[:, None] * sbk + n[None, :] * sbn)
            acc = tl.dot(a, c, acc)
        acc = acc * INV_LO
    # the hi parts: A's hi half is its second half of K if A_LO_FIRST ([lo | hi]), else its first; B's the opposite
    a_hi: tl.constexpr = K if SPLIT and A_LO_FIRST else 0
    b_hi: tl.constexpr = K if SPLIT and not A_LO_FIRST else 0
    for kk in range(0, K, BK):
        a = tl.load(a_b + m[:, None] * sam + (a_hi + kk + k)[None, :] * sak)
        c = tl.load(b_b + (b_hi + kk + k)[:, None] * sbk + n[None, :] * sbn)
        acc = tl.dot(a, c, acc)
    up = m[:, None] <= n[None, :]
    cb = C + b * N * N
    tl.store(cb + m[:, None] * N + n[None, :], acc, up)
    tl.store(cb + n[None, :] * N + m[:, None], acc, up)


def _cfg(n):
    """Tiles for H100 (128 x 256, 8 warps: one 128-register accumulator per thread) or smaller GPUs (128 x 128)."""
    big = torch.cuda.get_device_capability()[0] >= 9 and n % 256 == 0
    return (128, 256, 32, 8, 4) if big else (128, 128, 32, 8, 3)


def _launch(A, B, C, n, k, split, a_lo_first):
    BM, BN, BK, w, st = _cfg(n)
    assert n % BN == 0 and n % BM == 0 and k % BK == 0
    nt = sum(n // BN - (i * BM) // BN for i in range(n // BM))
    _sym_kernel[(A.size(0) * nt,)](
        A,
        B,
        C,
        n,
        k,
        nt,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        split,
        bool(a_lo_first),
        BM,
        BN,
        BK,
        1.0 / LO,
        num_warps=w,
        num_stages=st,
    )
    return C


def mm16_sym(A2, B2, a_lo_first, out=None):
    """The FP32-accurate product of mm16(A2, B2, a_lo_first) for a result known to be symmetric (k, n, n), from its
    upper-triangle tiles (n a multiple of 128). A2 (k, n, 2n) and B2 (k, 2n, n) FP16, any strides."""
    k, n, n2 = A2.shape
    assert n2 == 2 * n and B2.shape == (k, 2 * n, n)
    assert A2.dtype == torch.float16 and B2.dtype == torch.float16
    C = out if out is not None else torch.empty(k, n, n, device=A2.device, dtype=torch.float32)
    return _launch(A2, B2, C, n, n, True, a_lo_first)


def mm_sym(A, B, out=None):
    """A @ B (FP16 inputs, FP32 accumulation and output) for a result known to be symmetric (k, n, n), from its
    upper-triangle tiles. A (k, n, kk), B (k, kk, n) FP16, any strides."""
    k, n, kk = A.shape
    assert B.shape == (k, kk, n) and A.dtype == torch.float16 and B.dtype == torch.float16
    C = out if out is not None else torch.empty(k, n, n, device=A.device, dtype=torch.float32)
    return _launch(A, B, C, n, kk, False, False)


def check():
    """mm16_sym vs roots.mm16 (full product, then symmetrized) and mm_sym vs torch.bmm, on random symmetric
    products: error vs FP64, exact symmetry, and the time."""
    import time
    import roots as PR

    def t_ms(f, reps=10):
        f()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0), "tiles", _cfg(3072))
    for k, n in ((24, 768), (8, 3072), (24, 512), (8, 2048)):
        X = torch.linalg.qr(torch.randn(k, n, n, device="cuda", dtype=torch.float64))[0].float()
        lam = torch.rand(k, n, device="cuda", dtype=torch.float64) ** 4 * 100
        C = ((X.double() * lam[:, None, :]) @ X.double().mT).float()
        Xb = torch.linalg.qr(X + 0.01 * torch.randn_like(X))[0]  # a nearby basis
        SB = PR.rows_lo(Xb)
        A2, mu = PR.scaled_cols_hi(C)
        Y = PR.mm16(A2, SB, False)
        Yh = PR.rows_hi(Y)
        full = PR.mm16(SB.mT, Yh, True)
        full = 0.5 * (full + full.mT)
        sym = mm16_sym(SB.mT, Yh, True)
        ref = Xb.double().mT @ (C.double() / mu.double()[:, None, None]) @ Xb.double()
        e_full = ((full.double() - ref).norm() / ref.norm()).item()
        e_sym = ((sym.double() - ref).norm() / ref.norm()).item()
        tf = t_ms(lambda: PR.mm16(SB.mT, Yh, True))
        ts = t_ms(lambda: mm16_sym(SB.mT, Yh, True))
        # the iteration's plain FP16 products: O O (O symmetric) and P W with W = D P + E P (P W symmetric)
        O = (0.01 * torch.randn(k, n, n, device="cuda")).half()
        O = (0.5 * (O.float() + O.float().mT)).half()
        o_ref = O.double() @ O.double()
        o_full = torch.bmm(O, O, out_dtype=torch.float32)
        o_sym = mm_sym(O, O)
        e_of = ((o_full.double() - o_ref).norm() / o_ref.norm()).item()
        e_os = ((o_sym.double() - o_ref).norm() / o_ref.norm()).item()
        tof = t_ms(lambda: torch.bmm(O, O, out_dtype=torch.float32))
        tos = t_ms(lambda: mm_sym(O, O))
        print(
            f"k={k} n={n}: split product rel err vs FP64 full {e_full:.2e} sym {e_sym:.2e}, asym "
            f"{(sym - sym.mT).abs().max().item():.0e}, time full {tf:.2f} ms sym {ts:.2f} ms | FP16 O O rel err full "
            f"{e_of:.2e} sym {e_os:.2e}, time full {tof:.2f} ms sym {tos:.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    check()
