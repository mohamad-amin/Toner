"""Batched symmetric Gram products of BF16 rows with FP32 accumulation and packed output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _syrk_kernel(X, S, R, n, nt, sxb, sxr, ssb, ssr, BM: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # tile index t -> (i, j) with i <= j, row-major over the upper triangle
    i = 0
    rem = t
    while rem >= nt - i:
        rem -= nt - i
        i += 1
    j = i + rem
    offs_m = i * BM + tl.arange(0, BM)
    offs_n = j * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    xb = X + b * sxb
    for r in range(0, R, BK):
        rows = r + offs_k
        a = tl.load(xb + rows[:, None] * sxr + offs_m[None, :], mask=rows[:, None] < R, other=0.0)
        c = tl.load(xb + rows[:, None] * sxr + offs_n[None, :], mask=rows[:, None] < R, other=0.0)
        acc = tl.dot(tl.trans(a), c, acc)
    sb = S + b * ssb
    tl.store(sb + offs_m[:, None] * ssr + offs_n[None, :], acc)
    if i != j:
        tl.store(sb + offs_n[:, None] * ssr + offs_m[None, :], tl.trans(acc))


@triton.jit
def _syrk_ptr_kernel(X0, OFFS, S, R, n, nt, ntt, sxr, ssb, ssr, BM: tl.constexpr, BK: tl.constexpr):
    """Compute Gram tiles from separately allocated row buffers, grouped by statistic."""
    b = tl.program_id(0) // ntt
    t = tl.program_id(0) % ntt
    i = 0
    rem = t
    while rem >= nt - i:
        rem -= nt - i
        i += 1
    j = i + rem
    offs_m = i * BM + tl.arange(0, BM)
    offs_n = j * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    xb = X0 + tl.multiple_of(tl.load(OFFS + b), 256)
    for r in range(0, R, BK):
        rows = r + offs_k
        a = tl.load(xb + rows[:, None] * sxr + offs_m[None, :], mask=rows[:, None] < R, other=0.0)
        c = tl.load(xb + rows[:, None] * sxr + offs_n[None, :], mask=rows[:, None] < R, other=0.0)
        acc = tl.dot(tl.trans(a), c, acc)
    sb = S + b * ssb
    tl.store(sb + offs_m[:, None] * ssr + offs_n[None, :], acc)
    if i != j:
        tl.store(sb + offs_n[:, None] * ssr + offs_m[None, :], tl.trans(acc))


def list_offsets(xs):
    """Element offsets of each tensor's data from xs[0]'s (int64, on the device), for syrk_list."""
    p0 = xs[0].data_ptr()
    offs = [(x.data_ptr() - p0) for x in xs]
    assert all(o % 512 == 0 for o in offs), "rows buffers must be 512-byte aligned"
    return torch.tensor(
        [o // xs[0].element_size() for o in offs], dtype=torch.int64, device=xs[0].device
    )


def syrk_list(xs, offs=None, BM=128, BK=64, num_warps=4, num_stages=3, out=None):
    """xs: list of k (R, n) bf16 contiguous tensors (same shape), offs: list_offsets(xs) (computed if None).
    Returns (k, n, n) fp32 with S[b] = xs[b]^T xs[b] (written into `out` if given)."""
    k = len(xs)
    R, n = xs[0].shape
    assert (
        all(x.shape == (R, n) and x.is_contiguous() and x.dtype == torch.bfloat16 for x in xs)
        and n % BM == 0
    )
    if offs is None:
        offs = list_offsets(xs)
    S = out if out is not None else torch.empty(k, n, n, device=xs[0].device, dtype=torch.float32)
    assert S.shape == (k, n, n) and S.is_contiguous() and S.dtype == torch.float32
    nt = n // BM
    ntt = nt * (nt + 1) // 2
    _syrk_ptr_kernel[(k * ntt,)](
        xs[0],
        offs,
        S,
        R,
        n,
        nt,
        ntt,
        xs[0].stride(0),
        S.stride(0),
        S.stride(1),
        BM=BM,
        BK=BK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return S


@triton.jit
def _syrk_packed_kernel(
    X0,
    OFFS,
    S,
    R: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    ACC: tl.constexpr,
):
    """Compute the packed upper triangle of each statistic's Gram matrix."""
    nt = tl.cdiv(N, BM)
    ntt = nt * (nt + 1) // 2
    b = tl.program_id(0) // ntt
    t = tl.program_id(0) % ntt
    i = 0
    rem = t
    while rem >= nt - i:
        rem -= nt - i
        i += 1
    j = i + rem
    m = i * BM + tl.arange(0, BM)
    n = j * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    base = X0 + tl.multiple_of(tl.load(OFFS + b), 256)
    acc = tl.zeros((BM, BM), tl.float32)
    for r in range(tl.cdiv(R, BK)):
        rr = r * BK + k
        a = tl.load(
            base + rr[:, None] * N + m[None, :], (rr[:, None] < R) & (m[None, :] < N), other=0
        )
        c = tl.load(
            base + rr[:, None] * N + n[None, :], (rr[:, None] < R) & (n[None, :] < N), other=0
        )
        acc = tl.dot(tl.trans(a), c, acc)
    off = m[:, None] * N - m[:, None] * (m[:, None] + 1) // 2 + n[None, :]
    keep = (m[:, None] <= n[None, :]) & (n[None, :] < N)
    if ACC:  # add to the Grams already there (a chunk of the step's rows: --toner-chunk)
        acc += tl.load(S + b * (N * (N + 1) // 2) + off, keep, other=0.0)
    tl.store(S + b * (N * (N + 1) // 2) + off, acc, keep)


@triton.jit
def _syrk_packed_rect_kernel(
    X0,
    OFFS,
    S,
    R: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACC: tl.constexpr,
):
    """As _syrk_packed_kernel with BM x BN tiles (the high-throughput Hopper GEMM shape 128 x 256, 8 warps): row block i
    covers the column blocks j >= i BM / BN, the tiles that meet the upper triangle, and stores its elements with
    row <= column. NT: tiles per statistic."""
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
    base = X0 + tl.multiple_of(tl.load(OFFS + b), 256)
    acc = tl.zeros((BM, BN), tl.float32)
    for r in range(tl.cdiv(R, BK)):
        rr = r * BK + k
        a = tl.load(base + rr[:, None] * N + m[None, :], rr[:, None] < R, other=0)
        c = tl.load(base + rr[:, None] * N + n[None, :], rr[:, None] < R, other=0)
        acc = tl.dot(tl.trans(a), c, acc)
    off = m[:, None] * N - m[:, None] * (m[:, None] + 1) // 2 + n[None, :]
    if ACC:
        acc += tl.load(S + b * (N * (N + 1) // 2) + off, m[:, None] <= n[None, :], other=0.0)
    tl.store(S + b * (N * (N + 1) // 2) + off, acc, m[:, None] <= n[None, :])


def syrk_list_packed(xs, offs, out, BM=128, BK=64, num_warps=4, num_stages=3, rect=True, acc=False):
    """xs: list of k (R, n) bf16 contiguous tensors; out: (k, n (n + 1) / 2) fp32, the packed upper triangles of
    xs[b]^T xs[b]. rect (n a multiple of 256): 128 x 256 tiles. Measured 10-09 on H100 (bench_syrk_packed.py), all 128
    statistics of a rank at 2M tokens per step: 9.77 vs 12.39 ms at 4 GPUs (36,608 rows), 4.91 vs 6.01 ms at 8 (18,304),
    ~700 vs ~520 TFLOP/s of MMA work; the same Grams bit for bit (the same fp32 summation order). acc: add to out
    (a chunk of the step's rows, --toner-chunk; FP32 sums of the same products in another order)."""
    k = len(xs)
    R, n = xs[0].shape
    assert out.shape == (k, n * (n + 1) // 2) and out.dtype == torch.float32 and out.is_contiguous()
    if (
        rect and n % 256 == 0 and torch.cuda.get_device_capability(out.device)[0] >= 9
    ):  # (smem: Hopper)
        # 300M (10-09, H100, 36,608 rows, bench_syrk_packed.py): the 32 n = 3072 statistics take 19.58 ms with 256 x 128
        # tiles (BK 64, 3 stages) vs 21.94 ms with 128 x 256 (BK 32, 5 stages), bitwise the same Grams
        BM, BN, BK, st = (256, 128, 64, 3) if n >= 3072 else (128, 256, 32, 5)
        nt = sum(n // BN - (i * BM) // BN for i in range(n // BM))
        _syrk_packed_rect_kernel[(k * nt,)](
            xs[0], offs, out, R, n, nt, BM, BN, BK, bool(acc), num_warps=8, num_stages=st
        )
        return out
    assert n % BM == 0
    nt = n // BM
    _syrk_packed_kernel[(k * nt * (nt + 1) // 2,)](
        xs[0], offs, out, R, n, BM, BK, bool(acc), num_warps=num_warps, num_stages=num_stages
    )
    return out


@triton.jit
def _unpack_ema_kernel(
    P, C, N: tl.constexpr, BETA: tl.constexpr, GAMMA: tl.constexpr, BLOCK: tl.constexpr
):
    b = tl.program_id(
        1
    )  # (the long axis on grid x: n^2 / BLOCK exceeds grid y's 65,535 at n = 4096)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = x // N
    j = x % N
    lo = tl.minimum(i, j)
    hi = tl.maximum(i, j)
    val = tl.load(P + b * (N * (N + 1) // 2) + lo * N - lo * (lo + 1) // 2 + hi, x < N * N, other=0)
    old = tl.load(C + b * N * N + x, x < N * N, other=0) * BETA
    tl.store(
        C + b * N * N + x, tl.fma(val, GAMMA, old), x < N * N
    )  # as C.mul_(beta).add_(S, alpha=gamma)


def unpack_ema_(packed, C, beta, gamma):
    """In place: C[b] = beta C[b] + gamma S[b] for the symmetric S[b] given by its packed upper triangle."""
    n = C.size(-1)
    assert C.is_contiguous() and packed.is_contiguous() and C.shape == (packed.size(0), n, n)
    assert packed.shape[-1] == n * (n + 1) // 2
    _unpack_ema_kernel[(triton.cdiv(n * n, 256), C.size(0))](
        packed, C, n, float(beta), float(gamma), 256, enable_fp_fusion=False
    )
    return C


@triton.jit
def _syrk_ema_kernel(
    X0, OFFS, C, R, n, nt, sxr, scb, scr, beta, gamma, BM: tl.constexpr, BK: tl.constexpr
):
    """C[b] <- beta C[b] + gamma X_b^T X_b for symmetric C (rows of X_b at X0 + OFFS[b]): each upper tile is read once and
    written with its transpose, so C stays exactly symmetric off the diagonal tiles."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    i = 0
    rem = t
    while rem >= nt - i:
        rem -= nt - i
        i += 1
    j = i + rem
    offs_m = i * BM + tl.arange(0, BM)
    offs_n = j * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    xb = X0 + tl.multiple_of(tl.load(OFFS + b), 256)
    for r in range(0, R, BK):
        rows = r + offs_k
        a = tl.load(xb + rows[:, None] * sxr + offs_m[None, :], mask=rows[:, None] < R, other=0.0)
        c = tl.load(xb + rows[:, None] * sxr + offs_n[None, :], mask=rows[:, None] < R, other=0.0)
        acc = tl.dot(tl.trans(a), c, acc)
    cb = C + b * scb
    new = tl.load(cb + offs_m[:, None] * scr + offs_n[None, :]) * beta + acc * gamma
    tl.store(cb + offs_m[:, None] * scr + offs_n[None, :], new)
    if i != j:
        tl.store(cb + offs_n[:, None] * scr + offs_m[None, :], tl.trans(new))


def syrk_ema(xs, offs, C, beta, gamma, BM=128, BK=64, num_warps=4, num_stages=3):
    """In place: C[b] = beta C[b] + gamma xs[b]^T xs[b] (C: (k, n, n) fp32 contiguous, symmetric)."""
    k = len(xs)
    R, n = xs[0].shape
    assert C.shape == (k, n, n) and C.is_contiguous() and C.dtype == torch.float32 and n % BM == 0
    nt = n // BM
    _syrk_ema_kernel[(k, nt * (nt + 1) // 2)](
        xs[0],
        offs,
        C,
        R,
        n,
        nt,
        xs[0].stride(0),
        C.stride(0),
        C.stride(1),
        float(beta),
        float(gamma),
        BM=BM,
        BK=BK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


def syrk(X, BM=128, BK=64, num_warps=8, num_stages=3):
    """X: (k, R, n) bf16 contiguous, n a multiple of BM. Returns (k, n, n) fp32 = X^T X per batch entry."""
    k, R, n = X.shape
    assert X.is_contiguous() and X.dtype == torch.bfloat16 and n % BM == 0
    S = torch.empty(k, n, n, device=X.device, dtype=torch.float32)
    nt = n // BM
    grid = (k, nt * (nt + 1) // 2)
    _syrk_kernel[grid](
        X,
        S,
        R,
        n,
        nt,
        X.stride(0),
        X.stride(1),
        S.stride(0),
        S.stride(1),
        BM=BM,
        BK=BK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return S


def check_list():
    """syrk_list on separate buffers vs bmm on the stacked rows (bitwise expected) and its speed."""
    import time

    torch.manual_seed(0)
    for k, R, n in ((48, 9152, 512), (8, 9152, 2048), (48, 9152, 768), (8, 9152, 3072)):
        xs = [torch.randn(R, n, device="cuda").bfloat16() for _ in range(k)]
        ptrs = list_offsets(xs)
        X = torch.stack(xs)
        ref = torch.bmm(X.mT, X, out_dtype=torch.float32)
        got = syrk_list(xs, ptrs, BM=128, BK=64, num_warps=4, num_stages=3)
        same = torch.equal(got, syrk(X, 128, 64, 4, 3))
        err = float((got - ref).abs().max() / ref.abs().max())
        for _ in range(3):
            syrk_list(xs, ptrs, 128, 64, 4, 3)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            syrk_list(xs, ptrs, 128, 64, 4, 3)
        torch.cuda.synchronize()
        print(
            f"k={k} R={R} n={n}: syrk_list {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms, max rel err vs bmm {err:.1e}, "
            f"bitwise equal to stacked syrk: {same}",
            flush=True,
        )


def check_ema():
    """syrk_ema vs the unfused EMA (b C + c bmm) and its speed."""
    import time

    torch.manual_seed(0)
    for k, R, n in ((48, 9152, 512), (8, 9152, 2048), (48, 9152, 768), (8, 9152, 3072)):
        xs = [torch.randn(R, n, device="cuda").bfloat16() for _ in range(k)]
        offs = list_offsets(xs)
        X = torch.stack(xs)
        C0 = torch.bmm(X.mT, X, out_dtype=torch.float32) / R
        b, g = 0.938, 0.062 / R
        ref = C0 * b + torch.bmm(X.mT, X, out_dtype=torch.float32) * g
        C = C0.clone()
        syrk_ema(xs, offs, C, b, g)
        err = float((C - ref).abs().max() / ref.abs().max())
        asym = float((C - C.mT).abs().max() / C.abs().max())
        for _ in range(3):
            syrk_ema(xs, offs, C, b, g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            syrk_ema(xs, offs, C, b, g)
        torch.cuda.synchronize()
        print(
            f"k={k} R={R} n={n}: syrk_ema {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms, max rel err vs unfused EMA "
            f"{err:.1e}, max asymmetry {asym:.1e}",
            flush=True,
        )


if (
    __name__ == "__main__"
    and len(__import__("sys").argv) > 1
    and __import__("sys").argv[1] == "list"
):
    check_list()
elif (
    __name__ == "__main__"
    and len(__import__("sys").argv) > 1
    and __import__("sys").argv[1] == "ema"
):
    check_ema()
elif __name__ == "__main__":  # correctness and speed vs torch.bmm(out_dtype=float32)
    import time

    torch.manual_seed(0)
    for k, R, n in ((48, 9152, 768), (8, 9152, 3072), (48, 9152, 512), (8, 9152, 2048)):
        X = torch.randn(k, R, n, device="cuda").bfloat16()
        ref = torch.bmm(X.mT, X, out_dtype=torch.float32)
        best = None
        for BM, BK, w, st in (
            (128, 64, 8, 3),
            (128, 32, 8, 4),
            (64, 64, 4, 3),
            (128, 64, 4, 3),
            (64, 32, 4, 4),
        ):
            if n % BM:
                continue
            got = syrk(X, BM, BK, w, st)
            err = float((got - ref).abs().max() / ref.abs().max())
            for _ in range(3):
                syrk(X, BM, BK, w, st)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                syrk(X, BM, BK, w, st)
            torch.cuda.synchronize()
            ts = (time.perf_counter() - t0) / 10 * 1e3
            if best is None or ts < best[0]:
                best = (ts, BM, BK, w, st, err)
        for _ in range(3):
            torch.bmm(X.mT, X, out_dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            torch.bmm(X.mT, X, out_dtype=torch.float32)
        torch.cuda.synchronize()
        tb = (time.perf_counter() - t0) / 10 * 1e3
        print(
            f"k={k} R={R} n={n}: bmm(out fp32) {tb:.2f} ms; syrk best {best[0]:.2f} ms (BM {best[1]}, BK {best[2]}, warps {best[3]}, "
            f"stages {best[4]}; max rel err {best[5]:.1e})",
            flush=True,
        )
