"""BF16 language-model head and token cross entropy."""

import torch
import triton
import triton.language as tl

IGNORE = -100


@triton.jit
def _fwd_kernel(
    h_ptr,
    w_ptr,
    z_ptr,
    pmax_ptr,
    psum_ptr,
    M,
    N,
    K,
    shm,
    shk,
    swn,
    swk,
    szm,
    szn,
    NT,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (GROUP * npn)
    first = gid * GROUP
    gs = min(npm - first, GROUP)
    pm = first + (pid % gs)
    pn = (pid % (GROUP * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(
            h_ptr + rm[:, None] * shm + (k0 + rk)[None, :] * shk, mask=rm[:, None] < M, other=0.0
        )
        b = tl.load(
            w_ptr + rn[None, :] * swn + (k0 + rk)[:, None] * swk, mask=rn[None, :] < N, other=0.0
        )
        acc = tl.dot(a, b, acc)
    zb = acc.to(tl.bfloat16)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    # 64-bit row base: M x N exceeds 2^31 from M = 16,744 rows at the Llama-3 vocabulary (int32 offsets wrap around)
    z_rows = z_ptr + (pm * BM).to(tl.int64) * szm
    tl.store(z_rows + tl.arange(0, BM)[:, None] * szm + rn[None, :] * szn, zb, mask=msk)
    zf = tl.where(rn[None, :] < N, zb.to(tl.float32), float("-inf"))
    mx = tl.max(zf, 1)
    s = tl.sum(tl.exp(zf - mx[:, None]), 1)
    tl.store(pmax_ptr + rm * NT + pn, mx, mask=rm < M)
    tl.store(psum_ptr + rm * NT + pn, s, mask=rm < M)


@triton.jit
def _lse_kernel(pmax_ptr, psum_ptr, z_ptr, t_ptr, lse_ptr, loss_ptr, NT, szm, BT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    m = tl.full([BT], float("-inf"), tl.float32)
    s = tl.zeros([BT], tl.float32)
    for j0 in range(0, NT, BT):
        j = j0 + tl.arange(0, BT)
        pm = tl.load(pmax_ptr + row * NT + j, mask=j < NT, other=float("-inf"))
        ps = tl.load(psum_ptr + row * NT + j, mask=j < NT, other=0.0)
        mn = tl.maximum(m, pm)
        s = s * tl.exp(tl.where(mn == float("-inf"), 0.0, m - mn)) + ps * tl.exp(
            tl.where(mn == float("-inf"), 0.0, pm - mn)
        )
        m = mn
    mx = tl.max(m, 0)
    lse = mx + tl.log(tl.sum(s * tl.exp(m - mx), 0))
    t = tl.load(t_ptr + row)
    valid = t != -100
    zt = tl.load(z_ptr + row * szm + tl.where(valid, t, 0)).to(tl.float32)
    tl.store(lse_ptr + row, lse)
    tl.store(loss_ptr + row, tl.where(valid, lse - zt, 0.0))


@triton.jit
def _dz_tile(z_ptr, lse_ptr, t_ptr, g_ptr, r0, rn, M, N, szm, szn, BR: tl.constexpr):
    # dz for rows r0 ... r0 + BR - 1 and columns rn, rounded to bf16 (reference rounding point); 0 outside the matrix.
    # The row base is 64-bit (M x N can exceed 2^31); offsets within the tile stay 32-bit.
    rl = tl.arange(0, BR)
    rm = r0 + rl
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    z = tl.load(
        z_ptr + r0.to(tl.int64) * szm + rl[:, None] * szm + rn[None, :] * szn, mask=msk, other=0.0
    ).to(tl.float32)
    lse = tl.load(lse_ptr + rm, mask=rm < M, other=0.0)
    t = tl.load(t_ptr + rm, mask=rm < M, other=-100)
    g = tl.load(g_ptr + rm, mask=rm < M, other=0.0)
    g = tl.where(t != -100, g, 0.0)
    d = (tl.exp(z - lse[:, None]) - tl.where(rn[None, :] == t[:, None], 1.0, 0.0)) * g[:, None]
    return tl.where(msk, d, 0.0).to(tl.bfloat16)


@triton.jit
def _dh_kernel(
    z_ptr,
    lse_ptr,
    t_ptr,
    g_ptr,
    w_ptr,
    dh_ptr,
    M,
    N,
    K,
    szm,
    szn,
    swn,
    swk,
    sdm,
    sdk,
    BM: tl.constexpr,
    BK_OUT: tl.constexpr,
    BN_RED: tl.constexpr,
    GROUP: tl.constexpr,
):
    # dh[M, K] = dz[M, N] @ W[N, K]; the reduction runs over the vocabulary N
    pid = tl.program_id(0)
    npm, npk = tl.cdiv(M, BM), tl.cdiv(K, BK_OUT)
    gid = pid // (GROUP * npk)
    first = gid * GROUP
    gs = min(npm - first, GROUP)
    pm = first + (pid % gs)
    pk = (pid % (GROUP * npk)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rko = pk * BK_OUT + tl.arange(0, BK_OUT)
    acc = tl.zeros((BM, BK_OUT), tl.float32)
    for n0 in range(0, N, BN_RED):
        rn = n0 + tl.arange(0, BN_RED)
        dz = _dz_tile(z_ptr, lse_ptr, t_ptr, g_ptr, pm * BM, rn, M, N, szm, szn, BM)
        w = tl.load(w_ptr + rn[:, None] * swn + rko[None, :] * swk, mask=rn[:, None] < N, other=0.0)
        acc = tl.dot(dz, w, acc)
    tl.store(
        dh_ptr + rm[:, None] * sdm + rko[None, :] * sdk, acc.to(tl.bfloat16), mask=rm[:, None] < M
    )


@triton.jit
def _dw_kernel(
    z_ptr,
    lse_ptr,
    t_ptr,
    g_ptr,
    h_ptr,
    dw_ptr,
    M,
    N,
    K,
    szm,
    szn,
    shm,
    shk,
    sdn,
    sdk,
    BN: tl.constexpr,
    BK_OUT: tl.constexpr,
    BM_RED: tl.constexpr,
    GROUP: tl.constexpr,
):
    # dW[N, K] = dz[M, N]^T @ h[M, K]; the reduction runs over tokens M
    pid = tl.program_id(0)
    npn, npk = tl.cdiv(N, BN), tl.cdiv(K, BK_OUT)
    gid = pid // (GROUP * npk)
    first = gid * GROUP
    gs = min(npn - first, GROUP)
    pn = first + (pid % gs)
    pk = (pid % (GROUP * npk)) // gs
    rn = pn * BN + tl.arange(0, BN)
    rko = pk * BK_OUT + tl.arange(0, BK_OUT)
    acc = tl.zeros((BN, BK_OUT), tl.float32)
    for m0 in range(0, M, BM_RED):
        rm = m0 + tl.arange(0, BM_RED)
        dz = _dz_tile(z_ptr, lse_ptr, t_ptr, g_ptr, m0, rn, M, N, szm, szn, BM_RED)  # [BM_RED, BN]
        h = tl.load(h_ptr + rm[:, None] * shm + rko[None, :] * shk, mask=rm[:, None] < M, other=0.0)
        acc = tl.dot(tl.trans(dz), h, acc)
    # autocast reference: the bf16 weight copy gets a bf16 gradient, then the cast back to the f32 master
    tl.store(
        dw_ptr + rn[:, None] * sdn + rko[None, :] * sdk,
        acc.to(tl.bfloat16).to(tl.float32),
        mask=rn[:, None] < N,
    )


CFG = {"fwd": (256, 128, 64, 8, 2), "dh": (128, 256, 64, 8, 2), "dw": (128, 256, 64, 8, 2)}


def head_ce_forward(h, w_bf16, t):
    M, K = h.shape
    N = w_bf16.shape[0]
    BM, BN, BK, nw, ns = CFG["fwd"]
    NT = triton.cdiv(N, BN)
    z = torch.empty(M, N, device=h.device, dtype=torch.bfloat16)
    pmax = torch.empty(M, NT, device=h.device, dtype=torch.float32)
    psum = torch.empty(M, NT, device=h.device, dtype=torch.float32)
    _fwd_kernel[(triton.cdiv(M, BM) * NT,)](
        h,
        w_bf16,
        z,
        pmax,
        psum,
        M,
        N,
        K,
        h.stride(0),
        h.stride(1),
        w_bf16.stride(0),
        w_bf16.stride(1),
        z.stride(0),
        z.stride(1),
        NT,
        BM=BM,
        BN=BN,
        BK=BK,
        GROUP=8,
        num_warps=nw,
        num_stages=ns,
    )
    lse = torch.empty(M, device=h.device, dtype=torch.float32)
    loss = torch.empty(M, device=h.device, dtype=torch.float32)
    _lse_kernel[(M,)](pmax, psum, z, t, lse, loss, NT, z.stride(0), BT=512, num_warps=4)
    return z, lse, loss


def head_ce_backward(z, lse, t, grow, h, w_bf16):
    M, N = z.shape
    K = h.shape[1]
    BM, BKo, BNr, nw, ns = CFG["dh"]
    dh = torch.empty(M, K, device=h.device, dtype=torch.bfloat16)
    _dh_kernel[(triton.cdiv(M, BM) * triton.cdiv(K, BKo),)](
        z,
        lse,
        t,
        grow,
        w_bf16,
        dh,
        M,
        N,
        K,
        z.stride(0),
        z.stride(1),
        w_bf16.stride(0),
        w_bf16.stride(1),
        dh.stride(0),
        dh.stride(1),
        BM=BM,
        BK_OUT=BKo,
        BN_RED=BNr,
        GROUP=8,
        num_warps=nw,
        num_stages=ns,
    )
    BN, BKo2, BMr, nw2, ns2 = CFG["dw"]
    dw = torch.empty(N, K, device=h.device, dtype=torch.float32)
    _dw_kernel[(triton.cdiv(N, BN) * triton.cdiv(K, BKo2),)](
        z,
        lse,
        t,
        grow,
        h,
        dw,
        M,
        N,
        K,
        z.stride(0),
        z.stride(1),
        h.stride(0),
        h.stride(1),
        dw.stride(0),
        dw.stride(1),
        BN=BN,
        BK_OUT=BKo2,
        BM_RED=BMr,
        GROUP=8,
        num_warps=nw2,
        num_stages=ns2,
    )
    return dh, dw


class HeadCE(torch.autograd.Function):
    """loss = mean over valid rows of CE(bf16(h @ W^T), t); h bf16 [M,K], W f32 master [N,K] (cast to bf16 inside)."""

    @staticmethod
    def forward(ctx, h, W, t):
        if h.shape[1] % 256:
            raise ValueError("The LM-head backward kernels require hidden width divisible by 256")
        wb = W.to(torch.bfloat16)
        z, lse, loss_rows = head_ce_forward(h, wb, t)
        nvalid = (t != IGNORE).sum()
        ctx.save_for_backward(z, lse, t, h, wb, nvalid)
        return loss_rows.sum() / nvalid

    @staticmethod
    def backward(ctx, gout):
        z, lse, t, h, wb, nvalid = ctx.saved_tensors
        grow = (gout / nvalid).float().expand(z.shape[0]).contiguous()
        dh, dw = head_ce_backward(z, lse, t, grow, h, wb)
        return dh, dw, None
