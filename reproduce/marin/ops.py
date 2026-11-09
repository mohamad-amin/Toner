# Ops shared by train.py and its tests.
#   doc_attn: document-masked causal attention on a packed micro-batch (B*T tokens) with FlashAttention's varlen kernels
#             (FA2, or FA3 once torch.nn.attention.activate_flash_attention_impl("FA3") is called on an H100).
import torch
import triton
import triton.language as tl


# Capture sampled inputs through one opaque operation after the forward pass.
@triton.jit
def _gather_rows(tab, n, at, idx, P, D, s_b, s_t, BLOCK: tl.constexpr):
    i = tl.program_id(0)  # source i: its tensor's rows at the sampled positions into its buffer
    r = tl.program_id(
        1
    )  # row b * P + p of this micro-batch's rows; written to row at[r] of the buffer
    src = tl.load(tab + i).to(tl.pointer_type(tl.bfloat16))
    dst = tl.load(tab + n + i).to(tl.pointer_type(tl.bfloat16))
    pos = tl.load(idx + r % P)
    row = tl.load(at + r)
    offs = tl.arange(0, BLOCK)
    for c in range(0, D, BLOCK):
        m = c + offs < D
        tl.store(
            dst + row * D + c + offs,
            tl.load(src + (r // P) * s_b + pos * s_t + c + offs, mask=m),
            mask=m,
        )


@torch.library.custom_op("toner_marin::capture_rows", mutates_args=("bufs",))
def capture_rows(
    srcs: list[torch.Tensor], bufs: list[torch.Tensor], at: torch.Tensor, idx: torch.Tensor
) -> None:
    """bufs[i][at[b P + p]] = srcs[i][b, idx[p]] for bf16 inputs srcs[i] (B, T, D_i) and rows buffers (rows, D_i)."""
    P = idx.numel()
    for D in sorted({s.size(-1) for s in srcs}):
        sel = [i for i, s in enumerate(srcs) if s.size(-1) == D]
        s0 = srcs[sel[0]]
        assert (
            all(
                srcs[i].dtype == torch.bfloat16
                and srcs[i].stride() == s0.stride()
                and bufs[i].is_contiguous()
                for i in sel
            )
            and s0.stride(-1) == 1
        )
        tab = torch.tensor(
            [srcs[i].data_ptr() for i in sel] + [bufs[i].data_ptr() for i in sel], dtype=torch.int64
        )
        tab = tab.pin_memory().to(at.device, non_blocking=True)
        _gather_rows[(len(sel), s0.size(0) * P)](
            tab, len(sel), at, idx, P, D, s0.stride(0), s0.stride(1), BLOCK=512
        )


@capture_rows.register_fake
def _(srcs, bufs, at, idx):
    return None


# Recompute sampled RMSNorm and SwiGLU rows without materializing full intermediate inputs.
@triton.jit
def _norm_rows(tab, n, at, idx, P, D: tl.constexpr, s_b, s_t, eps, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    r = tl.program_id(1)
    src = tl.load(tab + i).to(tl.pointer_type(tl.bfloat16))
    w = tl.load(tab + n + i).to(tl.pointer_type(tl.float32))
    dst = tl.load(tab + 2 * n + i).to(tl.pointer_type(tl.bfloat16))
    pos = tl.load(idx + r % P)
    row = tl.load(at + r)
    offs = tl.arange(0, BLOCK)  # the whole row (BLOCK = D rounded up to a power of two)
    m = offs < D
    x = tl.load(src + (r // P) * s_b + pos * s_t + offs, mask=m, other=0.0).to(tl.float32)
    out = (x * tl.math.rsqrt(tl.sum(x * x, 0) / D + eps)).to(tl.bfloat16)
    h = tl.load(w + offs, mask=m, other=0.0).to(tl.bfloat16).to(tl.float32) * out.to(tl.float32)
    tl.store(dst + row * D + offs, h.to(tl.bfloat16), mask=m)


@triton.jit
def _silu_mul_rows(tab, n, at, idx, P, D, s_b, s_t, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    r = tl.program_id(1)
    g = tl.load(tab + i).to(tl.pointer_type(tl.bfloat16))
    u = tl.load(tab + n + i).to(tl.pointer_type(tl.bfloat16))
    dst = tl.load(tab + 2 * n + i).to(tl.pointer_type(tl.bfloat16))
    pos = tl.load(idx + r % P)
    row = tl.load(at + r)
    offs = tl.arange(0, BLOCK)
    base = (r // P) * s_b + pos * s_t
    for c in range(0, D, BLOCK):
        m = c + offs < D
        gv = tl.load(g + base + c + offs, mask=m).to(tl.float32)
        uv = tl.load(u + base + c + offs, mask=m).to(tl.float32)
        # Match the compiled SwiGLU input: FP32 arithmetic, then one BF16 rounding.
        tl.store(
            dst + row * D + c + offs,
            (gv * (1.0 / (1.0 + tl.exp(-gv))) * uv).to(tl.bfloat16),
            mask=m,
        )


_TABLES = {}  # capture_recomputed's device pointer tables by their content: the activations' addresses repeat across
# micro-batches (the caching allocator), so in steady state no table is built or copied, and the checks run once


@torch.library.custom_op("toner_marin::capture_recomputed", mutates_args=("bufs",))
def capture_recomputed(
    xs: list[torch.Tensor],
    attn_out: list[torch.Tensor],
    xmid: list[torch.Tensor],
    gs: list[torch.Tensor],
    us: list[torch.Tensor],
    w1: list[torch.Tensor],
    w2: list[torch.Tensor],
    bufs: list[torch.Tensor],
    at: torch.Tensor,
    idx: torch.Tensor,
    eps: float,
) -> None:
    """Per layer l (bufs in blocks of 4: attn, o, mlp, down): norm(ln1, x) | a | norm(ln2, x_mid) | silu(g) * u at the
    sampled rows into the rows buffers. All inputs bf16 (B, T, D), the norm weights f32."""
    P, L, dev = idx.numel(), len(xs), at.device
    x0, g0 = xs[0], gs[0]
    B, D = x0.size(0), x0.size(-1)
    b_attn, b_o, b_mlp, b_down = bufs[0::4], bufs[1::4], bufs[2::4], bufs[3::4]
    cols = (
        xs,
        w1,
        b_attn,
        xmid,
        w2,
        b_mlp,
        attn_out,
        b_o,
        gs,
        us,
        b_down,
    )  # the kernels' tables, in this order
    key = tuple(t.data_ptr() for col in cols for t in col)
    tab = _TABLES.get(key)
    if tab is None:
        assert all(
            t.dtype == torch.bfloat16 and t.stride() == x0.stride() for t in xs + attn_out + xmid
        )
        assert (
            all(t.dtype == torch.bfloat16 and t.stride() == g0.stride() for t in gs + us)
            and x0.stride(-1) == 1
        )
        if len(_TABLES) >= 64:  # addresses that keep changing: keep the cache bounded
            _TABLES.clear()
        tab = _TABLES[key] = (
            torch.tensor(key, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
        )
    NB = triton.next_power_of_2(D)
    _norm_rows[(L, B * P)](tab[0:], L, at, idx, P, D, x0.stride(0), x0.stride(1), eps, BLOCK=NB)
    _norm_rows[(L, B * P)](
        tab[3 * L :], L, at, idx, P, D, x0.stride(0), x0.stride(1), eps, BLOCK=NB
    )
    _gather_rows[(L, B * P)](tab[6 * L :], L, at, idx, P, D, x0.stride(0), x0.stride(1), BLOCK=512)
    _silu_mul_rows[(L, B * P)](
        tab[8 * L :], L, at, idx, P, g0.size(-1), g0.stride(0), g0.stride(1), BLOCK=512
    )


@capture_recomputed.register_fake
def _(xs, attn_out, xmid, gs, us, w1, w2, bufs, at, idx, eps):
    return None


# Only cu[:n_docs + 1] and the true max length reach the kernel, so its grid stays tight; both are read from the CPU
# tensor inside the op, so there is no GPU sync and nothing for torch.compile to specialize on.
ATTN_HOOK = None  # capture_codegen.Context.attn: sees each attention output here, inside the opaque op (no graph change)


@torch.library.custom_op("toner_marin::doc_attn", mutates_args=())
def doc_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu: torch.Tensor, meta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n, mx = meta.tolist()
    c = cu[: n + 1]
    out, lse, _, _, _ = torch.ops.aten._flash_attention_forward(
        q, k, v, c, c, mx, mx, 0.0, True, return_debug_mask=False
    )
    if ATTN_HOOK is not None:
        ATTN_HOOK(out)
    return out, lse


@doc_attn.register_fake
def _(q, k, v, cu, meta):
    return torch.empty_like(q), q.new_empty((q.size(1), q.size(0)), dtype=torch.float32)


@torch.library.custom_op("toner_marin::doc_attn_bwd", mutates_args=())
def doc_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu: torch.Tensor,
    meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, mx = meta.tolist()
    c = cu[: n + 1]
    rng = torch.zeros(2, dtype=torch.uint64, device=q.device)
    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        dout.contiguous(),
        q,
        k,
        v,
        out,
        lse,
        c,
        c,
        mx,
        mx,
        0.0,
        True,
        rng,
        torch.empty(0, device=q.device),
    )
    return dq, dk, dv


@doc_attn_bwd.register_fake
def _(dout, q, k, v, out, lse, cu, meta):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _attn_setup(ctx, inputs, output):
    q, k, v, cu, meta = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], cu, meta)


def _attn_backward(ctx, dout, dlse):
    q, k, v, out, lse, cu, meta = ctx.saved_tensors
    dq, dk, dv = doc_attn_bwd(dout, q, k, v, out, lse, cu, meta)
    return dq, dk, dv, None, None


doc_attn.register_autograd(_attn_backward, setup_context=_attn_setup)
