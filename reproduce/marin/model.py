"""Marin Llama-style transformer with document-packed attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def norm(layer, x, residual_dtype="bf16"):
    if residual_dtype == "f32":
        return layer(x)
    xf = x.float()
    out = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + layer.eps)).to(torch.bfloat16)
    return layer.weight.to(torch.bfloat16) * out


def sample_positions(sequence_length, head, stride):
    return list(range(head)) + list(range(head + 1, sequence_length, stride))


class Block(nn.Module):
    def __init__(self, config, *, positions=None, row_capacity=0):
        super().__init__()
        self.config = config
        h, m = config["hidden"], config["mlp"]
        self.ln1, self.ln2 = (nn.RMSNorm(h, eps=config["norm_eps"]) for _ in range(2))
        self.q, self.k, self.v, self.o = (nn.Linear(h, h, bias=False) for _ in range(4))
        self.gate, self.up = (nn.Linear(h, m, bias=False) for _ in range(2))
        self.down = nn.Linear(m, h, bias=False)
        if positions is not None:
            for kind, dim in (("attn", h), ("o", h), ("mlp", h), ("down", m)):
                self.register_buffer(
                    f"input_rows_{kind}",
                    torch.zeros(row_capacity, dim, dtype=torch.bfloat16),
                    persistent=False,
                )
            self.register_buffer("stat_idx", torch.tensor(positions), persistent=False)

    def forward(self, x, cos, sin, cu, meta, captures=None):
        from . import ops

        c = self.config
        B, T, H = x.shape
        heads, hd = c["heads"], H // c["heads"]
        x_in = x
        h = norm(self.ln1, x, c["residual_dtype"])
        q, k, v = (f(h).view(B, T, heads, hd) for f in (self.q, self.k, self.v))

        def rotate(t):
            return t * cos + torch.cat((-t[..., hd // 2 :], t[..., : hd // 2]), -1) * sin

        q, k, v = rotate(q).to(torch.bfloat16), rotate(k).to(torch.bfloat16), v.to(torch.bfloat16)
        a, _ = ops.doc_attn(
            q.reshape(B * T, heads, hd),
            k.reshape(B * T, heads, hd),
            v.reshape(B * T, heads, hd),
            cu,
            meta,
        )
        a = a.view(B, T, H)
        x = x + self.o(a)
        h2 = norm(self.ln2, x, c["residual_dtype"])
        g, u = self.gate(h2), self.up(h2)
        m = F.silu(g) * u
        if captures is not None:
            captures.append(tuple(t.detach() for t in (x_in, a, x, g, u)))
        return x + self.down(m)


class Body(nn.Module):
    def __init__(self, config, *, positions=None, row_capacity=0, rows_per_microbatch=0):
        super().__init__()
        self.config = config
        h, hd = config["hidden"], config["hidden"] // config["heads"]
        self.emb = nn.Embedding(config["vocab_size"], h)
        self.blocks = nn.ModuleList(
            Block(config, positions=positions, row_capacity=row_capacity)
            for _ in range(config["layers"])
        )
        self.norm = nn.RMSNorm(h, eps=config["norm_eps"])
        inv = 1.0 / config["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
        ang = torch.outer(torch.arange(config["sequence_length"], dtype=torch.float32), inv)
        self.register_buffer("cos", torch.cat((ang, ang), -1).cos()[:, None], persistent=False)
        self.register_buffer("sin", torch.cat((ang, ang), -1).sin()[:, None], persistent=False)
        self.capture = positions is not None
        if self.capture:
            self.register_buffer("stat_at", torch.arange(rows_per_microbatch), persistent=False)

    def forward(self, tokens, cu, meta):
        x = self.emb(tokens)
        if self.config["residual_dtype"] == "bf16":
            x = x.to(torch.bfloat16)
        captures = [] if self.capture else None
        for block in self.blocks:
            x = block(x, self.cos, self.sin, cu, meta, captures)
        if self.capture:
            from . import ops

            xi, attn, xm, g, u = (list(c) for c in zip(*captures))
            bufs = [
                getattr(b, f"input_rows_{k}")
                for b in self.blocks
                for k in ("attn", "o", "mlp", "down")
            ]
            ops.capture_recomputed(
                xi,
                attn,
                xm,
                g,
                u,
                [b.ln1.weight.detach() for b in self.blocks],
                [b.ln2.weight.detach() for b in self.blocks],
                bufs,
                self.stat_at,
                self.blocks[0].stat_idx,
                self.config["norm_eps"],
            )
        return norm(self.norm, x, self.config["residual_dtype"])


class Model(nn.Module):
    def __init__(self, config, **capture):
        super().__init__()
        self.body = Body(config, **capture)
        self.head = nn.Linear(config["hidden"], config["vocab_size"], bias=False)


def initialize_model(model, seed, embedding_scale="sqrt_hidden"):
    """Truncated-normal initialization; the head is equal to the standard-scale embedding draw."""
    if embedding_scale not in ("sqrt_hidden", "hidden"):
        raise ValueError(f"Unknown embedding scale {embedding_scale}")
    gen = torch.Generator(device=model.head.weight.device).manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear) and module is not model.head:
                nn.init.trunc_normal_(module.weight, 0.0, 1.0, -3.0, 3.0, generator=gen).div_(
                    math.sqrt(module.in_features)
                )
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        emb = model.body.emb.weight
        nn.init.trunc_normal_(emb, 0.0, 1.0, -3.0, 3.0, generator=gen)
        h = emb.shape[1]
        model.head.weight.copy_(emb).div_(math.sqrt(h))
        emb.div_(h if embedding_scale == "hidden" else math.sqrt(h))
