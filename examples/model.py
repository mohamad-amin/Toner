# Adapted from modded-nanoGPT: https://github.com/KellerJordan/modded-nanogpt
# MIT License
#
# Copyright (c) 2024 Keller Jordan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""A compact modded-nanoGPT model with separate attention projections."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_embd: int = 768
    head_dim: int = 128


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.to(x.dtype))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


class Rotary(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        frequency = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([frequency, torch.zeros_like(frequency)]))

    def forward(self, x):
        position = torch.arange(x.size(1), device=x.device, dtype=torch.float32)
        angle = torch.outer(position, self.angular_freq)[None, :, None, :]
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat(
            [left * angle.cos() + right * angle.sin(), -left * angle.sin() + right * angle.cos()],
            dim=-1,
        ).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = width // head_dim, head_dim
        self.q, self.k, self.v = [Linear(width, width) for _ in range(3)]
        self.proj = Linear(width, width)
        self.rotary = Rotary(head_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = [
            layer(x).view(batch, length, self.num_heads, self.head_dim)
            for layer in (self.q, self.k, self.v)
        ]
        q = self.rotary(F.rms_norm(q, (self.head_dim,)))
        k = self.rotary(F.rms_norm(k, (self.head_dim,)))
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.12, is_causal=True
        )
        return self.proj(output.transpose(1, 2).contiguous().view(batch, length, width))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc, self.proj = Linear(width, 4 * width), Linear(4 * width, width)

    def forward(self, x):
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn, self.mlp = CausalSelfAttention(width, head_dim), MLP(width)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if min(config.vocab_size, config.n_layer, config.n_embd, config.head_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        if config.n_embd % config.head_dim or config.head_dim % 4:
            raise ValueError("Width must divide into heads; head_dim must be divisible by four")
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config.n_embd, config.head_dim) for _ in range(config.n_layer)]
        )
        self.proj = Linear(config.n_embd, config.vocab_size)
        self.norm1, self.norm2 = RMSNorm(config.n_embd), RMSNorm(config.n_embd)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("weight"):
                    if "proj" in name:
                        parameter.zero_()
                    elif "embed" in name:
                        parameter.normal_()
                    else:
                        parameter.normal_(std=(0.33 / parameter.size(-1)) ** 0.5)
                elif name.endswith("bias"):
                    parameter.zero_()
                else:
                    parameter.fill_(1)

    def forward(self, tokens, targets=None):
        x = self.embed(tokens)
        if x.is_cuda:
            x = x.bfloat16()
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
