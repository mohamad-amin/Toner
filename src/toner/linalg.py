# Muon numerical routines adapted from Levanter (Apache-2.0).
"""Input factors and finite Newton–Schulz operations used by Toner."""

from __future__ import annotations

import torch
from torch import Tensor

_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _mm(left: Tensor, right: Tensor, mode: str) -> Tensor:
    if mode == "tpu":
        if left.device.type != "cuda":
            raise ValueError("ns_mode='tpu' requires CUDA; use ns_mode='f32' for the CPU reference")

        # and return FP32. Casting a BF16 output to FP32 is NOT equivalent.
        return torch.mm(left.to(torch.bfloat16), right.to(torch.bfloat16), out_dtype=torch.float32)
    return left @ right


def newton_schulz(matrix: Tensor, steps: int = 10, eps: float = 1e-5, mode: str = "tpu") -> Tensor:
    """finite polar approximation for a single 2-D matrix.

    ``tpu`` uses BF16 operands with FP32 accumulation/output on CUDA, as in
    CUDA. ``f32`` is the full-FP32 reference. ``bf16`` keeps
    the iteration in BF16 and is a different numerical configuration.
    """
    if matrix.ndim != 2:
        raise ValueError("newton_schulz expects a 2-D matrix")
    if mode not in {"tpu", "f32", "bf16"}:
        raise ValueError("mode must be 'tpu', 'f32', or 'bf16'")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if mode == "tpu" and matrix.device.type != "cuda":
        raise ValueError("ns_mode='tpu' requires CUDA; use ns_mode='f32' for the CPU reference")
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        a, b, c = _NS_COEFFICIENTS
        x = matrix.float()
        x = x / (torch.linalg.matrix_norm(x) + eps)
        if mode == "bf16":
            x = x.to(torch.bfloat16)
        transpose = x.size(0) > x.size(1)
        if transpose:
            x = x.T
        for _ in range(steps):
            gram = _mm(x, x.T, mode)
            x = a * x + _mm(b * gram + c * _mm(gram, gram, mode), x, mode)
        return (x.T if transpose else x).float()


def input_factors(
    covariance: Tensor,
    inner_strength: float,
    outer_strength: float,
    damping: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return unnormalized inner/outer factors and normalized outer-square decay.

    Symmetrization, eigenvalue clipping, trace-relative damping and the FP64
    eigensolver fallback follow the reference ``TonerStats._root_batch``.
    The inner factor remains unnormalized, including for AdamT's moments.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("input covariance must be square")
    if inner_strength < 0 or outer_strength < 0 or damping <= 0:
        raise ValueError("strengths must be nonnegative and damping positive")
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        c = covariance.float()
        c = 0.5 * (c + c.T)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(c)
        except RuntimeError:
            eigenvalues, eigenvectors = torch.linalg.eigh(c.double())
            eigenvalues, eigenvectors = eigenvalues.float(), eigenvectors.float()
        eigenvalues = eigenvalues.clamp_min(0)
        unit = eigenvalues / eigenvalues.mean().clamp_min(1e-30)
        inner_values = (unit + damping).pow(-inner_strength)
        inner = (eigenvectors * inner_values.unsqueeze(-2)) @ eigenvectors.T
        if inner_strength == outer_strength:
            outer_values, outer = inner_values, inner
        else:
            outer_values = (unit + damping).pow(-outer_strength)
            outer = (eigenvectors * outer_values.unsqueeze(-2)) @ eigenvectors.T
        decay = (eigenvectors * outer_values.square().unsqueeze(-2)) @ eigenvectors.T
        decay = decay / decay.diagonal().mean()
        return inner, outer, decay
