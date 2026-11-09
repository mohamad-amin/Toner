"""Explicit forward capture for the readable Toner optimizers."""

from contextlib import contextmanager
from collections.abc import Mapping

import torch
from torch import nn
import torch.distributed as dist


class ActivationStatistics:
    """Accumulate uncentered input second moments across forward microbatches.

    ``layers`` maps names to the Linear modules optimized by Toner. Capture only
    the intended forward pass with ``capture()``; backward recomputation and
    evaluation are excluded unless explicitly captured. Call ``commit`` once
    after all microbatches and before the optimizer step.

    For (..., sequence, features) inputs, sample the first ``head`` positions,
    then positions head+1, head+1+stride, ... in each sequence.
    Two-dimensional inputs are treated as independent rows and all rows are used.
    DDP defaults to globally pooled statistics when distributed is initialized.
    This collector computes statistics synchronously.
    """

    def __init__(
        self,
        layers: Mapping[str, nn.Linear],
        *,
        head: int = 32,
        stride: int = 16,
        distributed: bool = True,
        process_group=None,
    ):
        if not layers:
            raise ValueError("Select at least one nn.Linear layer.")
        if head < 0 or stride < 1:
            raise ValueError("head must be nonnegative and stride positive.")
        self.layers = dict(layers)
        self.head, self.stride = head, stride
        self.distributed, self.process_group = distributed, process_group
        self._active = False
        self._closed = False
        self._sums, self._counts = {}, {}
        seen = set()
        for name, layer in self.layers.items():
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"{name}: expected nn.Linear.")
            if id(layer.weight) in seen:
                raise ValueError(
                    "Shared/tied weights require an explicit statistic; do not register them twice."
                )
            seen.add(id(layer.weight))
        self._handles = [
            layer.register_forward_pre_hook(self._hook(name)) for name, layer in self.layers.items()
        ]

    def _hook(self, name):
        @torch.no_grad()
        def collect(module, inputs):
            if not self._active:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError(f"{name}: capture expects a positional tensor input.")
            x = inputs[0].detach()
            if x.ndim < 2 or x.shape[-1] != module.in_features:
                raise ValueError(f"{name}: expected rows of {module.in_features} input features.")
            if x.ndim > 2:
                seq = x.shape[-2]
                positions = list(range(min(self.head, seq)))
                positions += list(range(self.head + 1 if self.head else 1, seq, self.stride))
                if not positions:
                    raise ValueError(f"{name}: sampling selects no input positions.")
                idx = torch.tensor(positions, device=x.device, dtype=torch.long)
                x = x.index_select(-2, idx)
            # Linear's autocast happens after its pre-hook. Match the inputs
            # actually used by its GEMM, not an unrounded FP32 norm output.
            if torch.is_autocast_enabled(x.device.type):
                x = x.to(torch.get_autocast_dtype(x.device.type))
            x = x.reshape(-1, x.shape[-1])
            if not x.shape[0]:
                raise ValueError(f"{name}: cannot estimate a statistic from zero rows.")
            # Autocast must not silently lower the accumulation precision.
            with torch.autocast(device_type=x.device.type, enabled=False):
                if x.is_cuda and x.dtype == torch.bfloat16:
                    gram = torch.mm(x.T, x, out_dtype=torch.float32)
                else:
                    xf = x.float()
                    gram = xf.T @ xf
            if name not in self._sums:
                self._sums[name] = gram
                self._counts[name] = x.shape[0]
            else:
                self._sums[name].add_(gram)
                self._counts[name] += x.shape[0]

        return collect

    @contextmanager
    def capture(self):
        """Capture one forward; errors discard the whole pending step."""
        if self._closed:
            raise RuntimeError("The statistic collector is closed.")
        if self._active:
            raise RuntimeError("Nested capture contexts would double-count activations.")
        self._active = True
        try:
            yield self
        except BaseException:
            # A failed forward may have captured only part of the model. Discard
            # the entire pending step rather than contaminate a retry.
            self._active = False
            self.clear()
            raise
        finally:
            self._active = False

    @torch.no_grad()
    def commit(self, optimizer):
        """Pool rows, update optimizer statistic EMAs once, and clear captures.

        With DDP, every rank must call this with the same registered layers.
        The caller must synchronize gradients separately (e.g. through DDP).
        """
        if self._active:
            raise RuntimeError("Exit capture() before committing statistics.")
        if self._closed:
            raise RuntimeError("The statistic collector is closed.")
        missing = [name for name in self.layers if name not in self._sums]
        if missing:
            raise RuntimeError(f"No captured inputs for: {', '.join(missing)}")
        means = {}
        for name, layer in self.layers.items():
            gram = self._sums[name].clone()
            count = torch.tensor(self._counts[name], device=gram.device, dtype=torch.int64)
            if self.distributed and dist.is_available() and dist.is_initialized():
                dist.all_reduce(gram, group=self.process_group)
                dist.all_reduce(count, group=self.process_group)
            means[layer.weight] = gram / count.to(dtype=gram.dtype)
        optimizer.update_statistics(means)
        self.clear()

    def clear(self):
        """Discard captured rows, for example after skipping a training step."""
        if self._active:
            raise RuntimeError("Cannot clear statistics during capture.")
        self._sums.clear()
        self._counts.clear()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()
        self._closed = True
