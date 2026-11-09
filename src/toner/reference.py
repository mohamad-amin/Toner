"""MuonT and AdamT for selected FP32 matrix weights. Supply input statistics once per step."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .linalg import input_factors, newton_schulz


def _finite_number(
    name: str, value: float, *, lower: float = 0.0, upper: float | None = None
) -> None:
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < lower
        or (upper is not None and value >= upper)
    ):
        suffix = f" and less than {upper}" if upper is not None else ""
        raise ValueError(f"{name} must be finite, at least {lower}{suffix}")


class _TonerReference(Optimizer):
    """Shared per-step statistic EMA."""

    def _validate_common(self) -> None:
        for group in self.param_groups:
            _finite_number("lr", group["lr"])
            _finite_number("weight_decay", group["weight_decay"])
            _finite_number("statistic_momentum", group["statistic_momentum"], upper=1)
            _finite_number("damping", group["damping"], lower=1e-30)
            for parameter in group["params"]:
                if parameter.ndim != 2 or parameter.numel() == 0:
                    raise ValueError(
                        "Toner expects nonempty 2-D matrix weights; optimize other parameters separately"
                    )
                if parameter.dtype != torch.float32:
                    raise ValueError(
                        "The reference matches the FP32 master weights; use FP32 parameters"
                    )
                if parameter.device.type not in {"cpu", "cuda"}:
                    raise ValueError("The reference supports CPU and CUDA parameters")

    @torch.no_grad()
    def update_statistics(self, second_moments: Mapping[Tensor, Tensor]) -> None:
        """Update the input EMA once from each parameter's current-step mean Gram.

        Each value is ``sum(x.T @ x) / total_rows`` across the step's sampled
        activations, including all accumulated microbatches and ranks. It is an
        uncentered second moment, not a centered covariance. This method does
        not all-reduce. Calling it twice without ``step()`` is an error.
        """
        groups = {parameter: group for group in self.param_groups for parameter in group["params"]}
        # Validate the complete input before changing any statistic state.
        for parameter, covariance in second_moments.items():
            if parameter not in groups:
                raise ValueError("Received a statistic for a parameter not owned by this optimizer")
            if covariance.shape != (parameter.shape[1], parameter.shape[1]):
                raise ValueError("Statistic shape must match the matrix weight's input dimension")
            if covariance.device != parameter.device:
                raise ValueError("Input statistics and matrix weights must be on the same device")
            if not covariance.is_floating_point() or not bool(torch.isfinite(covariance).all()):
                raise ValueError("Input statistics must be finite floating-point matrices")
            if self.state[parameter].get("statistics_ready", False):
                raise RuntimeError(
                    "Statistics already updated; aggregate microbatches before update_statistics()"
                )
        for parameter, covariance in second_moments.items():
            state, beta = self.state[parameter], groups[parameter]["statistic_momentum"]
            if "input_ema" not in state:
                state["input_ema"] = torch.zeros_like(covariance, dtype=torch.float32)
                state["input_weight"] = 0.0
            state["input_ema"].mul_(beta).add_(covariance.float(), alpha=1 - beta)
            state["input_weight"] = beta * state["input_weight"] + (1 - beta)
            state["statistics_ready"] = True

    def _check_step(self) -> None:
        self._validate_common()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Toner does not support sparse gradients")
                if not self.state[parameter].get("statistics_ready", False):
                    raise RuntimeError(
                        "Call update_statistics() with this step's input second moments before step()"
                    )

    def _consume_statistics(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["statistics_ready"] = False


class MuonT(_TonerReference):
    """Muon with input preconditioning and shaped weight decay.

    ``lr`` is the matrix learning rate. Defaults use half/eighth geometry.
    ``statistic_momentum=None`` inherits each parameter group’s momentum at
    construction; pass a number to override it. ``ns_mode='tpu'`` uses BF16 operands with FP32 accumulation; select
    ``'f32'`` explicitly for a CPU reference. Roots are recomputed by exact
    eigendecomposition every step, rather than the accelerated root engine.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        *,
        lr: float,
        weight_decay: float = 0.0,
        inner_strength: float = 0.5,
        outer_strength: float = 0.125,
        statistic_momentum: float | None = None,
        damping: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 10,
        ns_eps: float = 1e-5,
        ns_mode: str = "tpu",
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                inner_strength=inner_strength,
                outer_strength=outer_strength,
                statistic_momentum=statistic_momentum,
                damping=damping,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                ns_eps=ns_eps,
                ns_mode=ns_mode,
            ),
        )
        self._validate_common()
        self._validate_muon()

    def add_param_group(self, param_group) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["statistic_momentum"] is None:
            group["statistic_momentum"] = group["momentum"]

    def _validate_muon(self) -> None:
        for group in self.param_groups:
            for name in ("inner_strength", "outer_strength"):
                _finite_number(name, group[name])
            _finite_number("momentum", group["momentum"], upper=1)
            _finite_number("ns_eps", group["ns_eps"], lower=1e-30)
            if group["ns_mode"] not in {"tpu", "f32", "bf16"}:
                raise ValueError("ns_mode must be 'tpu', 'f32', or 'bf16'")
            if (
                not isinstance(group["ns_steps"], int)
                or isinstance(group["ns_steps"], bool)
                or group["ns_steps"] < 1
            ):
                raise ValueError("ns_steps must be a positive integer")

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._check_step()
        self._validate_muon()
        # Reject unsupported precision before advancing any optimizer moments.
        for group in self.param_groups:
            if group["ns_mode"] == "tpu" and any(
                p.grad is not None and p.device.type != "cuda" for p in group["params"]
            ):
                raise ValueError(
                    "ns_mode='tpu' requires CUDA; use ns_mode='f32' for the CPU reference"
                )
        for group in self.param_groups:
            lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    state = self.state[parameter]
                    inner, outer, decay = input_factors(
                        state["input_ema"] / state["input_weight"],
                        group["inner_strength"],
                        group["outer_strength"],
                        group["damping"],
                    )
                    gradient = parameter.grad.float()
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    buffer = state["momentum_buffer"]
                    buffer.mul_(mu).add_(gradient)
                    direction = gradient.add(buffer, alpha=mu) if group["nesterov"] else buffer
                    polar = newton_schulz(
                        direction @ inner, group["ns_steps"], group["ns_eps"], group["ns_mode"]
                    )
                    outer_normalized = outer * (
                        parameter.shape[1] ** 0.5 / outer.norm().clamp_min(1e-30)
                    )
                    update = (polar @ outer_normalized) * max(
                        1.0, parameter.shape[0] / parameter.shape[1]
                    ) ** 0.5
                    # This is the original weight's shaped decay, not decay of
                    # the already-adapted parameter and not a gradient penalty.
                    parameter.sub_(parameter @ decay, alpha=lr * wd)
                    parameter.add_(update, alpha=-lr)
        self._consume_statistics()
        return loss


class AdamT(_TonerReference):
    """Matched-strength AdamT, with an unnormalized second-moment factor.

    The first moment averages the original gradient. The second averages
    ``(G @ R)**2`` using the current *unnormalized* root. Mapping the adaptive
    update through the same root preserves its pre-mapping Frobenius norm.
    ``lr`` is the matrix learning rate. When omitted or ``None``,
    ``statistic_momentum`` inherits each parameter group’s ``betas[1]`` at
    construction or when the group is added. An explicit value overrides it.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        *,
        lr: float,
        weight_decay: float = 0.0,
        strength: float = 0.25,
        statistic_momentum: float | None = None,
        damping: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-20,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                strength=strength,
                statistic_momentum=statistic_momentum,
                damping=damping,
                betas=betas,
                eps=eps,
            ),
        )
        self._validate_common()
        self._validate_adam()

    def add_param_group(self, param_group) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["statistic_momentum"] is None:
            if len(group["betas"]) != 2:
                raise ValueError("betas must contain two coefficients")
            group["statistic_momentum"] = group["betas"][1]

    def _validate_adam(self) -> None:
        for group in self.param_groups:
            _finite_number("strength", group["strength"])
            _finite_number("eps", group["eps"], lower=1e-30)
            if len(group["betas"]) != 2:
                raise ValueError("betas must contain two coefficients")
            for beta in group["betas"]:
                _finite_number("beta", beta, upper=1)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._check_step()
        self._validate_adam()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    state = self.state[parameter]
                    root, _, decay = input_factors(
                        state["input_ema"] / state["input_weight"],
                        group["strength"],
                        group["strength"],
                        group["damping"],
                    )
                    gradient = parameter.grad.float()
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(gradient)
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                    state["step"] += 1
                    b1, b2 = group["betas"]
                    t = state["step"]
                    state["exp_avg"].mul_(b1).add_(gradient, alpha=1 - b1)
                    transformed = gradient @ root
                    state["exp_avg_sq"].mul_(b2).addcmul_(transformed, transformed, value=1 - b2)
                    numerator = (state["exp_avg"] / (1 - b1**t)) @ root
                    update = numerator / ((state["exp_avg_sq"] / (1 - b2**t)).sqrt() + group["eps"])
                    norm_before = update.norm()
                    update = update @ root
                    update = update * (norm_before / update.norm().clamp_min(1e-30))
                    parameter.sub_(parameter @ decay, alpha=lr * wd)
                    parameter.add_(update, alpha=-lr)
        self._consume_statistics()
        return loss
