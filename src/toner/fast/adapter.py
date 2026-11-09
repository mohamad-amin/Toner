"""CUDA Toner optimizers for uniform Llama-style and modded-nanoGPT blocks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import math
import weakref

import torch
from torch import nn
import torch.distributed as dist


@dataclass(frozen=True)
class ProjectionBlock:
    """Seven projections of one uniform, bias-free Llama-style block.

    q/k/v must receive the same input tensor, as must gate/up. All layers must
    share hidden and MLP widths. Biases and non-block parameters are deliberately
    excluded; update those with the caller's ordinary AdamW optimizer.
    """

    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    o: nn.Linear
    gate: nn.Linear
    up: nn.Linear
    down: nn.Linear

    @classmethod
    def from_module(cls, block: nn.Module) -> "ProjectionBlock":
        """Adapt a block exposing q, k, v, o, gate, up and down attributes."""
        return cls(**{name: getattr(block, name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class NanoGPTBlock:
    """Six projections of a uniform modded-nanoGPT block.

    Q/K/V share one input. Attention projection, MLP expansion and MLP
    projection each have their own input statistic. Biases are allowed but
    excluded from Toner; give them to the caller's auxiliary optimizer.
    """

    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    attn_proj: nn.Linear
    fc: nn.Linear
    mlp_proj: nn.Linear

    @classmethod
    def from_module(cls, block: nn.Module) -> "NanoGPTBlock":
        """Map block.attn.{q,k,v,proj} and block.mlp.{fc,proj}."""
        return cls(
            block.attn.q, block.attn.k, block.attn.v, block.attn.proj, block.mlp.fc, block.mlp.proj
        )


# The preserved root kernels use process-global strengths and compiled graphs.
# Silently mixing configurations could compute the wrong optimizer.
_ACTIVE = weakref.WeakSet()
_LLAMA_NAMES = ("q", "k", "v", "o", "gate", "up", "down")
_NANOGPT_NAMES = ("q", "k", "v", "attn_proj", "fc", "mlp_proj")
_KINDS = {
    "q": "attn",
    "k": "attn",
    "v": "attn",
    "o": "o",
    "gate": "mlp",
    "up": "mlp",
    "down": "down",
    "attn_proj": "o",
    "fc": "mlp",
    "mlp_proj": "down",
}
_NANOGPT_PATHS = {
    "q": "attn.q",
    "k": "attn.k",
    "v": "attn.v",
    "attn_proj": "attn.proj",
    "fc": "mlp.fc",
    "mlp_proj": "mlp.proj",
}


def _projection_block(block):
    if isinstance(block, (ProjectionBlock, NanoGPTBlock)):
        return block
    if hasattr(block, "attn") and hasattr(block, "mlp"):
        return NanoGPTBlock.from_module(block)
    return ProjectionBlock.from_module(block)


class _FastToner(torch.optim.Optimizer):
    """Shared adapter; use FastMuonT or FastAdamT.

    lr is the matrix learning rate, used directly.
    The Muon shape factor is applied separately by the core.

    State serialization is collective: every DDP rank must call state_dict(),
    but only rank zero receives a checkpoint. Loading requires that every rank
    receive the same checkpoint, into a fresh optimizer before any captured step.
    """

    def __init__(
        self,
        blocks,
        lr,
        *,
        kind,
        micro_batch_size,
        sequence_length,
        accumulation_steps=1,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        betas=(0.8, 0.98),
        eps=1e-15,
        inner_strength=0.5,
        outer_strength=0.125,
        statistic_momentum=0.938,
        statistic_head=32,
        statistic_stride=16,
        ns_steps=10,
        ns_eps=1e-5,
        root_refresh=4,
        root_iterations=2,
        root_inline_steps=20,
        root_repair_rounds=3,
        root_graphs=True,
        host_workers=2,
        host_threads=4,
    ):
        if _ACTIVE:
            raise RuntimeError(
                "Only one fast Toner instance may be active per process; call close() first."
            )
        for name, value in (
            ("micro_batch_size", micro_batch_size),
            ("sequence_length", sequence_length),
            ("accumulation_steps", accumulation_steps),
            ("statistic_stride", statistic_stride),
            ("ns_steps", ns_steps),
            ("root_refresh", root_refresh),
            ("root_iterations", root_iterations),
            ("host_workers", host_workers),
            ("host_threads", host_threads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(statistic_head, int) or not 0 <= statistic_head <= sequence_length:
            raise ValueError("statistic_head must be between zero and sequence_length")
        for name, value in (("lr", lr), ("weight_decay", weight_decay)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("eps", eps), ("ns_eps", ns_eps)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= statistic_momentum < 1 or not 0 <= momentum < 1:
            raise ValueError("Momentum coefficients must be in [0, 1)")
        if len(betas) != 2 or not all(0 <= x < 1 for x in betas):
            raise ValueError("betas must contain two coefficients in [0, 1)")
        if any(not isinstance(x, int) or x < 0 for x in (root_inline_steps, root_repair_rounds)):
            raise ValueError("Root warmup and repair counts must be nonnegative integers")
        strengths = (float(inner_strength), float(outer_strength))
        if not set(strengths) <= {0.125, 0.25, 0.5}:
            raise ValueError("The fast roots support only strengths 1/8, 1/4 and 1/2")
        if kind == "adamt" and strengths != (0.25, 0.25):
            raise ValueError("FastAdamT preserves the matched-quarter configuration")
        self.blocks = [_projection_block(b) for b in blocks]
        if not self.blocks:
            raise ValueError("At least one projection block is required")
        first = self.blocks[0]
        if not all(type(block) is type(first) for block in self.blocks):
            raise ValueError("Use one uniform projection layout per fast optimizer")
        self._layout = "nanogpt" if isinstance(first, NanoGPTBlock) else "llama"
        self._names = _NANOGPT_NAMES if self._layout == "nanogpt" else _LLAMA_NAMES
        self.micro_batch_size, self.sequence_length = micro_batch_size, sequence_length
        self.accumulation_steps = accumulation_steps
        # Preserve toner.sample_positions exactly: the first position after `head`
        # is head + 1, not head.
        positions = (
            list(range(statistic_head))
            + list(range(statistic_head + 1, sequence_length, statistic_stride))
            if statistic_head > 0
            else list(range(1, sequence_length, statistic_stride))
        )
        if not positions:
            raise ValueError("The sampling rule selected no token positions")
        if not all(isinstance(getattr(first, n), nn.Linear) for n in self._names):
            raise TypeError("Projection block members must be nn.Linear modules")
        hidden = first.q.in_features
        mlp = first.fc.out_features if self._layout == "nanogpt" else first.gate.out_features
        if hidden % 256 or mlp % 256:
            raise ValueError("This CUDA adapter requires hidden and MLP widths divisible by 256")
        expected = {n: (hidden, hidden) for n in ("q", "k", "v")}
        if self._layout == "nanogpt":
            expected.update(attn_proj=(hidden, hidden), fc=(mlp, hidden), mlp_proj=(hidden, mlp))
        else:
            expected.update(
                o=(hidden, hidden), gate=(mlp, hidden), up=(mlp, hidden), down=(hidden, mlp)
            )
        named, bindings, modules, seen = {}, {}, [], set()
        device = first.q.weight.device
        if device.type != "cuda":
            raise ValueError("Fast Toner requires CUDA; use the reference implementation on CPU")
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("Fast Toner requires an Ampere-or-newer CUDA GPU")
        for i, block in enumerate(self.blocks):
            for name in self._names:
                module = getattr(block, name)
                if (
                    not isinstance(module, nn.Linear)
                    or tuple(module.weight.shape) != expected[name]
                ):
                    raise ValueError(
                        f"Block {i}.{name} must be a Linear with weight shape {expected[name]}"
                    )
                if self._layout == "llama" and module.bias is not None:
                    raise ValueError(
                        "ProjectionBlock supports bias-free projections; use NanoGPTBlock for biased nanoGPT layers"
                    )
                p = module.weight
                if (
                    p.device != device
                    or p.dtype != torch.float32
                    or not p.requires_grad
                    or not p.is_contiguous()
                ):
                    raise ValueError(
                        "All projection weights must be contiguous, trainable FP32 tensors on one CUDA device"
                    )
                if id(p) in seen:
                    raise ValueError("Shared projection parameters are not supported")
                seen.add(id(p))
                parameter_name = (
                    f"blocks.{i}.{_NANOGPT_PATHS[name]}.weight"
                    if self._layout == "nanogpt"
                    else f"body.blocks.{i}.{name}.weight"
                )
                named[parameter_name] = p
                bindings[parameter_name] = (i, _KINDS[name])
                modules.append((i, name, module))
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if world > 1 and dist.get_backend() != "nccl":
            raise ValueError(
                "Multi-GPU fast Toner requires NCCL and replicated, synchronized gradients"
            )
        if len(self.blocks) % world:
            raise ValueError("The number of blocks must divide evenly across the DDP ranks")
        super().__init__(
            list(named.values()), {"lr": float(lr), "weight_decay": float(weight_decay)}
        )
        self._closed = False
        self._capturing = False
        self._microbatches = 0
        self._seen = set()
        self._input_signatures = {}
        self._hooks = []
        self._device = device
        self._positions = torch.tensor(positions, device=device, dtype=torch.long)
        self._rows_per_microbatch = micro_batch_size * len(positions)
        local_rows = self._rows_per_microbatch * accumulation_steps
        buffers = [
            SimpleNamespace(
                **{
                    f"input_rows_{k}": torch.empty(
                        local_rows, n, device=device, dtype=torch.bfloat16
                    )
                    for k, n in (("attn", hidden), ("o", hidden), ("mlp", hidden), ("down", mlp))
                }
            )
            for _ in self.blocks
        ]
        self._buffers = buffers
        args = SimpleNamespace(
            toner_decay=float(statistic_momentum),
            toner_iters=root_iterations,
            toner_split=(strengths[0] != strengths[1] or strengths[0] == 0.125),
            toner_alpha=strengths[0],
            toner_alpha_in=strengths[0],
            toner_alpha_out=strengths[1],
            toner_root="ref",
            toner_refresh=root_refresh,
            toner_refresh_flagged=1,
            toner_owner_weights="",
            toner_lr_mult=1.0,
            wd=float(weight_decay),
            toner_overlap=1,
            toner_own_pg=1,
            toner_stats_ctas=0,
            toner_offload="",
            toner_host_workers=host_workers,
            toner_host_threads=host_threads,
            toner_host_cores="",
            toner_tol="floor",
            toner_tol_frac=0.2 / 3 if 0.5 in strengths else 0.2,
            toner_tol_min=5e-6,
            toner_eps=1e-4,
            toner_rho_max=0.2,
            toner_graphs=bool(root_graphs),
            toner_inline=root_inline_steps,
            toner_freeze=0.0,
            toner_repair_rounds=root_repair_rounds,
            toner_check_every=0,
            toner_outer=1,
            beta1=float(betas[0]),
            beta2=float(betas[1]),
            eps=float(eps),
            muon_momentum=float(momentum),
            muon_nesterov=bool(nesterov),
            muon_ns_steps=ns_steps,
            muon_eps=float(ns_eps),
            muon_ns_mode="tpu",
        )
        self._config = {
            "layout": self._layout,
            "kind": kind,
            "inner_strength": strengths[0],
            "outer_strength": strengths[1],
            "statistic_momentum": statistic_momentum,
            "momentum": momentum,
            "nesterov": bool(nesterov),
            "betas": tuple(betas),
            "eps": eps,
            "ns_steps": ns_steps,
            "ns_eps": ns_eps,
            "statistic_head": statistic_head,
            "statistic_stride": statistic_stride,
            "root_refresh": root_refresh,
            "root_iterations": root_iterations,
            "root_inline_steps": root_inline_steps,
            "root_repair_rounds": root_repair_rounds,
            "root_graphs": bool(root_graphs),
        }
        # Lazy import keeps ordinary CPU use free of CUDA/Triton dependencies.
        from .engine import FastTonerCore

        self._core = FastTonerCore(
            buffers,
            named,
            kind,
            hid=hidden,
            mlp=mlp,
            rows_total=local_rows * world,
            args=args,
            world=world,
            rank=rank,
            device=device,
            parameter_statistics=bindings,
        )
        self.param_groups = self._core.param_groups
        self.param_groups[0]["lr"] = float(lr)
        for i, name, module in modules:
            self._hooks.append(module.register_forward_pre_hook(self._make_hook(i, name)))
        _ACTIVE.add(self)

    def _assert_open(self):
        if self._closed:
            raise RuntimeError("This fast Toner optimizer has been closed")
        if torch.cuda.current_stream(self._device) != self._core.main_stream:
            raise RuntimeError("Use fast Toner on the CUDA stream on which it was constructed")

    def _make_hook(self, layer, name):
        def hook(module, inputs):
            if not self._capturing:
                return
            key = (layer, name)
            if key in self._seen:
                raise RuntimeError("Each projection must execute once per captured microbatch")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("Linear input must be a tensor")
            x = inputs[0].detach()
            if x.shape != (self.micro_batch_size, self.sequence_length, module.in_features):
                raise ValueError(
                    "Captured inputs must have the configured [micro_batch, sequence, features] shape"
                )
            autocast = torch.is_autocast_enabled("cuda")
            if autocast and torch.get_autocast_dtype("cuda") != torch.bfloat16:
                raise ValueError("Fast Toner supports BF16 autocast, not FP16 autocast")
            if x.dtype != torch.bfloat16 and not autocast:
                raise ValueError("Fast Toner expects BF16 Linear inputs or CUDA BF16 autocast")
            kind = _KINDS[name]
            signature = (x.data_ptr(), tuple(x.shape), tuple(x.stride()))
            previous = self._input_signatures.get((layer, kind))
            if previous is not None and previous != signature:
                raise ValueError(
                    "q/k/v must share an input tensor, as must gate/up; use separate statistics otherwise"
                )
            self._seen.add(key)
            if previous is None:
                self._input_signatures[(layer, kind)] = signature
                rows = x.index_select(1, self._positions).reshape(-1, x.size(-1))
                lo = self._microbatches * self._rows_per_microbatch
                getattr(self._buffers[layer], f"input_rows_{kind}")[
                    lo : lo + self._rows_per_microbatch
                ].copy_(rows)

        return hook

    @contextmanager
    def capture_microbatch(self):
        """Capture one forward, excluding backward/checkpoint recomputation.

        Call exactly accumulation_steps times between optimizer updates. Divide
        each microbatch loss consistently before backward; Toner does not rescale
        gradients. DDP synchronization remains the caller's responsibility.
        """
        self._assert_open()
        if self._capturing or self._core.begun or self._microbatches >= self.accumulation_steps:
            raise RuntimeError(
                "Invalid capture lifecycle: finish the current step before another forward"
            )
        self._seen.clear()
        self._input_signatures.clear()
        self._capturing = True
        try:
            yield
            if len(self._seen) != len(self.blocks) * len(self._names):
                raise RuntimeError(
                    "The forward did not execute every configured projection exactly once"
                )
        except BaseException:
            # Discard the partial step; no statistics have been submitted yet.
            self._microbatches = 0
            raise
        else:
            self._microbatches += 1
        finally:
            self._capturing = False

    def begin(self):
        """Start asynchronous statistics/roots after the final forward, before backward."""
        self._assert_open()
        if self._capturing or self._microbatches != self.accumulation_steps or self._core.begun:
            raise RuntimeError(
                "begin() requires all captured microbatches and may be called once per step"
            )
        self._core.begin()

    @torch.no_grad()
    def step(self, closure=None):
        self._assert_open()
        if closure is not None:
            raise ValueError(
                "Closures are unsupported; explicitly capture the forward before backward"
            )
        if self._capturing or self._microbatches != self.accumulation_steps:
            raise RuntimeError("Capture all microbatches before step()")
        if any(p.grad is None or p.grad.is_sparse for p in self.param_groups[0]["params"]):
            raise RuntimeError("Every projection requires a dense, synchronized gradient")
        self._core.step()
        self._microbatches = 0

    def state_dict(self):
        """Collective on all ranks; returns a full checkpoint only on rank zero.

        Call after a completed step. Save this together with the model and the
        auxiliary optimizer. Root bases are recomputed when loading.
        """
        self._assert_open()
        if self._capturing or self._microbatches or self._core.begun or not self._core.t:
            raise RuntimeError("Checkpoint only after a completed optimizer step")
        state = self._core.full_state()
        if state is None:
            return None
        return {
            "format": "toner.fast.v3",
            "config": dict(self._config),
            "core": state,
            "lr": self.param_groups[0]["lr"],
            "weight_decay": self.param_groups[0]["weight_decay"],
        }

    def load_state_dict(self, state_dict):
        """Load the same full checkpoint on every rank into a fresh optimizer."""
        self._assert_open()
        if self._microbatches or self._capturing or self._core.t:
            raise RuntimeError("Load into a fresh fast Toner optimizer before capturing inputs")
        if not isinstance(state_dict, dict) or state_dict.get("format") not in (
            "toner.fast.v2",
            "toner.fast.v3",
        ):
            raise ValueError("Expected a toner.fast.v3 checkpoint, broadcast to every rank")
        saved_config = dict(state_dict.get("config", {}))
        if state_dict["format"] == "toner.fast.v2":
            # Version 2 supported only the seven-projection Llama layout.
            if self._layout != "llama" or saved_config.get("layout", "llama") != "llama":
                raise ValueError("Version 2 checkpoints support only the Llama layout")
            saved_config.setdefault("layout", "llama")
        if saved_config != self._config:
            raise ValueError("Checkpoint layout or optimizer settings do not match this optimizer")
        core = state_dict["core"]
        if not isinstance(core.get("t"), int) or core["t"] < 1:
            raise ValueError("Checkpoint must contain a completed optimizer step")
        if not isinstance(core.get("weight"), (int, float)) or not 0 < core["weight"] <= 1:
            raise ValueError("Checkpoint statistic EMA weight is invalid")
        for key in ("lr", "weight_decay"):
            if (
                not isinstance(state_dict.get(key), (int, float))
                or not math.isfinite(state_dict[key])
                or state_dict[key] < 0
            ):
                raise ValueError(f"Checkpoint {key} must be finite and nonnegative")
        expected_ema = {
            f"{i}.{kind}": tuple(c.shape[1:])
            for group in self._core.groups.values()
            for i, kind in group.keys
            for c in [group.C]
        }
        expected_moments = {
            name: tuple(p.shape) for name, p in zip(self._core.names, self._core.params)
        }
        if set(core.get("ema", {})) != set(expected_ema) or set(core.get("moments", {})) != set(
            expected_moments
        ):
            raise ValueError("Checkpoint projection/statistic names do not match these blocks")
        for key, shape in expected_ema.items():
            if tuple(core["ema"][key].shape) != shape:
                raise ValueError(f"Checkpoint statistic shape differs for {key}")
        moment_keys = {"m"} if self._config["kind"] == "muont" else {"m", "v"}
        for key, shape in expected_moments.items():
            if set(core["moments"][key]) != moment_keys:
                raise ValueError(f"Checkpoint moments differ for {key}")
            if any(tuple(x.shape) != shape for x in core["moments"][key].values()):
                raise ValueError(f"Checkpoint moment shape differs for {key}")
        self._core.load_full_state(core)
        self.param_groups[0]["lr"] = state_dict["lr"]
        self.param_groups[0]["weight_decay"] = state_dict["weight_decay"]

    @property
    def diagnostics(self):
        """Root fallback counts and recent timing diagnostics from the core."""
        self._assert_open()
        return {
            "fallbacks": self._core.fallbacks,
            "extra_iterations": self._core.extra,
            "timing": dict(self._core.timing),
            "gpu_ms": dict(self._core.gpu_ms),
        }

    def close(self):
        """Finish root work, detach hooks and release the owned NCCL group (all ranks)."""
        if getattr(self, "_closed", True):
            return
        self._core.side.synchronize()
        self._core.pool.shutdown(wait=True)
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._core.pg is not None and dist.is_initialized():
            dist.destroy_process_group(self._core.pg)
        self._closed = True
        _ACTIVE.discard(self)


class FastMuonT(_FastToner):
    """Fast MuonT for uniform Llama-style or modded-nanoGPT blocks.

    Defaults use (1/2, 1/8), input-statistic momentum matching Muon’s momentum,
    ten BF16-input
    / FP32-output Newton–Schulz iterations, and shaped decay. Pass the matrix learning rate directly.
    """

    def __init__(self, blocks, lr, *, statistic_momentum=None, **kwargs):
        if statistic_momentum is None:
            statistic_momentum = kwargs.get("momentum", 0.95)
        super().__init__(blocks, lr, kind="muont", statistic_momentum=statistic_momentum, **kwargs)


class FastAdamT(_FastToner):
    """Fast AdamT, retaining matched quarter-power factors.

    The original gradient enters the first moment; the unnormalized transformed
    gradient enters the second. Mapping back preserves the adaptive update norm.
    Input-statistic momentum defaults to ``betas[1]``; an explicit value overrides it.
    """

    def __init__(self, blocks, lr, *, statistic_momentum=None, **kwargs):
        kwargs.setdefault("betas", (0.9, 0.98))
        if statistic_momentum is None:
            if len(kwargs["betas"]) != 2:
                raise ValueError("betas must contain two coefficients")
            statistic_momentum = kwargs["betas"][1]
        kwargs.setdefault("inner_strength", 0.25)
        kwargs.setdefault("outer_strength", 0.25)
        super().__init__(blocks, lr, kind="adamt", statistic_momentum=statistic_momentum, **kwargs)
