# Fast CUDA path

The fast backend computes sampled input Grams, pooled statistics and inverse-power factors on a
side CUDA stream, overlapping the final microbatch's backward pass. It reuses a
recent eigenbasis, refines factors from the current statistic every step, and
repairs or falls back to exact eigendecomposition when needed. Refreshed host
bases are installed when ready; optimizer updates use the current statistic,
not a deliberately lagged root.

## Usage

Install the CUDA dependency with `pip install -e '.[fast]'`, then connect the
optimizer to compatible projection blocks:

```python
from toner.fast import FastMuonT, NanoGPTBlock

optimizer = FastMuonT(
    [NanoGPTBlock.from_module(block) for block in model.blocks],
    lr=matrix_lr,
    weight_decay=0.1,
    micro_batch_size=micro_batch_size,
    sequence_length=sequence_length,
    accumulation_steps=accumulation_steps,
)

optimizer.zero_grad(set_to_none=True)
for microbatch_index, batch in enumerate(microbatches):
    with optimizer.capture_microbatch(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = compute_loss(model, batch) / accumulation_steps
    if microbatch_index == accumulation_steps - 1:
        optimizer.begin()             # async statistics/roots before backward
    loss.backward()
optimizer.step()                     # waits for factors; gradients must already be synchronized
# Step the separate auxiliary optimizer as well.
# At the end, on every distributed rank:
optimizer.close()
```

## Projection layouts

`NanoGPTBlock.from_module(block)` maps the six hidden matrices in a modded-nanoGPT
block: `attn.q`, `attn.k`, `attn.v`, `attn.proj`, `mlp.fc` and `mlp.proj`.
Projection biases are allowed but remain in the auxiliary optimizer. Toner uses
four input statistics: one shared by Q/K/V, plus one each for attention output,
MLP expansion and MLP output.

For uniform, bias-free Llama-style blocks, use
`ProjectionBlock.from_module(block)` instead. This maps `q`, `k`, `v`, `o`,
`gate`, `up` and `down`; gate/up share their input statistic.

The fast and reference backends use different floating-point products and root
computations, so their trajectories need not be bitwise identical.

Choosing a layout does not change the optimizer's numerical rule. Both use the
same FastMuonT or FastAdamT updates, current-step statistics and root engine.
FastMuonT defaults to ten BF16-input/FP32-output Newton–Schulz iterations and raw
momentum accumulation; `NanoGPTBlock` does not substitute a different Muon rule.

## Supported integration

- All blocks in one optimizer use the same layout and dimensions. Hidden and MLP
  widths must be multiples of 256. Grouped-query attention, fused QKV and uneven
  attention projection widths are not supported by these adapters.
- Q/K/V share the same input tensor; Llama gate/up share another. Each configured
  projection runs exactly once per captured microbatch, with the configured
  batch and sequence shape. Capture ends before backward, excluding activation
  recomputation.
- Parameters are contiguous FP32 masters on one Ampere-or-newer CUDA GPU per
  process. Inputs are BF16, or the forward runs under CUDA BF16 autocast. FP16
  autocast is rejected. Capture, begin and step use the construction CUDA stream.
  Sampled inputs must be finite and have nonzero second-moment trace; the retained
  relative-ridge normalization is undefined for an all-zero input statistic.
- Multi-GPU use requires an initialized default NCCL process group and replicated
  parameters with synchronized gradients (e.g. DDP). The block count must divide
  evenly across ranks. Statistics are pooled globally and updates are sharded
  internally. FSDP/ZeRO-sharded parameters are not supported by this adapter.
- Only one fast Toner optimizer may be active per process because the retained
  root engine uses shared compiled configuration. Close it before creating another.
- Fast MuonT supports the inverse-root strengths ⅛, ¼ and ½. Fast AdamT
  retains matched quarters. Damping is fixed at 0.001. Use the
  readable implementation for arbitrary strengths or damping.

`lr` is the matrix learning rate and is used directly. FastMuonT inherits
`statistic_momentum` from `momentum` when omitted or `None`; an explicit value
overrides it. FastAdamT similarly inherits `betas[1]` when `statistic_momentum` is omitted or
`None`.
Update `optimizer.param_groups[0]["lr"]` to apply a learning-rate schedule;
`weight_decay` in the same group is also read at every step. Embedding, output-head,
normalization parameters and projection biases require a separate optimizer. The first twenty
steps use exact roots by default. `root_inline_steps`, `root_iterations`,
`root_refresh` and `root_graphs` expose the root-engine controls;
changing them changes the numerical/performance configuration.

## Checkpoints

After a completed step, call `optimizer.state_dict()` on **every rank**. It is a
collective operation; only rank zero receives the full CPU checkpoint. Save it
along with model and auxiliary-optimizer state. To resume, broadcast/load the
same optimizer checkpoint on every rank into a fresh optimizer with the same
layout and settings. New checkpoints use `toner.fast.v3` and record the layout.
The earlier `toner.fast.v2` format loads only into the Llama layout.

Statistic EMAs and gradient moments are preserved. Root bases are recomputed and
the exact-root startup phase restarts after loading, so resumed
numerical trajectories need not be bitwise identical. The adapter validates the
optimizer configuration and parameter names/shapes when loading. When finished,
all ranks call `close()`: it waits for root work, detaches capture hooks and
releases its own NCCL group without destroying the application’s default group.

## Execution

The adapter collects inputs with forward hooks. Call `begin()` after the last
forward and before backward to overlap factor computation. Startup compilation
and exact eigendecompositions can make the first steps slower.
