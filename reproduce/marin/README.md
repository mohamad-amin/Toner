# Reproduce the Marin comparisons

This directory contains the **1× training-budget runs in Figure 1**: Muon,
MuonT with strengths (½, ⅛) and (¼, ¼), and the plotted AdamW/AdamT comparisons.
Each JSON config specifies the model, batch, schedule, optimizer settings and
reference final C4 validation loss.

| Model | Training tokens | Muon / MuonT batches | AdamW / AdamT batches | Configs |
|---|---:|---|---|---|
| 130M | 2.68B | 0.5M, 1M, 2M, 4M | 1M, 2M, 4M | [130M](configs/130m) |
| 300M | 6.04B | 0.5M, 1M, 2M, 4M | 1M, 2M, 4M | [300M](configs/300m) |
| 520M | 10.74B | 0.5M, 1M, 2M, 4M | 2M, 4M | [520M](configs/520m) |
| 1.2B | 24.16B | 1M, 2M, 4M, 8M | — | [1.2B](configs/1p2b) |

Batch labels use powers of two: 1M means 1,048,576 tokens. Exact budgets are
in each config; rounding to complete updates can slightly change the budget.
Files named `muont_*.json` use (½, ⅛); `muont_quarter_*.json` use (¼, ¼).

## Prepare the data

Install the data tools and accept the relevant dataset/tokenizer access terms
on Hugging Face, then authenticate with `hf auth login`:

```bash
pip install -e '.[examples,fast,marin-data]'
python reproduce/marin/prepare_data.py --out data/marin --workers 8
```

The preparation script uses fixed revisions of DCLM, StarCoderData and
Proof-Pile-2, tokenized with Llama 3.1's tokenizer. It also prepares all 16
Paloma validation subsets, including C4. The training token files occupy
approximately **102 GB**, excluding download caches and validation data.

[`data_plan.json`](data_plan.json) fixes source files, document-prefix quotas,
output order and expected token/document counts. Preparation checks those
counts before accepting a shard and stores a checksum for subsequent checks.
Use `--check-only` to check an existing directory. These generated checksums
protect file integrity; the original token files did not have recorded hashes.
For existing files without checksum receipts, document-count checks assume each
EOS marks one document. The tokenizer revision is pinned to the
available cached snapshot. Use `--split train` or `--split validation` to prepare
these separately.

For background on Marin's data and training pipeline, see the
[Marin documentation](https://marin.readthedocs.io/en/latest/). Use the fixed
plan here for these PyTorch runs rather than today's default data mixture.

## Train

The PyTorch configurations use eight CUDA GPUs, BF16 computation, a 4,096-token
context and document-packed attention. The recorded attention backend is FA3
on H100 GPUs. Install a PyTorch build that provides FA3 and the matching CUDA
and Triton dependencies. The integration checks use PyTorch 2.11 and Triton 3.6;
FA3 requires a compatible Hopper setup.

```bash
torchrun --standalone --nproc_per_node=8 --module reproduce.marin.train \
    --config reproduce/marin/configs/130m/muon_1m.json \
    --data data/marin --out runs/marin_130m_muon_1m

torchrun --standalone --nproc_per_node=8 --module reproduce.marin.train \
    --config reproduce/marin/configs/130m/muont_1m.json \
    --data data/marin --out runs/marin_130m_muont_1m
```

Choose another config to change the model, optimizer or batch. Learning rates
are explicit: no transfer multiplier is applied by the runner. The configs also
preserve input-statistic momentum, fast-root settings, matrix chunking and the
older 300M Muon baseline's distinct embedding initialization.

Toner collects sampled layer inputs during the forward pass and overlaps
statistic/factor computation with backpropagation. This runner uses the
recomputation-based capture from the Marin experiments, rather than the general
package's forward-hook adapter.

The runner evaluates C4, the metric in Figure 1, rather than every Paloma subset.
Add `--max-steps 2` for an early-stop execution check with the configured batch
and schedule. The final model and loss log are saved under `--out`; this is not
a training-resume interface.

The reported losses are reference outcomes. Software, hardware and numerical
execution can affect reproduced losses; matching a configuration does not
promise bitwise-identical training.

## Native Marin baselines

The Muon references at 520M/0.5M and 1.2B/1M come from Marin's native JAX runs. Their configs retain the recorded model, optimizer, data mixture, and training settings. They do not run through the PyTorch trainer.

To export these settings for inspection:

```bash
python reproduce/marin/native.py reproduce/marin/configs/520m/muon_0p5m.json --output /tmp/muon-520m-native.json
```

The original runtime is not fully recovered. The 520M run records Levanter `1.2.dev1190`, Haliax `1.4.dev327`, and JAX/JAXlib `0.5.0`, but that published Levanter wheel does not contain Muon. The 1.2B run's resumed-run metadata lists multiple Levanter, Haliax, and JAX versions. We therefore provide recorded settings rather than an unverified installation or launch command. The recorded Git commits identify **Marin**, not Levanter. Native data settings use Marin's original tokenized caches, not the PyTorch data plan.
