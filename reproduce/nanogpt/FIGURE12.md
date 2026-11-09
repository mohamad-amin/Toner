# Figure 12 hyperparameters

Figure 12 compares optimizers on nanoGPT/FineWeb at approximately 1.7B tokens.
Learning rates below are the values applied to the hidden weight matrices.
The dashed baselines in the figure are published results from Dang et al.;
the runnable Muon configs in this repository reproduce our local controls.

## MuonT

The [loadable reproduction configs](README.md#training) include all settings.
MuonT uses strengths (½, ⅛), momentum 0.95 with Nesterov, input-statistic
momentum 0.95 and damping 0.001. All learning rates stay constant before the cooldown, then decay linearly
to zero. The cooldown fraction varies by batch, as shown below.

| Batch | Matrix LR | Shaped weight decay | Cooldown fraction | Config |
|---|---:|---:|---:|---|
| 128K | 0.017678 | 0.025 | 0.8 | [JSON](configs/muont_128k.json) |
| 512K | 0.035355 | 0.05 | 0.7 | [JSON](configs/muont_512k.json) |
| 1M | 0.035355 | 0.1 | 0.8 | [JSON](configs/muont_1m.json) |
| 2M | 0.035355 | 0.1 | 0.4 | [JSON](configs/muont_2m.json) |

## AdamT

AdamT uses strengths (¼, ¼), damping 0.001, Adam epsilon 1e-10 and
**input-statistic momentum 0.938** in these experiments. This explicit setting
differs from the library default, which now follows beta2. All learning rates
warm up linearly, stay constant, then decay linearly during the final 40% of
training at 128K–1M, or the final 60% at 2M. Auxiliary parameters use AdamW with zero weight decay.

| Batch | Matrix LR | Shaped weight decay | Matrix betas | Warmup updates |
|---|---:|---:|---|---:|
| 128K | 0.0010607 | 0.3 | (0.94, 0.94903) | 1000 |
| 512K | 0.0021213 | 0.6 | (0.9, 0.85858) | 250 |
| 1M | 0.0021213 | 0.8 | (0.85858, 0.93106) | 125 |
| 2M | 0.003 | 1.8 | (0.9, 0.86884) | 63 |

| Batch | Embedding LR | Head LR | Scalar LR | Auxiliary betas |
|---|---:|---:|---:|---|
| 128K | 0.84853 | 0.0088388 | 0.0282843 | (0.71716, 0.9955) |
| 512K | 0.84853 | 0.0088388 | 0.0282843 | (0.85858, 0.975) |
| 1M | 0.6 | 0.00625 | 0.02 | (0.85858, 0.9025) |
| 2M | 3.3941 | 0.0353553 | 0.113137 | (0.92929, 0.96721) |

[Recorded AdamT settings](figure12_adamt_settings.json) also specify
sampling and refresh settings. The 128K runs pool statistics once per step and
refresh cached factors through staggered background submissions every 16
updates after 200 initial updates; new factors are installed on the next update.
The larger batches use rank-local statistics updated per forward, adjusting
the EMA coefficient for accumulation, and refresh factors every step.
These are settings for the
plotted implementation; the current nanoGPT reproduction runner supports Muon
and MuonT, so this AdamT file is not a runnable trainer config.
