# Reproduce the nanoGPT comparison

These configurations train the same 162M-parameter modded-nanoGPT model with
Muon or MuonT at 128K, 512K, 1M and 2M tokens per update. They specify the model, initialization,
optimizer arithmetic, data order, schedule, and evaluation settings.

## Data

```bash
python reproduce/nanogpt/download_data.py --out data/fineweb10B
```

The downloader uses [FineWeb10B tokenized with GPT-2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2)
at revision `889765ea1f903759787add96995d81171b632d0c` and verifies all 33 files
against `data.sha256`. The download is approximately 6.6 GB. To verify an
existing copy, add `--check-only`.

Use this exact shard set. Adding more shards changes the training sequence. Sequence rows remain
contiguous within each shard.

## Training

Install the package with `pip install -e '.[examples,fast]'`, then run on four
CUDA GPUs:

```bash
torchrun --standalone --nproc_per_node=4 reproduce/nanogpt/train.py \
    --config reproduce/nanogpt/configs/muon_512k.json \
    --data-dir data/fineweb10B --out runs/muon_512k

torchrun --standalone --nproc_per_node=4 reproduce/nanogpt/train.py \
    --config reproduce/nanogpt/configs/muont_512k.json \
    --data-dir data/fineweb10B --out runs/muont_512k
```

The commands above run the 512K comparison. Choose another configuration from
the table below; each loss links to its JSON config.

| Batch tokens | Updates | Muon loss / config | MuonT (½, ⅛) loss / config |
|---|---:|---:|---:|
| 128K | 13,000 | [3.26920](configs/muon_128k.json) | [3.26281](configs/muont_128k.json) |
| 512K | 3,250 | [3.28023](configs/muon_512k.json) | [3.26856](configs/muont_512k.json) |
| 1M | 1,625 | [3.31017](configs/muon_1m.json) | [3.28610](configs/muont_1m.json) |
| 2M | 813 | [3.36554](configs/muon_2m.json) | [3.33353](configs/muont_2m.json) |

The 128K, 512K and 1M runs each process 1,703,936,000 tokens. The 2M runs
process 1,704,984,576 tokens because their budget rounds to 813 updates.

These are the paired local runs. Hardware, software versions, and numerical
execution can affect the reproduced losses; these values are reference outcomes,
not assertions of bitwise reproducibility.

## Numerical settings

The reproduction runner reads every optimizer setting from the configuration.
It uses the nanoGPT comparison’s momentum, finite Newton–Schulz map, statistic
updates, and fast-root implementation. It does not silently inherit the general
package’s convenience defaults. Learning rates are used directly, and the
auxiliary AdamW groups keep their own configured rates.

For a smaller, general integration example, use
[`examples/train_nanogpt.py`](../../examples/train_nanogpt.py).

For a short integration check, add `--max-steps 2`. This retains the configured
model, batch, validation set and learning-rate schedule, but stops early; it
does not reproduce the final-loss comparison. The runner saves the model and
auxiliary state plus one optimizer/statistic checkpoint per rank.

For the plotted MuonT and AdamT hyperparameters, see [Figure 12 settings](FIGURE12.md).
