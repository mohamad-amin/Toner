# Toner

### Improving Muon and AdamW with input geometry

Toner uses each layer’s input statistics to precondition its weight update.
Use **MuonT** with Muon or **AdamT** with AdamW, through either the general
PyTorch implementation or the asynchronous CUDA backend.

[Read the blog](https://muon-sfd-half-eighth-amin-1105.leamin.chatgpt.site/)
· [Training example](examples/train_nanogpt.py)
· [Fast CUDA guide](docs/FAST.md)

## Quick start

```bash
git clone https://github.com/mohamad-amin/Toner.git
cd Toner
pip install -e '.[examples,fast]'

python examples/prepare_text.py
python examples/train_nanogpt.py --optimizer muon --backend fast
```

This trains the included modded-nanoGPT model on Shakespeare, reports training
and validation loss, and saves a checkpoint. Use `--optimizer adam` for AdamT.
The model has separate Q/K/V projections, rotary attention, and a squared-ReLU MLP.

For your own text, pass `--input corpus.txt` to `prepare_text.py`. Prepared
uint16 token files are also supported through `--data-dir` and `--vocab-size`.
Set the matrix and auxiliary learning rates directly with `--lr` and `--aux-lr`.

## Reproduce a comparison

The [nanoGPT reproduction guide](reproduce/nanogpt/README.md) provides loadable
Muon/MuonT configurations at **128K, 512K, 1M and 2M** tokens per update, pinned
data shards, and four-GPU training commands.

The [Marin reproduction guide](reproduce/marin/README.md) provides the Figure 1
configurations for **130M, 300M, 520M and 1.2B models** at their 1× training
budgets, with data preparation and an eight-GPU PyTorch runner. The two native
Marin baselines retain separate JAX configurations.

## Choose an optimizer

| Optimizer | Inner strength | Outer strength | Input-statistic momentum |
|---|---:|---:|---|
| `MuonT` / `FastMuonT` | ½ | ⅛ | Matches `momentum` by default |
| `AdamT` / `FastAdamT` | ¼ | ¼ | Matches `betas[1]` by default |

Both use shaped weight decay and take `lr` exactly as supplied. Set
`statistic_momentum` explicitly when you want it to differ from the default.

## Add Toner to your model

Select the hidden `nn.Linear` layers to optimize. Embeddings, output heads,
normalization gains, and biases belong in a separate optimizer; the
[training example](examples/train_nanogpt.py) shows the complete setup.

Set `matrix_lr` to the matrix learning rate for your training setup. To reproduce
our experiments, use the value in the corresponding [reproduction config](#reproduce-a-comparison).

```python
from toner import ActivationStatistics, MuonT

layers = {"projection": model.projection}
optimizer = MuonT(
    [layer.weight for layer in layers.values()],
    lr=matrix_lr,
    weight_decay=0.1,
    momentum=0.95,
    ns_mode="tpu",
)
statistics = ActivationStatistics(layers)

optimizer.zero_grad(set_to_none=True)
with statistics.capture():
    loss = compute_loss(model, batch)
loss.backward()
statistics.commit(optimizer)
optimizer.step()

statistics.close()  # once training is finished
```

For gradient accumulation, capture each forward and commit once per optimizer
step. Capture only the forward pass so backward recomputation does not count
inputs twice. With DDP, the collector pools input statistics across ranks;
DDP synchronizes gradients separately.

## Fast CUDA integration

`FastMuonT` and `FastAdamT` compute statistics and factors alongside the final
microbatch’s backward pass. Choose the adapter for your model:

| Adapter | Projection layout |
|---|---|
| `NanoGPTBlock` | Modded-nanoGPT: separate Q/K/V, attention output, and a two-layer MLP |
| `ProjectionBlock` | Llama-style: separate Q/K/V, attention output, and a gated MLP |

The fast backend requires FP32 weights, BF16 activations, uniform block shapes,
and hidden/MLP widths divisible by 256. The example selects it automatically
on CUDA. See the [fast guide](docs/FAST.md) for integration and checkpointing.

Use the general implementation for other `nn.Linear` layouts. It computes
factors by eigendecomposition every step; MuonT still uses a finite
Newton–Schulz iteration for its polar operation.

---

Apache-2.0 · [License](LICENSE)

Mohamad Amin Mohamadi · Kaiyue Wen · Tianhao Wang · Zhiyuan Li
