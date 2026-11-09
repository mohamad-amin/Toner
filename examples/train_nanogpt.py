"""Train modded-nanoGPT with MuonT or AdamT on prepared train.bin / val.bin token files."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from toner import ActivationStatistics, MuonT, AdamT
from model import GPT, GPTConfig


def build_optimizers(model, args, backend):
    """Partition hidden weights and construct their optimizers and input capture."""
    layers = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.startswith("blocks.")
    }
    matrices = [layer.weight for layer in layers.values()]
    matrix_ids = {id(p) for p in matrices}
    auxiliary_parameters = [p for p in model.parameters() if id(p) not in matrix_ids]
    options = dict(lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "muon":
        options.update(momentum=args.momentum, statistic_momentum=args.statistic_momentum)
    else:
        options["eps"] = 1e-15
        if args.statistic_momentum is not None:
            options["statistic_momentum"] = args.statistic_momentum
    if backend == "fast":
        from toner.fast import FastMuonT, FastAdamT, NanoGPTBlock

        blocks = [NanoGPTBlock.from_module(block) for block in model.blocks]
        optimizer_cls = FastMuonT if args.optimizer == "muon" else FastAdamT
        optimizer = optimizer_cls(
            blocks,
            micro_batch_size=args.batch_size,
            sequence_length=args.block_size,
            accumulation_steps=args.accumulation_steps,
            **options,
        )
        statistics = None
    else:
        if args.optimizer == "muon":
            options["ns_mode"] = "tpu" if args.device == "cuda" else "f32"
        optimizer_cls = MuonT if args.optimizer == "muon" else AdamT
        optimizer = optimizer_cls(matrices, **options)
        statistics = ActivationStatistics(layers)
    # Embedding, untied output head, biases and norm gains use separate AdamW.
    auxiliary = torch.optim.AdamW(auxiliary_parameters, lr=args.aux_lr, weight_decay=0.0)

    return optimizer, auxiliary, statistics, auxiliary_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/shakespeare_char"))
    parser.add_argument("--vocab-size", type=int, help="Defaults to meta.json in data-dir")
    parser.add_argument("--backend", choices=["auto", "reference", "fast"], default="auto")
    parser.add_argument("--optimizer", choices=["muon", "adam"], default="muon")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--aux-lr", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--statistic-momentum", type=float, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/nanogpt.pt"))
    args = parser.parse_args()
    for key in (
        "steps",
        "batch_size",
        "block_size",
        "layers",
        "head_dim",
        "width",
        "accumulation_steps",
        "eval_every",
        "eval_batches",
    ):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.width % args.head_dim or args.head_dim % 4:
        parser.error("width must divide into heads, and head-dim must be divisible by four")
    if args.vocab_size is None:
        meta = args.data_dir / "meta.json"
        if not meta.is_file():
            parser.error("Supply --vocab-size or a data-dir containing meta.json")
        args.vocab_size = json.loads(meta.read_text())["vocab_size"]
    if not isinstance(args.vocab_size, int) or args.vocab_size < 1:
        parser.error("vocab-size must be a positive integer")
    backend = (
        ("fast" if args.device == "cuda" else "reference")
        if args.backend == "auto"
        else args.backend
    )
    if backend == "fast" and (args.device != "cuda" or args.width % 256):
        parser.error("fast backend needs CUDA and width divisible by 256")
    data = {}
    for split in ("train", "val"):
        file = args.data_dir / f"{split}.bin"
        if not file.is_file() or file.stat().st_size % 2:
            parser.error(f"Expected a uint16 token file: {file}")
        data[split] = np.memmap(file, dtype=np.uint16, mode="r")
        if len(data[split]) <= args.block_size:
            parser.error(f"{split}.bin is too short for block-size={args.block_size}")
        if int(data[split].max()) >= args.vocab_size:
            parser.error(f"{split}.bin contains token IDs outside --vocab-size")
    torch.manual_seed(0)
    train_rng, eval_rng = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)

    def batch(split):
        tokens = data[split]
        starts = torch.randint(
            len(tokens) - args.block_size,
            (args.batch_size,),
            generator=train_rng if split == "train" else eval_rng,
        )
        x = torch.stack(
            [
                torch.from_numpy(np.array(tokens[i : i + args.block_size], dtype=np.int64))
                for i in starts.tolist()
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(np.array(tokens[i + 1 : i + args.block_size + 1], dtype=np.int64))
                for i in starts.tolist()
            ]
        )
        return x.to(args.device), y.to(args.device)

    config = GPTConfig(
        vocab_size=args.vocab_size, n_layer=args.layers, n_embd=args.width, head_dim=args.head_dim
    )
    model = GPT(config).to(args.device)
    optimizer, auxiliary, statistics, auxiliary_parameters = build_optimizers(model, args, backend)

    def autocast():
        return torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda")

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            with autocast():
                _, loss = model(*batch("val"))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    try:
        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            auxiliary.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro in range(args.accumulation_steps):
                capture = (
                    optimizer.capture_microbatch() if backend == "fast" else statistics.capture()
                )
                with capture, autocast():
                    _, loss = model(*batch("train"))
                if backend == "fast" and micro == args.accumulation_steps - 1:
                    optimizer.begin()  # overlap current-step statistics and factors with backward
                (loss / args.accumulation_steps).backward()
                loss_sum += loss.item() / args.accumulation_steps
            torch.nn.utils.clip_grad_norm_(
                auxiliary_parameters if args.optimizer == "muon" else model.parameters(), 1.0
            )
            if statistics is not None:
                statistics.commit(optimizer)
            optimizer.step()
            auxiliary.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                print(f"step {step}: train {loss_sum:.4f}, val {evaluate():.4f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": vars(config),
                "step": args.steps,
                "backend": backend,
                "optimizer": optimizer.state_dict(),
                "auxiliary_optimizer": auxiliary.state_dict(),
            },
            args.out,
        )
    finally:
        if statistics is not None:
            statistics.close()
        else:
            optimizer.close()


if __name__ == "__main__":
    main()
