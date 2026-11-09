"""Train the configured nanoGPT Muon or MuonT comparison with torchrun."""

import argparse
import json
import os
from pathlib import Path


def lr_scale(step, total_steps, cooldown_fraction):
    if not 0 <= step < total_steps or not 0 < cooldown_fraction <= 1:
        raise ValueError("Invalid schedule position or cooldown fraction")
    progress = step / total_steps
    return 1.0 if progress < 1 - cooldown_fraction else (1 - progress) / cooldown_fraction


def load_config(path):
    config = json.loads(Path(path).read_text())
    train = config["training"]
    opt = config["optimizer"]
    if train["gradient_reduction"] != "sum" or train["clip_grad_norm"] is not None:
        raise ValueError("This training recipe uses summed gradients without clipping")
    if (opt["ns_steps"], opt["ns_coefficients"], opt["ns_epsilon"], opt["ns_dtype"]) != (
        12,
        [2.0, -1.5, 0.5],
        1e-7,
        "bfloat16",
    ):
        raise ValueError("Unsupported Muon numerical recipe")
    if train["batch_tokens"] % (
        train["world_size"] * train["sequence_length"] * train["micro_batch_size"]
    ):
        raise ValueError("Global batch does not divide into local microbatches")
    if not config["statistics"]["global"]:
        raise ValueError("This recipe requires globally pooled statistics")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/nanogpt-reproduction"))
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop early for an integration check; retain the configured schedule",
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable model compilation for debugging"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["OMP_NUM_THREADS"] = str(cfg.get("runtime", {}).get("omp_num_threads", 2))
    import torch
    import torch.distributed as dist
    from model import GPT, initialize_model
    from optimizer import build_optimizers
    from data import token_batches

    if not torch.cuda.is_available():
        parser.error("This reproduction entry point requires CUDA")
    required = cfg["data"]["files"]
    for name in required:
        if not (args.data_dir / name).is_file():
            parser.error(f"Missing configured shard: {name}")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world != cfg["training"]["world_size"]:
        parser.error(
            f"Configuration requires {cfg['training']['world_size']} processes; use torchrun"
        )
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    dist.init_process_group(backend="nccl", device_id=device)
    auxiliary = body = None
    hooks = []
    try:
        train = cfg["training"]
        evaluation = cfg["evaluation"]
        total = train["steps"]
        stop = total if args.max_steps is None else args.max_steps
        if not 0 < stop <= total:
            raise ValueError("max-steps must be between1 and the configured step count")
        sequence = train["sequence_length"]
        micro = train["micro_batch_size"]
        val_files = [args.data_dir / name for name in required if name.startswith("fineweb_val_")]
        train_files = [
            args.data_dir / name for name in required if name.startswith("fineweb_train_")
        ]
        val_tokens = cfg["data"]["validation_tokens"]
        validation = token_batches(
            val_files, val_tokens, sequence, world, rank, device, shuffle_seed=0
        )
        val_x, val_y = next(validation)
        model = GPT(**cfg["model"]).to(device)
        if cfg.get("runtime", {}).get("compile_model", True) and not args.no_compile:
            model.compile(dynamic=False)
        initialize_model(model, train.get("seed", 1))
        auxiliary, body, hooks = build_optimizers(model, cfg)
        loaders = token_batches(
            train_files,
            train["batch_tokens"],
            sequence,
            world,
            rank,
            device,
            shuffle_seed=train.get("seed", 1),
        )
        for parameter in model.parameters():
            dist.broadcast(parameter.detach(), 0)
        args.out.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            (args.out / "execution.json").write_text(
                json.dumps(
                    {
                        "requested_steps": stop,
                        "configured_steps": total,
                        "model_compile": not args.no_compile
                        and cfg.get("runtime", {}).get("compile_model", True),
                        "world_size": world,
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(device),
                    },
                    indent=2,
                )
                + "\n"
            )
        model.zero_grad(set_to_none=True)
        for step in range(stop + 1):
            frequency = (
                evaluation["early_interval"]
                if step < evaluation["switch_fraction"] * total
                else evaluation["late_interval"]
            )
            if step == stop or step % frequency == 0:
                dist.barrier()
                model.eval()
                if len(val_x) % micro:
                    raise ValueError("Validation batch must divide into microbatches")
                loss = torch.zeros((), device=device)
                with torch.no_grad():
                    for first in range(0, len(val_x), micro):
                        loss += model(val_x[first : first + micro], val_y[first : first + micro])
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= val_tokens
                if rank == 0:
                    print(f"step {step}/{total}: val_loss {loss.item():.5f}", flush=True)
                    with (args.out / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps({"step": step, "val_loss": loss.item()}) + "\n")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Validation loss is not finite")
                model.train()
                dist.barrier()
            if step == stop:
                break
            inputs, targets = next(loaders)
            for first in range(0, len(inputs), micro):
                loss = model(inputs[first : first + micro], targets[first : first + micro])
                if first + micro == len(inputs) and cfg["optimizer"]["name"] == "muont":
                    body.begin()
                loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError("Missing gradient")
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            scale = lr_scale(step, total, cfg["schedule"]["cooldown_fraction"])
            for optimizer in [auxiliary, body]:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                optimizer.step()
            model.zero_grad(set_to_none=True)
        # Optimizer state is owner-sharded. Retain one file per rank, plus the model.
        statistics = {
            name: {"ema": m.stat_cov.cpu(), "weight": m.stat_weight.cpu()}
            for name, m in model.named_modules()
            if hasattr(m, "stat_cov")
        }
        torch.save(
            {"step": stop, "optimizer": body.state_dict(), "statistics": statistics},
            args.out / f"optimizer-rank{rank}.pt",
        )
        if rank == 0:
            torch.save(
                {
                    "step": stop,
                    "model": model.state_dict(),
                    "auxiliary": auxiliary.state_dict(),
                    "config": cfg,
                },
                args.out / "model.pt",
            )
    finally:
        if body is not None:
            body.close()
        for hook in hooks:
            hook.remove()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
