"""Reproduce the recorded PyTorch Marin configurations with torchrun."""

import argparse
import json
import math
import os
from pathlib import Path


def schedule_factor(step, schedule):
    """Learning-rate multiplier at a zero-based optimizer step."""
    warm = schedule["warmup_steps"]
    if warm and step < warm:
        return step / warm
    step -= warm
    stable, decay = schedule["stable_steps"], schedule["decay_steps"]
    if step < stable or not decay or schedule["kind"] == "constant":
        return 1.0
    fraction = min(step - stable, decay) / decay
    minimum = schedule["min_lr_ratio"]
    if schedule["kind"] == "linear":
        return 1.0 + (minimum - 1.0) * fraction
    if schedule["kind"] == "cosine":
        return (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * fraction)) + minimum
    raise ValueError(f"Unknown schedule {schedule['kind']}")


def load_config(path):
    with open(path) as stream:
        config = json.load(stream)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Prepared uint32 shard directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop early for an execution check; does not alter the schedule",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    config = load_config(args.config)
    if config["engine"] != "pytorch":
        raise SystemExit(
            "This point uses native Marin/Levanter. Follow the native-run instructions in the Marin README."
        )
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F

    from . import head_ce
    from .data import MarinMixture, eval_sets, pack, validate_data_plan
    from .model import Model, initialize_model, sample_positions
    from .optimizer import build_optimizers, clip_gradients

    if not torch.cuda.is_available():
        raise SystemExit(
            "These configurations require CUDA, Triton and document-packed FlashAttention."
        )
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    m, t, runtime = config["model"], config["training"], config["runtime"]
    if m["hidden"] % 256:
        raise ValueError("The recorded LM-head kernels require hidden width divisible by 256")
    if runtime["attention"] not in ("FA2", "FA3"):
        raise ValueError("attention must be FA2 or FA3")
    if runtime["head"] not in ("compiled", "triton"):
        raise ValueError("head must be compiled or triton")
    if runtime["compile"] not in (
        "eager",
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ):
        raise ValueError("Unsupported torch.compile mode")
    if runtime["eval_every"] <= 0 or t["steps"] <= 0:
        raise ValueError("Training steps and evaluation interval must be positive")
    validate_data_plan(args.data, Path(__file__).parent / config["data"]["manifest"])
    if world != t["world_size"]:
        raise SystemExit(f"Config requires {t['world_size']} ranks; received {world}.")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(t.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if runtime["attention"] == "FA3":
        torch.nn.attention.activate_flash_attention_impl("FA3")
    micro, seq, batch = t["micro_batch_size"], m["sequence_length"], t["global_batch_sequences"]
    if batch % (world * micro):
        raise ValueError("Global batch must divide into equal full microbatches")
    accumulation = batch // (world * micro)
    capture = config["optimizer"]["name"] in ("muont", "adamt")
    positions, rows, chunk = None, 0, accumulation
    if capture:
        s = config["statistics"]
        positions = sample_positions(seq, s["head"], s["stride"])
        rows = micro * len(positions)
        chunk = config["root"]["chunk"] or accumulation
        if accumulation % chunk:
            raise ValueError("Statistic chunks must divide the accumulated microbatches")
        if runtime["capture"] != "recompute":
            raise ValueError("This reproduction uses recomputed input-row capture")
    model = Model(m, positions=positions, row_capacity=chunk * rows, rows_per_microbatch=rows).to(
        device
    )
    initialize_model(model, t.get("seed", 0), m["embedding_init_scale"])
    optimizers, fast, clipping = build_optimizers(
        model, config, rows_total=accumulation * rows * world, world=world, rank=rank, device=device
    )
    body = (
        model.body
        if runtime["compile"] == "eager"
        else torch.compile(model.body, mode=runtime["compile"], dynamic=False)
    )

    def compiled_loss(h, w, target):
        return F.cross_entropy(F.linear(h, w.to(torch.bfloat16)).float(), target, ignore_index=-100)

    loss_fn = (
        compiled_loss
        if runtime["compile"] == "eager"
        else torch.compile(compiled_loss, mode=runtime["compile"], dynamic=False)
    )

    def forward_loss(token, target, cu, meta):
        h = body(token, cu, meta).to(torch.bfloat16).reshape(-1, m["hidden"])
        if runtime["head"] == "triton":
            return head_ce.HeadCE.apply(h, model.head.weight, target.reshape(-1))
        return loss_fn(h, model.head.weight, target.reshape(-1))

    model.forward = forward_loss
    run = model
    if world > 1:
        # Attention metadata stays on the CPU; moving it creates a synchronization per layer.
        run = nn.parallel.DistributedDataParallel(
            model, device_ids=None, gradient_as_bucket_view=True, broadcast_buffers=False
        )
    armed = {"value": False}

    def begin_hook(_):
        if armed["value"]:
            armed["value"] = False
            fast.begin()

    if fast and config["root"]["begin"] == "head":
        model.head.weight.register_post_accumulate_grad_hook(begin_hook)
    mixture = MarinMixture(
        args.data,
        weights=config["data"]["mixture_weights"],
        seq=seq,
        seed=t.get("seed", 0),
        heldout_windows=t["heldout_windows"],
    )
    evaluation = eval_sets(args.data, include=["c4_en"], seq=seq)
    if "paloma/c4_en" not in evaluation:
        raise ValueError("Missing Paloma C4 shards under DATA/paloma/c4_en")
    cap = micro * seq // 2 + 1

    def to_device(tokens):
        tok, target, cu, meta = pack(tokens, cap, config["data"]["eos_token_id"])
        return (
            tok.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
            cu.to(device, non_blocking=True),
            meta,
        )

    def evaluate(dataset):
        n = len(dataset)
        if runtime["eval_max_windows"]:
            n = min(n, runtime["eval_max_windows"])
        per_rank = math.ceil(n / world)
        total = torch.zeros(2, dtype=torch.float64, device=device)
        # Use the training graph, then detach, to keep its BF16 rounding and compilation path.
        for offset in range(0, per_rank, micro):
            start = rank * per_rank + offset
            real = max(0, min(micro, per_rank - offset, n - start))
            tokens = dataset.batch(start, real) if real else np.empty((0, seq), dtype=np.uint32)
            if real < micro:
                tokens = np.concatenate(
                    (tokens, np.repeat(dataset.batch(0, 1), micro - real, axis=0))
                )
            tok, target, cu, meta = to_device(tokens)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = body(tok, cu, meta).to(torch.bfloat16).detach().reshape(-1, m["hidden"])
                _, _, losses = head_ce.head_ce_forward(
                    h, model.head.weight.to(torch.bfloat16), target.reshape(-1)
                )
            total[0] += losses.reshape(micro, seq)[:real].double().sum()
            total[1] += (target[:real] != -100).sum()
        if world > 1:
            dist.all_reduce(total)
        return (total[0] / total[1]).item()

    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    steps = min(t["steps"], args.max_steps) if args.max_steps else t["steps"]
    for step in range(steps):
        model.zero_grad(set_to_none=True)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * schedule_factor(step, config["schedule"])
        loss_sum = torch.zeros((), device=device)
        for i in range(accumulation):
            if world > 1:
                run.require_backward_grad_sync = i == accumulation - 1
            if fast:
                if chunk < accumulation and i % chunk == 0 and i:
                    fast.chunk_wait()
                model.body.stat_at.copy_(
                    torch.arange((i % chunk) * rows, (i % chunk + 1) * rows, device=device)
                )
            start = step * batch + (rank * accumulation + i) * micro
            packed = to_device(mixture.batch(start, micro))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = run(*packed)
            if fast and chunk < accumulation and (i + 1) % chunk == 0 and i < accumulation - 1:
                fast.chunk_stats()
            if fast and i == accumulation - 1:
                if config["root"]["begin"] == "head":
                    armed["value"] = True
                else:
                    fast.begin()
            (loss / accumulation).backward()
            loss_sum += loss.detach() / accumulation
        clip_gradients(clipping, config["optimizer"]["clip_norm"])
        for optimizer in optimizers:
            optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
            loss_sum /= world
        metrics = {"step": step + 1, "train_loss": loss_sum.item()}
        if step % runtime["eval_every"] == 0 or step + 1 == steps:
            metrics["c4_loss"] = evaluate(evaluation["paloma/c4_en"])
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            with (args.out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
    if rank == 0:
        torch.save(
            {"model": model.state_dict(), "step": steps, "config": config}, args.out / "model.pt"
        )
    if fast:
        fast.side.synchronize()
        fast.pool.shutdown(wait=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
