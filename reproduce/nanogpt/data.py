"""FineWeb token shards with the 256-int32 header format."""

from pathlib import Path
import glob
import random
import torch


def load_shard(path, pin_memory=False):
    path = Path(path)
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    if header.numel() != 256 or header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"Invalid token-shard header: {path}")
    count = int(header[2])
    if count < 1 or path.stat().st_size != 1024 + 2 * count:
        raise ValueError(f"Token count does not match file size: {path}")
    tokens = torch.empty(count, dtype=torch.uint16, pin_memory=pin_memory)
    with path.open("rb", buffering=0) as file:
        file.seek(1024)
        if file.readinto(tokens.numpy()) != 2 * count:
            raise ValueError("Incomplete token shard")
    return tokens


def token_batches(
    pattern, batch_tokens, sequence_length=1024, world_size=1, rank=0, device="cpu", shuffle_seed=0
):
    files = sorted(
        Path(x) for x in (glob.glob(str(pattern)) if isinstance(pattern, (str, Path)) else pattern)
    )
    if not files:
        raise ValueError("No token shards found")
    if shuffle_seed:
        random.Random(shuffle_seed).shuffle(files)
    if (
        not 0 <= rank < world_size
        or batch_tokens % world_size
        or (batch_tokens // world_size) % sequence_length
    ):
        raise ValueError("Batch tokens must divide into rank-local sequences")
    local = batch_tokens // world_size
    file_iter = iter(files)
    tokens, pos = load_shard(next(file_iter), pin_memory=str(device).startswith("cuda")), 0
    while True:
        if pos + batch_tokens + 1 >= len(tokens):
            tokens, pos = load_shard(next(file_iter), pin_memory=str(device).startswith("cuda")), 0
        if len(tokens) < batch_tokens + 1:
            raise ValueError("Shard cannot hold one global batch")
        buf = tokens[pos + rank * local :][: local + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_tokens
        yield inputs.view(-1, sequence_length), targets.view(-1, sequence_length)
