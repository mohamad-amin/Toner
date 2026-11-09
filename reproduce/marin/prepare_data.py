"""Prepare the fixed document prefixes used by the PyTorch Marin comparisons."""

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

TOK = None


def init_worker(tokenizer, revision):
    global TOK
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(tokenizer, revision=revision)


def encode(texts):
    ids = TOK(
        [t + " " + TOK.eos_token for t in texts],
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]
    return [np.asarray(x, dtype="<u4") for x in ids]


def iter_texts(repo, rev, path):
    """Yield document texts from a dataset file, streaming (stop early by closing the generator)."""
    from urllib.parse import quote

    import requests

    url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{quote(path)}"  # e.g. "val_09_c#.jsonl.gz" needs %23
    from huggingface_hub import get_token

    token = get_token()
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(f"datasets/{repo}@{rev}/{path}", "rb", block_size=8 << 20) as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(columns=["content"], batch_size=256):
                for t in batch.column(0).to_pylist():
                    if t:
                        yield t
        return
    with requests.get(url, headers=hdr, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        if path.endswith(".zst"):
            import zstandard

            raw = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(r.raw)
        elif path.endswith(".gz"):
            raw = gzip.GzipFile(fileobj=r.raw)
        else:
            raw = r.raw
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
            if line.strip():
                t = json.loads(line).get("text")
                if t:
                    yield t


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, record):
    if not path.exists() or path.stat().st_size != record["tokens"] * 4:
        return False
    receipt = path.with_suffix(".bin.json")
    if receipt.exists():
        metadata = json.loads(receipt.read_text())
        return (metadata.get("tokens"), metadata.get("docs"), metadata.get("sha256")) == (
            record["tokens"],
            record["docs"],
            checksum(path),
        )
    if record["tokens"] == 0:
        return record["docs"] == 0
    # Counts catch truncation and tokenizer mismatches; they are not content hashes.
    values = np.memmap(path, mode="r", dtype="<u4")
    documents = sum(
        int(np.count_nonzero(values[i : i + 1_000_000] == 128001))
        for i in range(0, len(values), 1_000_000)
    )
    return documents == record["docs"]


def work(job):
    record, source, out = job
    path = Path(out) / record["component"] / record["file"]
    if path.exists():
        if not verify_file(path, record):
            raise ValueError(f"Existing shard does not match expected counts: {path}")
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".bin.tmp")
    for attempt in range(4):
        try:
            got, docs, buf = 0, 0, []
            with tmp.open("wb") as fh:
                for text in iter_texts(source["repo"], source["revision"], record["path"]):
                    buf.append(text)
                    if len(buf) == 64:
                        for ids in encode(buf):
                            ids.tofile(fh)
                            got += len(ids)
                            docs += 1
                        buf = []
                        if record["quota"] is not None and got >= record["quota"]:
                            break
                if buf and (record["quota"] is None or got < record["quota"]):
                    for ids in encode(buf):
                        ids.tofile(fh)
                        got += len(ids)
                        docs += 1
            if (got, docs) != (record["tokens"], record["docs"]):
                raise ValueError(f"Token/document count mismatch for {path}: {got}, {docs}")
            digest = checksum(tmp)
            os.replace(tmp, path)
            path.with_suffix(".bin.json").write_text(
                json.dumps({"tokens": got, "docs": docs, "sha256": digest}) + "\n"
            )
            return str(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=["all", "train", "validation"], default="all")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    plan = json.loads(Path(__file__).with_name("data_plan.json").read_text())
    records = [
        r
        for r in plan["files"]
        if args.split == "all"
        or r["component"].startswith("paloma/") == (args.split == "validation")
    ]
    if args.check_only:
        bad = [
            str(Path(r["component"]) / r["file"])
            for r in records
            if not verify_file(args.out / r["component"] / r["file"], r)
        ]
        if bad:
            raise SystemExit(f"Missing or mismatched shards ({len(bad)}): {bad[:10]}")
        print(f"Verified token/document counts for {len(records)} shards.")
        return
    init_worker(plan["tokenizer"], plan["tokenizer_revision"])
    jobs = [(r, plan["sources"][r["component"]], str(args.out)) for r in records]
    with mp.get_context("spawn").Pool(
        args.workers,
        initializer=init_worker,
        initargs=(plan["tokenizer"], plan["tokenizer_revision"]),
    ) as pool:
        for count, path in enumerate(pool.imap_unordered(work, jobs), 1):
            print(f"[{count}/{len(jobs)}] {path}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_plan.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
