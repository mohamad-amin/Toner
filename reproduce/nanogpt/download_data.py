"""Download and verify the FineWeb shards used by the nanoGPT configurations."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "889765ea1f903759787add96995d81171b632d0c"
BASE_URL = f"https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/{REVISION}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/fineweb10B"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    entries = [
        line.split()
        for line in Path(__file__).with_name("data.sha256").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    for expected, name in entries:
        if Path(name).name != name:
            raise ValueError("Shard names must be plain filenames")
        target = args.out / name
        if target.is_file() and sha256(target) == expected:
            print(f"verified {name}", flush=True)
            continue
        if args.check_only:
            raise SystemExit(f"Missing or mismatched shard: {target}")
        partial = target.with_suffix(target.suffix + ".part")
        print(f"downloading {name}", flush=True)
        try:
            with urlopen(f"{BASE_URL}/{name}", timeout=120) as response, partial.open("wb") as file:
                while chunk := response.read(8 * 1024 * 1024):
                    file.write(chunk)
            if sha256(partial) != expected:
                raise RuntimeError(f"Checksum mismatch for {name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    print(f"All {len(entries)} shards verified.")


if __name__ == "__main__":
    main()
