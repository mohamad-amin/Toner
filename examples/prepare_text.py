"""Prepare character-level train.bin / val.bin files from UTF-8 text."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="UTF-8 file; defaults to downloading Shakespeare"
    )
    parser.add_argument("--out", type=Path, default=Path("data/shakespeare_char"))
    args = parser.parse_args()
    if args.input:
        text = args.input.read_text(encoding="utf-8")
    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = urlopen(url, timeout=60).read().decode("utf-8")
    chars = sorted(set(text))
    if not chars or len(chars) > 65536 or len(text) < 20:
        parser.error("Expected nonempty text with at most 65,536 distinct characters")
    encoding = {char: index for index, char in enumerate(chars)}
    tokens = np.fromiter((encoding[char] for char in text), dtype=np.uint16, count=len(text))
    cut = int(len(tokens) * 0.9)
    args.out.mkdir(parents=True, exist_ok=True)
    tokens[:cut].tofile(args.out / "train.bin")
    tokens[cut:].tofile(args.out / "val.bin")
    (args.out / "meta.json").write_text(
        json.dumps({"vocab_size": len(chars), "characters": chars}, ensure_ascii=False) + "\n"
    )
    print(
        f"Prepared {cut:,} training and {len(tokens) - cut:,} validation tokens; vocabulary {len(chars)}"
    )


if __name__ == "__main__":
    main()
