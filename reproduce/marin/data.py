"""Deterministic uint32 token streams and the Marin training mixture."""

import os
import resource

import numpy as np

# every shard stays memory-mapped (one descriptor each; the image has ~2,300 shards): raise the soft limit to the hard one
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft != resource.RLIM_INFINITY and (_hard == resource.RLIM_INFINITY or _soft < _hard):
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))

SEQ = 4096
WEIGHTS = {
    "dclm": 3.8,
    "starcoder": 0.25,
    "proofpile": 0.055,
}  # Marin's DCLM mixture (unnormalized)


class Stream:
    """A component's token stream: its shards back to back, read as windows of `seq` tokens."""

    def __init__(self, files, seq=SEQ):
        files = [
            f for f in files if os.path.getsize(f) > 0
        ]  # a source file can be empty (e.g. 5 tiny ProofPile files)
        self.mm = [np.memmap(f, dtype=np.uint32, mode="r") for f in files]
        self.sizes = np.array([m.size for m in self.mm], dtype=np.int64)
        self.cum = np.concatenate(([0], np.cumsum(self.sizes)))
        self.seq = seq
        self.n_windows = int(self.cum[-1] // seq)

    def window(self, w):
        start, end = w * self.seq, (w + 1) * self.seq
        s = int(np.searchsorted(self.cum, start, side="right") - 1)
        parts, pos = [], start
        while pos < end:
            off = pos - self.cum[s]
            take = min(end - pos, self.sizes[s] - off)
            parts.append(self.mm[s][off : off + take])
            pos += take
            s += 1
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


class MarinMixture:
    def __init__(self, root, weights=WEIGHTS, seq=SEQ, block_size=2048, seed=0, heldout_windows=0):
        self.block_size, self.seed = block_size, seed
        self.names = list(weights)
        self.streams, self.perms, self.heldout = [], [], []
        for i, name in enumerate(self.names):
            files = sorted(
                os.path.join(root, name, f)
                for f in os.listdir(os.path.join(root, name))
                if f.endswith(".bin")
            )
            st = Stream(files, seq)
            # `heldout_windows` random windows of each stream are never trained on (a held-out set from the same data)
            perm = np.random.default_rng([seed, i]).permutation(st.n_windows)
            self.streams.append(st)
            self.perms.append(perm[heldout_windows:])
            self.heldout.append(np.sort(perm[:heldout_windows]))
        wsum = sum(weights.values())
        counts = np.zeros(len(self.names), dtype=np.int32)
        for i, name in enumerate(self.names):
            counts[i] = (
                weights[name] / wsum * block_size
            )  # truncation, as Levanter's int32 assignment
        counts[np.argmax(counts)] += block_size - counts.sum()
        self.counts = counts
        self.base = np.concatenate(
            [np.stack((np.full(c, i), np.arange(c)), 1) for i, c in enumerate(counts)]
        )
        self._block = (None, None)

    def describe(self):
        return {
            n: {
                "windows": st.n_windows,
                "tokens": int(st.cum[-1]),
                "per_block": int(c),
                "epochs_per_1B_train_tokens": 1e9 * c / self.block_size / (st.n_windows * st.seq),
            }
            for n, st, c in zip(self.names, self.streams, self.counts)
        }

    def _block_ids(self, b):
        if self._block[0] != b:
            perm = np.random.default_rng([self.seed, 1 << 20, b]).permutation(self.block_size)
            self._block = (b, self.base[perm])
        return self._block[1]

    def source(self, g):
        """(component, window) of global sequence g."""
        b, j = divmod(g, self.block_size)
        comp, k = self._block_ids(b)[j]
        idx = b * int(self.counts[comp]) + int(k)
        perm = self.perms[comp]
        return int(comp), int(perm[idx % perm.size])

    def sequence(self, g):
        comp, w = self.source(g)
        return self.streams[comp].window(w)

    def batch(self, start, n):
        return np.stack([self.sequence(g) for g in range(start, start + n)])


class EvalSet:
    """An evaluation set: a token stream cut into seq windows (the last partial window dropped), as Levanter's eval."""

    def __init__(self, files, seq=SEQ, windows=None):
        self.stream = Stream(files, seq)
        self.windows = np.arange(self.stream.n_windows) if windows is None else np.asarray(windows)

    def __len__(self):
        return len(self.windows)

    def batch(self, start, n):
        return np.stack([self.stream.window(int(w)) for w in self.windows[start : start + n]])


def eval_sets(root, mixture=None, include=None, seq=SEQ):
    """Paloma subsets found under root/paloma (each <subset>/*.bin) plus the held-out mixture windows."""
    out = {}
    pal = os.path.join(root, "paloma")
    if os.path.isdir(pal):
        for sub in sorted(os.listdir(pal)):
            files = sorted(
                os.path.join(pal, sub, f)
                for f in os.listdir(os.path.join(pal, sub))
                if f.endswith(".bin")
            )
            if files and (include is None or sub in include):
                es = EvalSet(files, seq=seq)
                if len(es):
                    out[f"paloma/{sub}"] = es
    if mixture is not None:
        for name, st, held in zip(mixture.names, mixture.streams, mixture.heldout):
            if held.size:
                es = EvalSet.__new__(EvalSet)
                es.stream, es.windows = st, held
                out[f"heldout/{name}"] = es
    return out


def pack(tokens, CAP, EOS=128001, pin_memory=True):
    """Pack EOS-separated attention; only each window's final target is masked."""
    import torch

    B, T = tokens.shape
    tok = tokens.astype(np.int64)
    tgt = np.empty_like(tok)
    tgt[:, :-1], tgt[:, -1] = tok[:, 1:], -100
    starts = np.zeros((B, T), dtype=bool)
    starts[:, 0] = True
    starts[:, 1:] = tok[:, :-1] == EOS
    idx = np.flatnonzero(starts.reshape(-1)).astype(np.int32)
    n = idx.size
    if n >= CAP:
        raise ValueError(f"{n} documents exceed attention capacity {CAP}")
    cu = np.full(CAP + 1, B * T, dtype=np.int32)
    cu[:n] = idx
    mx = int(np.diff(np.append(idx, B * T)).max())
    values = [torch.from_numpy(a) for a in (tok, tgt, cu)]
    if pin_memory:
        values = [a.pin_memory() for a in values]
    return (*values, torch.tensor([n, mx], dtype=torch.int64))


def validate_data_plan(root, plan_path):
    """Check shard names and token counts before constructing the deterministic mixture."""
    import json
    from pathlib import Path

    root = Path(root)
    with open(plan_path) as stream:
        plan = json.load(stream)
    required = set(WEIGHTS) | {"paloma/c4_en"}
    for component in required:
        records = [r for r in plan["files"] if r["component"] == component]
        if not records:
            raise ValueError(f"No data records for {component}")
        expected = {r["file"] for r in records}
        directory = root / component
        actual = {p.name for p in directory.glob("*.bin")}
        if actual != expected:
            raise ValueError(f"Shard names differ from data plan for {component}")
        for record in records:
            path = directory / record["file"]
            if path.stat().st_size != 4 * record["tokens"]:
                raise ValueError(f"Token count differs from data plan: {path}")
