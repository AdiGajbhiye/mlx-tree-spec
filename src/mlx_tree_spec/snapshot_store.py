"""Disk tier for the server's prompt-prefix snapshots.

The server keeps up to BANK snapshots of its caches in memory (server.Engine.prefill).
SnapshotStore also writes each one to disk, so a prompt can resume from a prefix seen
before a restart, or one that has since left the in-memory bank: loading a 16K-token
snapshot takes about a second, against ~22 s to prefill it again.

- One safetensors file per snapshot, `<tokens>-<hash>.safetensors`, holding the prefix
  tokens, every target and MTP cache layer's state (bf16, exactly as prefill left it)
  and the last target hidden state. Files sit in a directory per setup key (checkpoint,
  this package's source, library versions, GPU, macOS), so a snapshot is only ever
  resumed by the setup that wrote it.
- Writes are queued and done by the engine thread while no request is active (flush),
  so a running request never waits for the disk. A file is written to a temporary name
  and renamed, so a crash leaves no partial snapshot.
- The newest-used files are kept up to `budget_bytes` across all setup directories;
  the rest are deleted, oldest use (mtime) first. A load touches its file.

Snapshots hold the prompt tokens, i.e. the prompt text, in plain form: the directory
is created readable by the owner only.
"""

from __future__ import annotations

import hashlib
import os
import time
from array import array
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

SUFFIX = ".safetensors"
TMP = "tmp-"  # files being written: <TMP><name>, renamed when complete


def prefix_hash(tokens) -> str:
    return hashlib.sha256(array("i", tokens).tobytes()).hexdigest()[:20]


def to_arrays(snap) -> tuple[dict[str, mx.array], dict[str, str]]:
    """(tokens, target cache, MTP cache, last hidden) -> safetensors arrays and metadata."""
    tokens, cache, mtp_cache, prev_h = snap
    arrays = {"tokens": mx.array(tokens, dtype=mx.int32), "prev_h": prev_h}
    kinds = []
    for prefix, layers in (("t", cache), ("m", mtp_cache)):
        for i, c in enumerate(layers):
            if isinstance(c, KVCache):
                kinds.append("K" if c.keys is not None else "k")
                if c.keys is not None:
                    arrays[f"{prefix}{i}.k"], arrays[f"{prefix}{i}.v"] = c.state
            elif isinstance(c, ArraysCache):
                kinds.append(f"A{len(c.cache)}")
                for j, x in enumerate(c.cache):
                    if x is not None:
                        arrays[f"{prefix}{i}.{j}"] = x
            else:
                raise TypeError(f"can't store cache type {type(c).__name__}")
        kinds.append("|")
    return arrays, {"kinds": ",".join(kinds)}


def from_arrays(arrays: dict[str, mx.array], meta: dict[str, str], cache, mtp_cache):
    """Fill fresh `cache` / `mtp_cache` (from make_cache) from to_arrays' output;
    returns the snapshot tuple. Raises ValueError if the layout doesn't match."""
    kinds = meta["kinds"].split(",")
    groups, cur = [], []
    for k in kinds:
        if k == "|":
            groups.append(cur)
            cur = []
        else:
            cur.append(k)
    if len(groups) != 2 or [len(g) for g in groups] != [len(cache), len(mtp_cache)]:
        raise ValueError("snapshot layout doesn't match the model's caches")
    for prefix, layers, ks in (("t", cache, groups[0]), ("m", mtp_cache, groups[1])):
        for i, (c, k) in enumerate(zip(layers, ks)):
            if k in ("K", "k") and isinstance(c, KVCache):
                if k == "K":
                    c.state = (arrays[f"{prefix}{i}.k"], arrays[f"{prefix}{i}.v"])
            elif k.startswith("A") and isinstance(c, ArraysCache) and int(k[1:]) == len(c.cache):
                c.cache = [arrays.get(f"{prefix}{i}.{j}") for j in range(len(c.cache))]
            else:
                raise ValueError(f"snapshot layer {prefix}{i} is {k}, the model's is {type(c).__name__}")
    return arrays["tokens"].tolist(), cache, mtp_cache, arrays["prev_h"]


class SnapshotStore:
    def __init__(self, root: Path, key: str, budget_bytes: int):
        self.root, self.budget = Path(root), budget_bytes
        self.dir = self.root / key
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.entries: dict[tuple[int, str], Path] = {}  # (tokens, prefix hash) -> file
        for f in self.dir.glob("*" + SUFFIX):
            n, _, h = f.stem.partition("-")
            if n.isdigit() and h:
                self.entries[(int(n), h)] = f
        for f in self.dir.glob(TMP + "*"):
            f.unlink(missing_ok=True)
        self.pending: list[tuple] = []

    def find(self, prompt: list[int], longer_than: int = 0) -> tuple[int, Path] | None:
        """The longest stored prefix of `prompt` (shorter than the prompt, longer than
        `longer_than`), as (tokens, file)."""
        best = None
        for (n, h), f in self.entries.items():
            if longer_than < n < len(prompt) and (best is None or n > best[0]) and prefix_hash(prompt[:n]) == h:
                best = (n, f)
        return best

    def load(self, f: Path, prompt: list[int], cache, mtp_cache):
        """The snapshot in `f` filled into fresh caches, or None if it is unreadable
        or isn't a prefix of `prompt` (the file is then dropped)."""
        try:
            arrays, meta = mx.load(str(f), return_metadata=True)
            snap = from_arrays(arrays, meta, cache, mtp_cache)
            if prompt[: len(snap[0])] != snap[0]:
                raise ValueError("hash collision")
            mx.eval(list(arrays.values()))
        except (OSError, ValueError, KeyError, RuntimeError):
            self._drop(f)
            return None
        os.utime(f)
        return snap

    def queue(self, snap) -> None:
        key = (len(snap[0]), prefix_hash(snap[0]))
        if key not in self.entries and all((len(p[0]), prefix_hash(p[0])) != key for p in self.pending):
            self.pending.append(snap)

    def flush(self) -> dict | None:
        """Write the oldest queued snapshot, then trim to the budget; returns what was
        written (tokens, MB, seconds), or None if the write failed."""
        t0 = time.perf_counter()
        snap = self.pending.pop(0)
        n, h = len(snap[0]), prefix_hash(snap[0])
        f = self.dir / f"{n}-{h}{SUFFIX}"
        tmp = self.dir / (TMP + f.name)
        arrays, meta = to_arrays(snap)
        try:
            mx.save_safetensors(str(tmp), arrays, meta)
            tmp.replace(f)
        except (OSError, RuntimeError):
            tmp.unlink(missing_ok=True)
            return None
        self.entries[(n, h)] = f
        self.trim()
        return {"tokens": n, "mb": round(f.stat().st_size / 2**20), "s": round(time.perf_counter() - t0, 2)}

    def trim(self) -> None:
        files = [f for f in self.root.glob("*/*" + SUFFIX) if not f.name.startswith(TMP)]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for f in files:
            total += f.stat().st_size
            if total > self.budget:
                self._drop(f)

    def _drop(self, f: Path) -> None:
        f.unlink(missing_ok=True)
        self.entries = {k: v for k, v in self.entries.items() if v != f}
