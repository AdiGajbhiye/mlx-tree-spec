"""Decode tok/s charts for the README and launch post -> docs/decode-chart.png, docs/decode-tasks-chart.png.

Usage: uv run --with matplotlib bench/runtimes/decode_chart.py
Medians per cell, as in summary-splash.md: T2 from results-splash3.jsonl and SP from
results-splash2.jsonl (the same night, 2026-10-29/30), Q4/Q5 (MTPLX) from results-base.jsonl.
"""
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "..", "docs")
SOURCES = {"Q4": "results-base.jsonl", "Q5": "results-base.jsonl", "SP": "results-splash2.jsonl", "T2": "results-splash3.jsonl"}
# Plain decoding is the reference, so it stays neutral; the three engines take validated categorical hues.
SERIES = [("Q4", "Plain decoding (MTPLX, no MTP)", "#b5b3aa"),
          ("Q5", "MTPLX, MTP depth 3", "#eb6834"),
          ("SP", "Splash 1.5, DFlash2", "#1baf7a"),
          ("T2", "mlx-tree-spec (this)", "#2a78d6")]
CTXS = [128, 1024, 4096, 8192, 16384, 32768]
TASKS = ["code", "prose", "math", "json", "edit", "creative"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"

cells = defaultdict(list)
for label, f in SOURCES.items():
    for line in open(os.path.join(HERE, f)):
        r = json.loads(line) if line.strip() else {}
        if r.get("label") == label and r.get("scenario") in ("s1", "s2") and r.get("decode_tps"):
            cells[(label, r["scenario"], r["task"], r["ctx"])].append(r["decode_tps"])


def med(label, sc, task, ctx):
    return statistics.median(cells[(label, sc, task, ctx)])


def chart(path, title, keys, ticks, xlabel, value):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n, width, gap = len(SERIES), 0.2, 0.02
    for i, (label, name, color) in enumerate(SERIES):
        xs = [k + (i - (n - 1) / 2) * width for k in range(len(keys))]
        ys = [value(label, k) for k in keys]
        ax.bar(xs, ys, width - gap, color=color, label=name, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.5, f"{y:.0f}", ha="center", va="bottom", zorder=4,
                    fontsize=13 if label == "T2" else 12, color=INK if label == "T2" else MUTED,
                    fontweight="bold" if label == "T2" else "normal")
    ax.set_xticks(range(len(keys)), ticks, fontsize=14, color=INK)
    ax.tick_params(axis="y", labelsize=14, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Decode tok/s", fontsize=15, color=MUTED)
    ax.set_xlabel(xlabel, fontsize=15, color=MUTED, labelpad=12)
    ax.grid(axis="y", color=GRID, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=4, frameon=False, fontsize=14,
              handlelength=1.1, handleheight=1.1, labelcolor=INK)
    fig.text(0.08, 0.95, title, fontsize=24, fontweight="bold", color=INK)
    fig.text(0.08, 0.905, "Stock Qwen3.8-27B 4-bit · M5 Max 36 GB · one request · greedy · thinking off · "
             "512 new tokens · median of 3 runs", fontsize=14, color=MUTED)
    fig.text(0.08, 0.02, "github.com/AdiGajbhiye/mlx-tree-spec · bench/runtimes/summary-splash.md · "
             "Splash runs the unsloth UD-Q4_K_M GGUF, the others the MLX 4-bit build", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.13)
    fig.savefig(path, facecolor=SURFACE)
    print("wrote", os.path.relpath(path))


chart(os.path.join(DOCS, "decode-chart.png"), "Qwen3.8-27B decode speed by prompt length", CTXS,
      ["128", "1K", "4K", "8K", "16K", "32K"], "Prompt context (tokens)",
      lambda L, k: med(L, "s1", "longctx", k))
chart(os.path.join(DOCS, "decode-tasks-chart.png"), "Qwen3.8-27B decode speed by task", TASKS,
      TASKS, "Task (short prompt; edit rewrites a 250-line file)",
      lambda L, k: med(L, "s2", k, 0))
