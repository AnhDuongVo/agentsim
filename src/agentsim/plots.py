"""Plots for sweep results (needs matplotlib, ``pip install agentsim[plot]``)."""

from __future__ import annotations

from pathlib import Path


def plot_sweep(rows: list[dict], path: str | Path, x: str = "n_agents", series: str = "policy") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("runs_per_min", "Throughput (runs / min)"),
        ("latency_p95_s", "Run latency p95 (s)"),
        ("ttft_p95_s", "Time to first token p95 (s)"),
        ("usd_per_1k_runs", "Cost (USD per 1 000 runs)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    keys = sorted({r[series] for r in rows})
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for k in keys:
            pts = sorted(((r[x], r[metric]) for r in rows if r[series] == k), key=lambda p: p[0])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=str(k))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x.replace("_", " "))
        ax.grid(alpha=0.3)
    axes.flat[0].legend(title=series)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
