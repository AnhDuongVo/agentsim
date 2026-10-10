"""Fit the cluster model to measurements of a real endpoint.

A measurement is a list of rows ``{"batch_size", "prompt_tokens", "ttft_ms", "itl_ms"}`` as produced by
``agentsim bench`` (or by any other load tool, converted to this shape). Two fits:

* inter-token latency against batch size gives ``iter_base_ms`` (intercept) and ``iter_per_seq_ms`` (slope);
* time to first token against prompt length, at the smallest batch size, gives ``prefill_tokens_per_s``
  (one over the slope, after subtracting one decode iteration).
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ClusterSpec


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares line y = a + b x. With a single distinct x the slope is 0."""
    n = len(xs)
    if n == 0:
        raise ValueError("no points to fit")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return my - b * mx, b


def fit(rows: list[dict], base: ClusterSpec | None = None) -> ClusterSpec:
    rows = [
        r
        for r in rows
        if r.get("n", 1) > 0
        and r.get("itl_ms") is not None
        and r.get("token_count_source", "server_usage") == "server_usage"
    ]
    if not rows:
        raise ValueError(
            "No measurements with server-reported token usage; estimated counts cannot calibrate token latency"
        )
    base = base or ClusterSpec()
    a, b = _linfit([float(r["batch_size"]) for r in rows], [float(r["itl_ms"]) for r in rows])
    iter_base_ms = max(0.5, a)
    iter_per_seq_ms = max(0.0, b)

    bmin = min(int(r["batch_size"]) for r in rows)
    small = [
        r
        for r in rows
        if int(r["batch_size"]) == bmin and r.get("prompt_count_source", "server_usage") == "server_usage"
    ]
    one_iter_ms = iter_base_ms + iter_per_seq_ms * bmin
    slope_ms_per_tok = 0.0
    if small:
        _, slope_ms_per_tok = _linfit(
            [float(r["prompt_tokens"]) for r in small],
            [max(0.0, float(r["ttft_ms"]) - one_iter_ms) for r in small],
        )
    prefill = 1000.0 / slope_ms_per_tok if slope_ms_per_tok > 0 else base.prefill_tokens_per_s
    return base.model_copy(
        update={
            "iter_base_ms": round(iter_base_ms, 3),
            "iter_per_seq_ms": round(iter_per_seq_ms, 4),
            "prefill_tokens_per_s": round(prefill, 1),
        }
    )


def load_rows(path: str | Path) -> list[dict]:
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]
