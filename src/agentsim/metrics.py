"""Summary metrics for a simulation result."""

from __future__ import annotations

from .schema import SimResult


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(res: SimResult) -> dict[str, float]:
    runs = res.runs
    lat = [r.latency_s for r in runs]
    ttft = [t for r in runs for t in r.ttft_s]
    gpu_hours = res.makespan_s * res.cluster.n_gpus / 3600.0
    out_tokens = sum(r.output_tokens for r in runs)
    n = len(runs)
    return {
        "runs": n,
        "n_gpus": res.cluster.n_gpus,
        "n_agents": res.n_agents,
        "makespan_s": round(res.makespan_s, 2),
        "runs_per_min": round(n / res.makespan_s * 60, 2) if res.makespan_s else 0.0,
        "output_tok_per_s": round(out_tokens / res.makespan_s, 1) if res.makespan_s else 0.0,
        "latency_p50_s": round(percentile(lat, 0.5), 2),
        "latency_p95_s": round(percentile(lat, 0.95), 2),
        "ttft_p50_s": round(percentile(ttft, 0.5), 3),
        "ttft_p95_s": round(percentile(ttft, 0.95), 3),
        "queue_wait_mean_s": round(sum(r.queue_wait_s for r in runs) / n, 2) if n else 0.0,
        "join_wait_mean_s": round(sum(r.join_wait_s for r in runs) / n, 2) if n else 0.0,
        "join_wait_share": round(sum(r.join_wait_s for r in runs) / sum(lat), 3) if sum(lat) else 0.0,
        "gpu_util": round(sum(res.gpu_busy_s) / (res.makespan_s * res.cluster.n_gpus), 3) if res.makespan_s else 0.0,
        "batch_mean": round(sum(res.batch_mean) / len(res.batch_mean), 1) if res.batch_mean else 0.0,
        "kv_occupancy": round(sum(res.kv_occupancy_mean) / len(res.kv_occupancy_mean), 3),
        "prefill_saved_share": round(res.prefill_tokens_saved / res.prefill_tokens_total, 3)
        if res.prefill_tokens_total
        else 0.0,
        "usd_per_1k_runs": round(gpu_hours * res.cluster.gpu_hour_usd / n * 1000, 2) if n else 0.0,
    }


def per_profile(res: SimResult) -> dict[str, dict[str, float]]:
    groups: dict[str, list] = {}
    for r in res.runs:
        groups.setdefault(r.profile, []).append(r)
    return {
        name: {
            "runs": len(rs),
            "latency_p50_s": round(percentile([r.latency_s for r in rs], 0.5), 2),
            "latency_p95_s": round(percentile([r.latency_s for r in rs], 0.95), 2),
            "join_wait_mean_s": round(sum(r.join_wait_s for r in rs) / len(rs), 2),
        }
        for name, rs in groups.items()
    }
