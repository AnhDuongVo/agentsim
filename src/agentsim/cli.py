"""Command line: ``agentsim run``, ``agentsim sweep``, ``agentsim workload``, ``agentsim bench``, ``agentsim calibrate``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .calibrate import fit, load_rows
from .metrics import per_profile, summarize
from .schema import ClusterSpec, SchedulerSpec
from .sim import simulate
from .workload import PROFILES, load_runs, make_workload, mixed_workload, save_runs


def _cluster_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cluster", help="JSON file with ClusterSpec fields (for example from `agentsim calibrate`)")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--kv", type=int, help="KV capacity per replica in tokens")
    p.add_argument("--max-batch", type=int)


def _workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile",
        default="react",
        help=f"one of {', '.join(PROFILES)}, a mix like chat:0.6,react:0.3,fanout:0.1, or a JSONL file",
    )
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)


def _build_cluster(a: argparse.Namespace) -> ClusterSpec:
    spec = ClusterSpec.model_validate(json.loads(Path(a.cluster).read_text())) if a.cluster else ClusterSpec()
    upd: dict = {"n_gpus": a.gpus}
    if a.no_prefix_cache:
        upd["prefix_cache"] = False
    if a.kv:
        upd["kv_capacity_tokens"] = a.kv
    if a.max_batch:
        upd["max_batch"] = a.max_batch
    return spec.model_copy(update=upd)


def _build_runs(a: argparse.Namespace):
    if Path(a.profile).suffix == ".jsonl" and Path(a.profile).exists():
        return load_runs(a.profile)
    if ":" in a.profile:
        mix = {k: float(v) for k, v in (part.split(":") for part in a.profile.split(","))}
        return mixed_workload(mix, a.runs, a.seed)
    return make_workload(a.profile, a.runs, a.seed)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))


def cmd_run(a: argparse.Namespace) -> None:
    res = simulate(
        _build_runs(a),
        _build_cluster(a),
        SchedulerSpec(policy=a.policy, routing=a.routing, max_inflight_runs=a.max_inflight),
        n_agents=a.agents,
        arrival_rate=a.rate,
        seed=a.seed,
    )
    s = summarize(res)
    for k, v in s.items():
        print(f"{k:22s} {v}")
    pp = per_profile(res)
    if len(pp) > 1:
        print()
        _print_table([{"profile": k, **v} for k, v in pp.items()])
    if a.json:
        Path(a.json).write_text(
            json.dumps({"summary": s, "per_profile": pp, "runs": [r.model_dump() for r in res.runs]}, indent=1)
        )


def cmd_sweep(a: argparse.Namespace) -> None:
    runs = _build_runs(a)
    cluster = _build_cluster(a)
    agents = [int(x) for x in a.agents.split(",")]
    policies = a.policy.split(",")
    rows: list[dict] = []
    for pol in policies:
        for n in agents:
            res = simulate(
                list(runs),
                cluster,
                SchedulerSpec(policy=pol, routing=a.routing, max_inflight_runs=a.max_inflight),
                n_agents=n,
                seed=a.seed,
            )
            rows.append({"policy": pol, **summarize(res)})
            print(
                f"{pol:9s} agents={n:4d}  runs/min={rows[-1]['runs_per_min']:7.2f}  p95={rows[-1]['latency_p95_s']:7.2f}s  join={rows[-1]['join_wait_share']:.2f}  util={rows[-1]['gpu_util']:.2f}",
                file=sys.stderr,
            )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")
    if a.plot:
        from .plots import plot_sweep

        plot_sweep(rows, a.plot)
        print(f"wrote {a.plot}")


def cmd_workload(a: argparse.Namespace) -> None:
    runs = _build_runs(a)
    save_runs(runs, a.out)
    n_llm = sum(r.n_llm_steps() for r in runs)
    print(f"wrote {len(runs)} runs, {n_llm} LLM steps, max fan-out {max(r.max_fanout() for r in runs)} to {a.out}")


def cmd_bench(a: argparse.Namespace) -> None:
    from .bench import measure

    rows = measure(
        a.url,
        a.model,
        [int(x) for x in a.concurrency.split(",")],
        [int(x) for x in a.prompt_tokens.split(",")],
        a.max_tokens,
    )
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {a.out}")


def cmd_calibrate(a: argparse.Namespace) -> None:
    spec = fit(load_rows(a.measurement))
    Path(a.out).write_text(spec.model_dump_json(indent=1))
    print(spec.model_dump_json(indent=1))
    print(f"wrote {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agentsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="simulate one configuration and print the summary")
    _workload_args(p)
    _cluster_args(p)
    p.add_argument("--agents", type=int, default=16, help="closed loop: concurrent agents (0 to use --rate)")
    p.add_argument("--rate", type=float, default=0.0, help="open loop: run arrivals per second")
    p.add_argument("--policy", default="fifo", choices=["fifo", "srpt", "progress"])
    p.add_argument("--routing", default="least_kv", choices=["least_kv", "affinity", "round_robin"])
    p.add_argument("--max-inflight", type=int, default=0)
    p.add_argument("--json", help="write summary and per-run results here")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="sweep concurrent agents for one or more policies, write CSV and a plot")
    _workload_args(p)
    _cluster_args(p)
    p.add_argument("--agents", default="4,8,16,32,64,128")
    p.add_argument("--policy", default="fifo,srpt,progress", help="comma-separated")
    p.add_argument("--routing", default="least_kv", choices=["least_kv", "affinity", "round_robin"])
    p.add_argument("--max-inflight", type=int, default=0)
    p.add_argument("--out", default="results/sweep.csv")
    p.add_argument("--plot", help="PNG path (needs matplotlib)")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("workload", help="generate a workload file to inspect or edit")
    _workload_args(p)
    p.add_argument("--out", default="workload.jsonl")
    p.set_defaults(func=cmd_workload)

    p = sub.add_parser("bench", help="measure an OpenAI-compatible endpoint for calibration")
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", default="1,4,16,64")
    p.add_argument("--prompt-tokens", default="500,2000,8000")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out", default="measurement.jsonl")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("calibrate", help="fit ClusterSpec parameters to a measurement file")
    p.add_argument("measurement")
    p.add_argument("--out", default="cluster.json")
    p.set_defaults(func=cmd_calibrate)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
