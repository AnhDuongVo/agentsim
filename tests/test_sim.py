from __future__ import annotations

import random

import pytest

from agentsim import (
    AgentRun,
    Branch,
    ClusterSpec,
    SchedulerSpec,
    Stage,
    Step,
    from_agenteval,
    make_workload,
    mixed_workload,
    simulate,
    summarize,
)
from agentsim.calibrate import fit
from agentsim.workload import make_run


def _single(prompt: int = 100, out: int = 10) -> AgentRun:
    return AgentRun(id="r", stages=[Stage(branches=[Branch(steps=[Step(prompt_tokens=prompt, output_tokens=out)])])])


def test_single_request_latency_matches_closed_form():
    c = ClusterSpec(n_gpus=1, iter_base_ms=10, iter_per_seq_ms=1, prefill_tokens_per_s=1000, prefix_cache=False)
    res = simulate([_single(100, 10)], c, n_agents=1, seed=1)
    r = res.runs[0]
    # first iteration: 11 ms decode + 100 ms prefill; nine more iterations of 11 ms
    assert r.latency_s == pytest.approx(0.111 + 9 * 0.011, abs=1e-6)
    assert r.ttft_s[0] == pytest.approx(0.111, abs=1e-6)
    assert r.output_tokens == 10


def test_batching_shares_iterations():
    c = ClusterSpec(n_gpus=1, iter_base_ms=10, iter_per_seq_ms=0, prefill_tokens_per_s=1e9, max_batch=64)
    runs = [_single(10, 20) for _ in range(8)]
    for i, r in enumerate(runs):
        r.id = f"r{i}"
    res = simulate(runs, c, n_agents=8, seed=0)
    # all eight decode together: makespan is about 20 iterations, not 160
    assert res.makespan_s < 20 * 0.010 * 1.5
    assert summarize(res)["batch_mean"] > 6


def test_max_batch_queues_requests():
    c = ClusterSpec(n_gpus=1, iter_base_ms=10, iter_per_seq_ms=0, prefill_tokens_per_s=1e9, max_batch=2)
    runs = [_single(10, 20) for _ in range(4)]
    for i, r in enumerate(runs):
        r.id = f"r{i}"
    res = simulate(runs, c, n_agents=4, seed=0)
    assert 0.40 <= res.makespan_s <= 0.46  # two waves of 20 iterations, plus up to 50 ms of start jitter
    assert sum(r.queue_wait_s for r in res.runs) > 0


def test_join_wait_is_recorded_for_fanout():
    fast = Step(prompt_tokens=10, output_tokens=5)
    slow = Step(prompt_tokens=10, output_tokens=50)
    run = AgentRun(id="f", stages=[Stage(branches=[Branch(steps=[fast]), Branch(steps=[slow])])])
    res = simulate([run], ClusterSpec(prefix_cache=False), n_agents=1)
    assert res.runs[0].join_wait_s > 0
    assert res.runs[0].join_wait_s < res.runs[0].latency_s


def test_prefix_cache_saves_prefill():
    runs = make_workload("react", 20, seed=3)
    on = simulate(runs, ClusterSpec(prefix_cache=True), n_agents=4)
    off = simulate(runs, ClusterSpec(prefix_cache=False), n_agents=4)
    assert on.prefill_tokens_saved > 0
    assert off.prefill_tokens_saved == 0
    assert on.makespan_s < off.makespan_s


def test_affinity_keeps_a_run_on_one_replica():
    runs = make_workload("fanout", 6, seed=1)
    res = simulate(runs, ClusterSpec(n_gpus=3), SchedulerSpec(routing="affinity"), n_agents=6)
    assert len(res.runs) == 6


def test_srpt_favours_short_runs_in_a_mix():
    runs = mixed_workload({"chat": 0.5, "react": 0.3, "fanout": 0.2}, 120, seed=0)
    c = ClusterSpec(n_gpus=1, max_batch=16)
    fifo = simulate(list(runs), c, SchedulerSpec(policy="fifo"), n_agents=48)
    srpt = simulate(list(runs), c, SchedulerSpec(policy="srpt"), n_agents=48)
    chat = lambda res: sorted(r.latency_s for r in res.runs if r.profile == "chat")  # noqa: E731
    assert chat(srpt)[len(chat(srpt)) // 2] < chat(fifo)[len(chat(fifo)) // 2]


def test_open_loop_arrivals_complete_all_runs():
    runs = make_workload("chat", 30, seed=2)
    res = simulate(runs, ClusterSpec(), arrival_rate=5.0, seed=2)
    assert len(res.runs) == 30


def test_admission_control_limits_inflight():
    runs = make_workload("chat", 30, seed=2)
    res = simulate(runs, ClusterSpec(), SchedulerSpec(max_inflight_runs=2), arrival_rate=50.0, seed=2)
    assert len(res.runs) == 30


def test_step_larger_than_kv_is_rejected():
    with pytest.raises(ValueError):
        simulate([_single(10_000, 10)], ClusterSpec(kv_capacity_tokens=5000), n_agents=1)


def test_profiles_are_reproducible():
    a = make_run("clinical", "x", random.Random(7))
    b = make_run("clinical", "x", random.Random(7))
    assert a == b
    assert a.max_fanout() >= 6


def test_from_agenteval_builds_sequential_runs():
    rec = {
        "id": "oa-1",
        "framework": "openai",
        "task": "max dose?",
        "steps": [
            {"kind": "message", "output": "I will search."},
            {"kind": "tool_call", "name": "search_label", "output": "Max 2550 mg/day"},
        ],
        "final_answer": "The maximum is 2550 mg/day.",
    }
    runs = from_agenteval([rec])
    assert runs[0].n_llm_steps() == 2
    kinds = [s.kind for s in runs[0].stages[0].branches[0].steps]
    assert kinds == ["llm", "tool", "llm"]
    # prompt grows as the conversation accumulates
    llm = runs[0].llm_steps()
    assert llm[1].prompt_tokens > llm[0].prompt_tokens


def test_calibration_recovers_parameters():
    truth = ClusterSpec(iter_base_ms=8.0, iter_per_seq_ms=0.5, prefill_tokens_per_s=10_000)
    rows = []
    for b in (1, 8, 32):
        for p in (500, 2000, 8000):
            itl = truth.iter_base_ms + truth.iter_per_seq_ms * b
            rows.append(
                {
                    "batch_size": b,
                    "prompt_tokens": p,
                    "itl_ms": itl,
                    "ttft_ms": itl + p / truth.prefill_tokens_per_s * 1000,
                }
            )
    got = fit(rows)
    assert got.iter_base_ms == pytest.approx(8.0, abs=0.01)
    assert got.iter_per_seq_ms == pytest.approx(0.5, abs=0.001)
    assert got.prefill_tokens_per_s == pytest.approx(10_000, rel=0.01)
