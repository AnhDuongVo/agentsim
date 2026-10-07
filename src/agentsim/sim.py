"""Discrete-event simulation of agent runs on a cluster of LLM replicas with continuous batching.

The event loop is a plain heap; there are three event kinds: a replica finishes a decode iteration,
a tool call finishes, and a new run starts (open-loop arrivals or closed-loop agents picking their next run).
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass, field

from .schema import AgentRun, ClusterSpec, RunResult, SchedulerSpec, SimResult, Step


@dataclass
class _Request:
    run: _RunState
    branch: _BranchState
    step: Step
    arrival: float
    remaining: int
    kv_tokens: int
    admitted: float | None = None


@dataclass
class _BranchState:
    steps: list[Step]
    cursor: int = 0
    done_at: float | None = None


@dataclass
class _RunState:
    run: AgentRun
    start: float
    stage: int = 0
    branches: list[_BranchState] = field(default_factory=list)
    llm_done: int = 0
    replica: int | None = None
    join_wait: float = 0.0
    queue_wait: float = 0.0
    ttft: list[float] = field(default_factory=list)
    output_tokens: int = 0

    @property
    def remaining_llm(self) -> int:
        return self.run.n_llm_steps() - self.llm_done


@dataclass
class _Replica:
    idx: int
    running: list[_Request] = field(default_factory=list)
    waiting: list[_Request] = field(default_factory=list)
    kv_used: int = 0
    busy_s: float = 0.0
    kv_time: float = 0.0  # integral of kv_used over time
    batch_time: float = 0.0  # integral of running batch size over busy time
    prefixes: OrderedDict[str, None] = field(default_factory=OrderedDict)
    in_iteration: bool = False


class Simulator:
    def __init__(
        self,
        runs: list[AgentRun],
        cluster: ClusterSpec,
        scheduler: SchedulerSpec | None = None,
        n_agents: int = 0,
        arrival_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        """``n_agents`` > 0 runs a closed loop: that many agents, each starting its next run when the previous one
        ends, until ``runs`` are used up. Otherwise runs arrive as a Poisson process at ``arrival_rate`` per second."""
        if not runs:
            raise ValueError("no runs to simulate")
        if n_agents <= 0 and arrival_rate <= 0:
            raise ValueError("set n_agents (closed loop) or arrival_rate (open loop)")
        self.pool = list(runs)
        self.cluster = cluster
        biggest = max((s.prompt_tokens + s.output_tokens for r in runs for s in r.llm_steps()), default=0)
        if biggest > cluster.kv_capacity_tokens:
            raise ValueError(f"a step needs {biggest} KV tokens but a replica holds {cluster.kv_capacity_tokens}")
        self.sched = scheduler or SchedulerSpec()
        self.n_agents = n_agents
        self.arrival_rate = arrival_rate
        self.rng = random.Random(seed)
        self.replicas = [_Replica(i) for i in range(cluster.n_gpus)]
        self.events: list[tuple[float, int, str, object]] = []
        self.counter = itertools.count()
        self.now = 0.0
        self.last_kv_t = [0.0] * cluster.n_gpus
        self.results: list[RunResult] = []
        self.inflight = 0
        self.pending_start: list[AgentRun] = []  # waiting on admission control
        self.rr = 0
        self.prefill_total = 0
        self.prefill_saved = 0

    # ------------------------------------------------------------------ events

    def _push(self, t: float, kind: str, payload: object = None) -> None:
        heapq.heappush(self.events, (t, next(self.counter), kind, payload))

    def run(self) -> SimResult:
        if self.n_agents > 0:
            for _ in range(min(self.n_agents, len(self.pool))):
                self._push(self.rng.uniform(0, 0.05), "start", None)
        else:
            self._push(self.rng.expovariate(self.arrival_rate), "arrival", None)

        while self.events:
            t, _, kind, payload = heapq.heappop(self.events)
            self.now = t
            if kind == "start":
                self._start_next()
            elif kind == "arrival":
                self._start_next()
                if self.pool or self.pending_start:
                    self._push(t + self.rng.expovariate(self.arrival_rate), "arrival", None)
            elif kind == "tool_done":
                run, branch = payload  # type: ignore[misc]
                self._advance(run, branch)
            elif kind == "iter_end":
                self._finish_iteration(payload)  # type: ignore[arg-type]

        for r in self.replicas:
            self._account_kv(r)
        return SimResult(
            cluster=self.cluster,
            scheduler=self.sched,
            n_agents=self.n_agents,
            runs=self.results,
            makespan_s=self.now,
            gpu_busy_s=[r.busy_s for r in self.replicas],
            kv_occupancy_mean=[
                (r.kv_time / self.now / self.cluster.kv_capacity_tokens) if self.now else 0.0 for r in self.replicas
            ],
            prefill_tokens_total=self.prefill_total,
            prefill_tokens_saved=self.prefill_saved,
            batch_mean=[(r.batch_time / r.busy_s) if r.busy_s else 0.0 for r in self.replicas],
        )

    # ------------------------------------------------------------------ runs

    def _start_next(self) -> None:
        if not self.pool:
            return
        if self.sched.max_inflight_runs and self.inflight >= self.sched.max_inflight_runs:
            self.pending_start.append(self.pool.pop(0))
            return
        self._start_run(self.pool.pop(0))

    def _start_run(self, run: AgentRun) -> None:
        self.inflight += 1
        state = _RunState(run=run, start=self.now)
        self._start_stage(state)

    def _start_stage(self, state: _RunState) -> None:
        stage = state.run.stages[state.stage]
        state.branches = [_BranchState(steps=list(b.steps)) for b in stage.branches]
        for b in state.branches:
            self._advance(state, b)

    def _advance(self, state: _RunState, branch: _BranchState) -> None:
        if branch.cursor >= len(branch.steps):
            branch.done_at = self.now
            if all(b.done_at is not None for b in state.branches):
                self._join(state)
            return
        step = branch.steps[branch.cursor]
        branch.cursor += 1
        if step.kind == "tool":
            self._push(self.now + step.duration_s, "tool_done", (state, branch))
        else:
            req = _Request(
                run=state,
                branch=branch,
                step=step,
                arrival=self.now,
                remaining=max(1, step.output_tokens),
                kv_tokens=step.prompt_tokens + step.output_tokens,
            )
            replica = self._route(state, req)
            replica.waiting.append(req)
            self._maybe_iterate(replica)

    def _join(self, state: _RunState) -> None:
        # mean time a branch of this stage sat finished while waiting for its siblings
        state.join_wait += sum(self.now - b.done_at for b in state.branches) / len(state.branches)  # type: ignore[operator]
        state.stage += 1
        if state.stage < len(state.run.stages):
            self._start_stage(state)
            return
        self.inflight -= 1
        self.results.append(
            RunResult(
                run_id=state.run.id,
                profile=state.run.profile,
                start_s=state.start,
                end_s=self.now,
                n_llm_steps=state.run.n_llm_steps(),
                output_tokens=state.output_tokens,
                join_wait_s=state.join_wait,
                queue_wait_s=state.queue_wait,
                ttft_s=state.ttft,
            )
        )
        if self.pending_start:
            self._start_run(self.pending_start.pop(0))
        elif self.n_agents > 0:
            self._push(self.now, "start", None)

    # ------------------------------------------------------------------ scheduling

    def _route(self, state: _RunState, req: _Request) -> _Replica:
        if self.sched.routing == "affinity":
            if state.replica is None:
                state.replica = self._least_loaded().idx
            return self.replicas[state.replica]
        if self.sched.routing == "round_robin":
            r = self.replicas[self.rr % len(self.replicas)]
            self.rr += 1
            return r
        return self._least_loaded()

    def _least_loaded(self) -> _Replica:
        return min(self.replicas, key=lambda r: r.kv_used + sum(q.kv_tokens for q in r.waiting))

    def _order(self, waiting: list[_Request]) -> list[_Request]:
        if self.sched.policy == "srpt":
            return sorted(waiting, key=lambda q: (q.run.remaining_llm, q.arrival))
        if self.sched.policy == "progress":
            return sorted(waiting, key=lambda q: (q.run.llm_done == 0, q.arrival))
        return sorted(waiting, key=lambda q: q.arrival)

    # ------------------------------------------------------------------ replica iteration loop

    def _account_kv(self, r: _Replica) -> None:
        r.kv_time += r.kv_used * (self.now - self.last_kv_t[r.idx])
        self.last_kv_t[r.idx] = self.now

    def _maybe_iterate(self, r: _Replica) -> None:
        if r.in_iteration:
            return
        self._account_kv(r)
        prefill = 0
        if r.waiting:
            admitted: list[_Request] = []
            for q in self._order(r.waiting):
                if len(r.running) + len(admitted) >= self.cluster.max_batch:
                    break
                if r.kv_used + q.kv_tokens > self.cluster.kv_capacity_tokens:
                    continue
                admitted.append(q)
                r.kv_used += q.kv_tokens
                q.admitted = self.now
                q.run.queue_wait += self.now - q.arrival
                prefill += self._prefill_tokens(r, q.step)
            for q in admitted:
                r.waiting.remove(q)
            r.running.extend(admitted)
        if not r.running:
            return
        dur = self.cluster.iteration_s(len(r.running), prefill)
        r.in_iteration = True
        r.busy_s += dur
        r.batch_time += dur * len(r.running)
        self._push(self.now + dur, "iter_end", r)

    def _prefill_tokens(self, r: _Replica, step: Step) -> int:
        tokens = step.prompt_tokens
        self.prefill_total += tokens
        if self.cluster.prefix_cache and step.prefix_id:
            if step.prefix_id in r.prefixes:
                r.prefixes.move_to_end(step.prefix_id)
                tokens -= step.prefix_tokens
                self.prefill_saved += step.prefix_tokens
            else:
                r.prefixes[step.prefix_id] = None
                while len(r.prefixes) > self.cluster.prefix_cache_slots:
                    r.prefixes.popitem(last=False)
        return max(0, tokens)

    def _finish_iteration(self, r: _Replica) -> None:
        self._account_kv(r)
        r.in_iteration = False
        finished: list[_Request] = []
        for q in r.running:
            if q.remaining == max(1, q.step.output_tokens):  # first token of this request
                q.run.ttft.append(self.now - q.arrival)
            q.remaining -= 1
            q.run.output_tokens += 1
            if q.remaining == 0:
                finished.append(q)
        for q in finished:
            r.running.remove(q)
            r.kv_used -= q.kv_tokens
            q.run.llm_done += 1
            self._advance(q.run, q.branch)
        self._maybe_iterate(r)


def simulate(
    runs: list[AgentRun],
    cluster: ClusterSpec,
    scheduler: SchedulerSpec | None = None,
    n_agents: int = 0,
    arrival_rate: float = 0.0,
    seed: int = 0,
) -> SimResult:
    return Simulator(runs, cluster, scheduler, n_agents=n_agents, arrival_rate=arrival_rate, seed=seed).run()
