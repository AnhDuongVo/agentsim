"""Workload and cluster models.

An agent run is a sequence of stages. Each stage holds one or more branches that run in parallel
(a fan-out); the next stage starts only when every branch of the current one has finished (a join).
A branch is a sequential list of steps, each either an LLM call or a tool call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Step(BaseModel):
    kind: str = "llm"  # "llm" | "tool"
    name: str = ""
    prompt_tokens: int = 0  # llm: full prompt length, including any shared prefix
    output_tokens: int = 0  # llm: tokens to decode
    prefix_id: str = ""  # llm: id of a cacheable prefix (system prompt, shared context), "" for none
    prefix_tokens: int = 0  # llm: how many of prompt_tokens belong to that prefix
    duration_s: float = 0.0  # tool: wall time the tool takes (no GPU involved)


class Branch(BaseModel):
    steps: list[Step] = Field(default_factory=list)


class Stage(BaseModel):
    branches: list[Branch] = Field(default_factory=list)


class AgentRun(BaseModel):
    id: str
    profile: str = ""
    stages: list[Stage] = Field(default_factory=list)

    def llm_steps(self) -> list[Step]:
        return [s for st in self.stages for b in st.branches for s in b.steps if s.kind == "llm"]

    def n_llm_steps(self) -> int:
        return len(self.llm_steps())

    def max_fanout(self) -> int:
        return max((len(st.branches) for st in self.stages), default=0)


class ClusterSpec(BaseModel):
    """A homogeneous pool of GPUs, each running one model replica with continuous batching.

    The decode model is the usual linear approximation: one decode iteration takes
    ``iter_base_ms + iter_per_seq_ms * batch_size`` and produces one token for every running sequence.
    Prefill for a newly admitted sequence costs ``prompt_tokens / prefill_tokens_per_s`` and is added to
    the iteration in which the sequence is admitted (chunked prefill is not modelled separately).
    """

    n_gpus: int = 1
    max_batch: int = 64  # max running sequences per replica
    kv_capacity_tokens: int = 200_000  # KV-cache capacity per replica, in tokens
    iter_base_ms: float = 12.0
    iter_per_seq_ms: float = 0.35
    prefill_tokens_per_s: float = 12_000.0
    prefix_cache: bool = True  # reuse a prefix already seen on the same replica
    prefix_cache_slots: int = 32  # how many distinct prefixes a replica keeps (LRU)
    gpu_hour_usd: float = 2.5

    def iteration_s(self, batch_size: int, prefill_tokens: int) -> float:
        decode = (self.iter_base_ms + self.iter_per_seq_ms * batch_size) / 1000.0
        prefill = prefill_tokens / self.prefill_tokens_per_s
        return decode + prefill


class SchedulerSpec(BaseModel):
    """What the scheduler decides: which waiting request a replica admits next, and which replica a run goes to.

    policy:
      fifo      first come, first served
      srpt      shortest remaining processing time: fewest LLM steps left in the run first
      progress  runs that have already started before runs that have not (keeps fewer runs in flight)
    routing:
      least_kv  the replica with the most free KV cache
      affinity  all steps of a run go to the same replica (prefix-cache locality)
      round_robin
    max_inflight_runs: admission control, 0 for unlimited
    """

    policy: str = "fifo"
    routing: str = "least_kv"
    max_inflight_runs: int = 0


class RunResult(BaseModel):
    run_id: str
    profile: str
    start_s: float
    end_s: float
    n_llm_steps: int
    output_tokens: int
    join_wait_s: float = 0.0  # time branches spent finished but waiting for siblings at joins
    queue_wait_s: float = 0.0  # time LLM steps spent waiting for admission
    ttft_s: list[float] = Field(default_factory=list)

    @property
    def latency_s(self) -> float:
        return self.end_s - self.start_s


class SimResult(BaseModel):
    cluster: ClusterSpec
    scheduler: SchedulerSpec
    n_agents: int
    runs: list[RunResult]
    makespan_s: float
    gpu_busy_s: list[float]
    kv_occupancy_mean: list[float]  # per replica, fraction of capacity, time-weighted
    prefill_tokens_total: int
    prefill_tokens_saved: int  # by the prefix cache
    batch_mean: list[float] = Field(default_factory=list)  # per replica, mean running batch size while busy
