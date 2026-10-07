"""Workload generators: synthetic agent profiles, and conversion of agenteval traces.

Token counts in the synthetic profiles are drawn from log-normal distributions around typical values seen
in the clinical agents of this portfolio (consult-to-note, trial-matcher, csr-assistant, ai-scientist).
They are meant to be plausible, not measured; use ``from_agenteval`` for real traces.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from .schema import AgentRun, Branch, Stage, Step

PROFILES = ("chat", "react", "fanout", "clinical")


def _ln(rng: random.Random, mean: float, sigma: float = 0.35, lo: int = 1) -> int:
    """Log-normal integer around ``mean``."""
    return max(lo, int(rng.lognormvariate(_mu(mean, sigma), sigma)))


def _mu(mean: float, sigma: float) -> float:
    import math

    return math.log(mean) - sigma * sigma / 2


def _llm(rng: random.Random, name: str, prompt: float, out: float, prefix_id: str = "", prefix: int = 0) -> Step:
    p = _ln(rng, prompt) + prefix
    return Step(
        kind="llm", name=name, prompt_tokens=p, output_tokens=_ln(rng, out), prefix_id=prefix_id, prefix_tokens=prefix
    )


def _tool(rng: random.Random, name: str, mean_s: float) -> Step:
    return Step(kind="tool", name=name, duration_s=max(0.01, rng.lognormvariate(_mu(mean_s, 0.5), 0.5)))


def make_run(profile: str, run_id: str, rng: random.Random) -> AgentRun:
    """One synthetic run of the given profile."""
    if profile == "chat":
        stages = [Stage(branches=[Branch(steps=[_llm(rng, "answer", 600, 250, "sys-chat", 400)])])]

    elif profile == "react":
        # a ReAct loop: think, call a tool, think again... the prompt grows with each observation
        k = rng.randint(3, 7)
        steps: list[Step] = []
        prompt = 900
        for i in range(k):
            steps.append(_llm(rng, f"think-{i}", prompt, 120, "sys-react", 600))
            steps.append(_tool(rng, "search", 0.8))
            prompt += 350
        steps.append(_llm(rng, "final", prompt, 300, "sys-react", 600))
        stages = [Stage(branches=[Branch(steps=steps)])]

    elif profile == "fanout":
        # plan, fan out to sub-agents that each search and summarise, then synthesise
        n = rng.randint(4, 8)
        plan = Stage(branches=[Branch(steps=[_llm(rng, "plan", 800, 200, "sys-lead", 700)])])
        workers = Stage(
            branches=[
                Branch(
                    steps=[
                        _llm(rng, f"worker-{j}-query", 700, 80, "sys-worker", 500),
                        _tool(rng, "search", 1.2),
                        _llm(rng, f"worker-{j}-summary", 2200, 350, "sys-worker", 500),
                    ]
                )
                for j in range(n)
            ]
        )
        synth = Stage(branches=[Branch(steps=[_llm(rng, "synthesise", 1200 + 350 * n, 600, "sys-lead", 700)])])
        stages = [plan, workers, synth]

    elif profile == "clinical":
        # consult-to-note shape: extract facts, draft a long note, judge each sentence in parallel, revise once
        n_judge = rng.randint(6, 12)
        extract = Stage(branches=[Branch(steps=[_llm(rng, "extract", 2500, 400, "sys-extract", 900)])])
        draft = Stage(branches=[Branch(steps=[_llm(rng, "draft", 3200, 900, "sys-draft", 1100)])])
        judge = Stage(
            branches=[Branch(steps=[_llm(rng, f"judge-{j}", 900, 40, "sys-judge", 500)]) for j in range(n_judge)]
        )
        revise = Stage(
            branches=[
                Branch(steps=[_llm(rng, "revise", 4200, 500, "sys-draft", 1100), _tool(rng, "fhir-export", 0.15)])
            ]
        )
        stages = [extract, draft, judge, revise]

    else:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")

    return AgentRun(id=run_id, profile=profile, stages=stages)


def make_workload(profile: str, n_runs: int, seed: int = 0) -> list[AgentRun]:
    rng = random.Random(seed)
    return [make_run(profile, f"{profile}-{i:04d}", rng) for i in range(n_runs)]


def mixed_workload(mix: dict[str, float], n_runs: int, seed: int = 0) -> list[AgentRun]:
    """``mix`` maps profile name to weight, e.g. {"chat": 0.6, "react": 0.3, "fanout": 0.1}."""
    rng = random.Random(seed)
    names, weights = zip(*mix.items(), strict=True)
    return [make_run(rng.choices(names, weights)[0], f"mix-{i:04d}", rng) for i in range(n_runs)]


# ---------------------------------------------------------------- agenteval traces


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def from_agenteval(runs: Iterable[dict], prefix_tokens: int = 600, tool_seconds: float = 0.5) -> list[AgentRun]:
    """Convert agenteval ``AgentRun`` dicts (sequential steps) into simulator runs.

    agenteval records what each step read and wrote but not token counts, so tokens are estimated at four
    characters per token, and the prompt is assumed to accumulate the conversation so far (a ReAct-style loop).
    Tool steps take ``tool_seconds`` unless the trace carries a ``duration_s`` field.
    """
    out: list[AgentRun] = []
    for r in runs:
        steps: list[Step] = []
        context = _tokens(r.get("task", "")) + prefix_tokens
        for s in r.get("steps", []):
            if s.get("kind") == "tool_call":
                steps.append(
                    Step(kind="tool", name=s.get("name", "tool"), duration_s=float(s.get("duration_s", tool_seconds)))
                )
                context += _tokens(s.get("output", ""))
            else:
                out_tok = _tokens(s.get("output", "") or s.get("input", ""))
                steps.append(
                    Step(
                        kind="llm",
                        name=s.get("name") or s.get("kind", "llm"),
                        prompt_tokens=context,
                        output_tokens=out_tok,
                        prefix_id=f"sys-{r.get('framework', 'agent')}",
                        prefix_tokens=prefix_tokens,
                    )
                )
                context += out_tok
        final = r.get("final_answer", "")
        steps.append(
            Step(
                kind="llm",
                name="final",
                prompt_tokens=context,
                output_tokens=_tokens(final) if final else 100,
                prefix_id=f"sys-{r.get('framework', 'agent')}",
                prefix_tokens=prefix_tokens,
            )
        )
        out.append(
            AgentRun(
                id=str(r.get("id", len(out))),
                profile=r.get("framework", "agenteval"),
                stages=[Stage(branches=[Branch(steps=steps)])],
            )
        )
    return out


def load_runs(path: str | Path) -> list[AgentRun]:
    """Load runs from a JSONL file of simulator ``AgentRun`` records, or of agenteval records (auto-detected)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if rows and "stages" in rows[0]:
        return [AgentRun.model_validate(r) for r in rows]
    return from_agenteval(rows)


def save_runs(runs: Iterable[AgentRun], path: str | Path) -> None:
    Path(path).write_text("".join(r.model_dump_json() + "\n" for r in runs))
