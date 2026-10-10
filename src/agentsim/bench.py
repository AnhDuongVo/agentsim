"""Measure an OpenAI-compatible endpoint (vLLM, NIM, TGI, Ollama, a hosted API) for calibration.

For each concurrency level and prompt length, ``concurrency`` streaming chat completions are sent at once
and, for each, the time to the first content token and the mean inter-token interval are recorded.
Needs only the standard library. Set ``AGENTSIM_API_KEY`` if the endpoint wants a key.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog. "


def _prompt(n_tokens: int) -> str:
    # ~10 tokens per filler sentence on most tokenizers
    return FILLER * max(1, n_tokens // 10)


def _one(base_url: str, model: str, prompt: str, max_tokens: int, out: list[dict], idx: int) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("AGENTSIM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    t0 = time.perf_counter()
    first = None
    stamps: list[float] = []
    completion_tokens = None
    prompt_usage = None
    content = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = usage["completion_tokens"]
                prompt_usage = usage.get("prompt_tokens")
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}).get("content") if choices else None
            if delta:
                now = time.perf_counter()
                if first is None:
                    first = now
                stamps.append(now)
                content.append(delta)
    if first is None:
        out[idx] = {"error": "No content returned"}
        return
    measured = completion_tokens is not None
    tokens = completion_tokens if measured else max(1, round(len("".join(content)) / 4))
    # Effective decode throughput, not a per-token timestamp measurement: chunks may contain many tokens.
    itl = (stamps[-1] - stamps[0]) / (tokens - 1) if tokens > 1 and len(stamps) > 1 else None
    out[idx] = {
        "ttft_ms": round((first - t0) * 1000, 1),
        "itl_ms": round(itl * 1000, 2) if itl is not None else None,
        "output_tokens": tokens,
        "prompt_tokens": prompt_usage,
        "token_count_source": "server_usage" if measured else "estimated_chars_div_4",
        "content_chunks": len(stamps),
        "timing_method": "effective_decode_interval",
    }


def _safe_one(*args) -> None:
    try:
        _one(*args)
    except Exception as err:
        args[-2][args[-1]] = {"error": f"{type(err).__name__}: {err}"}


def measure(
    base_url: str, model: str, concurrencies: list[int], prompt_tokens: list[int], max_tokens: int = 64
) -> list[dict]:
    rows: list[dict] = []
    for c in concurrencies:
        for p in prompt_tokens:
            prompt = _prompt(p)
            out: list[dict] = [{} for _ in range(c)]
            threads = [
                threading.Thread(target=_safe_one, args=(base_url, model, prompt, max_tokens, out, i)) for i in range(c)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            good = [o for o in out if o and not o.get("error") and o.get("itl_ms") is not None]
            if not good:
                rows.append(
                    {
                        "batch_size": c,
                        "prompt_tokens": p,
                        "n": 0,
                        "errors": [o.get("error", "Insufficient content for timing") for o in out],
                    }
                )
                continue
            rows.append(
                {
                    "batch_size": c,
                    "prompt_tokens": round(sum(o["prompt_tokens"] or p for o in good) / len(good)),
                    "prompt_count_source": "server_usage"
                    if all(o["prompt_tokens"] is not None for o in good)
                    else "estimated_prompt_length",
                    "token_count_source": "server_usage"
                    if all(o["token_count_source"] == "server_usage" for o in good)
                    else "estimated_or_mixed",
                    "errors": [o["error"] for o in out if o.get("error")],
                    "ttft_ms": round(sum(o["ttft_ms"] for o in good) / len(good), 1),
                    "itl_ms": round(sum(o["itl_ms"] for o in good) / len(good), 2),
                    "n": len(good),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    return rows
