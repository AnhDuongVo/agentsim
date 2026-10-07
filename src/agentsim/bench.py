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
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                now = time.perf_counter()
                if first is None:
                    first = now
                stamps.append(now)
    if first is None:
        return
    itl = (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else 0.0
    out[idx] = {"ttft_ms": round((first - t0) * 1000, 1), "itl_ms": round(itl * 1000, 2), "output_tokens": len(stamps)}


def measure(
    base_url: str, model: str, concurrencies: list[int], prompt_tokens: list[int], max_tokens: int = 64
) -> list[dict]:
    rows: list[dict] = []
    for c in concurrencies:
        for p in prompt_tokens:
            prompt = _prompt(p)
            out: list[dict] = [{} for _ in range(c)]
            threads = [
                threading.Thread(target=_one, args=(base_url, model, prompt, max_tokens, out, i)) for i in range(c)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            good = [o for o in out if o]
            if not good:
                continue
            rows.append(
                {
                    "batch_size": c,
                    "prompt_tokens": p,
                    "ttft_ms": round(sum(o["ttft_ms"] for o in good) / len(good), 1),
                    "itl_ms": round(sum(o["itl_ms"] for o in good) / len(good), 2),
                    "n": len(good),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    return rows
