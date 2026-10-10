import json

import pytest

from agentsim import bench
from agentsim.calibrate import fit


class Response:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter([("data: " + json.dumps(c) + "\n").encode() for c in self.chunks])


def test_streaming_chunks_are_not_tokens(monkeypatch):
    chunks = [{"choices": [{"delta": {"content": text}}]} for text in ["hello world ", "from model"]]
    chunks += [{"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 10}}]
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: Response(chunks))
    out = [{}]
    bench._one("http://localhost/v1", "model", "prompt", 10, out, 0)
    assert out[0]["output_tokens"] == 7 and out[0]["content_chunks"] == 2
    assert out[0]["token_count_source"] == "server_usage"


def test_missing_usage_is_explicit_estimate(monkeypatch):
    monkeypatch.setattr(
        bench.urllib.request,
        "urlopen",
        lambda *a, **kw: Response([{"choices": [{"delta": {"content": "hello world"}}]}]),
    )
    out = [{}]
    bench._one("http://localhost/v1", "model", "prompt", 10, out, 0)
    assert out[0]["token_count_source"].startswith("estimated")


def test_failed_request_is_recorded(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(bench.urllib.request, "urlopen", fail)
    out = [{}]
    bench._safe_one("http://localhost/v1", "model", "prompt", 10, out, 0)
    assert "500" in out[0]["error"]


def test_calibration_rejects_estimated_token_counts():
    with pytest.raises(ValueError, match="server-reported"):
        fit(
            [
                {
                    "batch_size": 1,
                    "prompt_tokens": 10,
                    "ttft_ms": 3,
                    "itl_ms": 2,
                    "token_count_source": "estimated_or_mixed",
                }
            ]
        )


def test_prefill_fit_does_not_use_estimated_prompt_lengths():
    from agentsim.schema import ClusterSpec

    base = ClusterSpec()
    fitted = fit(
        [
            {
                "batch_size": 1,
                "prompt_tokens": p,
                "ttft_ms": p,
                "itl_ms": 2,
                "token_count_source": "server_usage",
                "prompt_count_source": "estimated_prompt_length",
            }
            for p in [10, 100]
        ],
        base,
    )
    assert fitted.prefill_tokens_per_s == base.prefill_tokens_per_s
