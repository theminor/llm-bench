"""Black-box measurement of an OpenAI-compatible endpoint via SSE streaming.

Captures, per request:
  - TTFT (time to first content token)
  - inter-token arrival latencies (mean / p50 / p90 / p99)
  - client-side prefill and decode throughput (from usage token counts)
  - server-reported timings when the engine provides them
    (llama.cpp sends a `timing` object in the final usage chunk)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import httpx

from .metrics import percentile

_WORDS = (
    "the world sees change in many forms and yet the measured pace of each "
    "day holds steady while distant signals drift slowly beyond reach of any "
    "single observer gathering notes from a crowded room where voices rise "
    "and fall like tides beneath a indifferent moon that watches without "
    "judgement as time carries every pattern toward some quiet horizon line"
).split()


def build_prompt(n_tokens: int, seed: int) -> str:
    """Deterministic pseudo-random prompt of approximately n_tokens tokens.

    Random content defeats prompt/prefix caching. Actual token counts are
    taken from the response `usage` fields, so approximation is fine.
    """
    rng = random.Random(seed)
    words: list[str] = []
    while len(words) < max(1, int(n_tokens * 0.8) + 1):
        words.append(rng.choice(_WORDS))
        if rng.random() < 0.08:
            words.append(str(rng.randrange(100000)))
    return " ".join(words[: max(1, n_tokens)]) + f" [bench-{seed}]"


@dataclass
class Sample:
    workload: str = ""
    rep: int = 0
    ok: bool = True
    error: str = ""

    ttft_ms: float | None = None
    e2e_ms: float | None = None
    pp_tps_client: float | None = None
    tg_tps_client: float | None = None
    pp_tps_server: float | None = None
    tg_tps_server: float | None = None
    itl_mean_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p90_ms: float | None = None
    itl_p99_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict = field(default_factory=dict)


async def measure_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    workload: str,
    rep: int,
    headers: dict[str, str] | None = None,
) -> Sample:
    sample = Sample(workload=workload, rep=rep)
    payload = {
        "model": model or "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first: float | None = None
    token_times: list[float] = []
    usage: dict = {}
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json=payload,
            headers=headers or {},
            timeout=httpx.Timeout(600.0),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    text = choice.get("text", "")
                    if text:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        token_times.append(now)
    except Exception as e:  # engine error / connection reset / timeout
        sample.ok = False
        sample.error = f"{type(e).__name__}: {e}"
        return sample

    t_end = time.perf_counter()
    sample.e2e_ms = (t_end - t0) * 1e3
    if t_first is None:
        sample.ok = False
        sample.error = "no content tokens received"
        return sample

    sample.ttft_ms = (t_first - t0) * 1e3
    completion_tokens = int(usage.get("completion_tokens", 0) or 0) or len(token_times)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    sample.prompt_tokens = prompt_tokens
    sample.completion_tokens = completion_tokens

    if prompt_tokens and sample.ttft_ms and sample.ttft_ms > 0:
        sample.pp_tps_client = prompt_tokens / (sample.ttft_ms / 1e3)

    # decode rate excludes the first token (produced with the prefill pass)
    if len(token_times) > 1:
        intervals = [
            (token_times[i] - token_times[i - 1]) * 1e3
            for i in range(1, len(token_times))
        ]
        decode_ms = (token_times[-1] - token_times[0]) * 1e3
        sample.itl_mean_ms = sum(intervals) / len(intervals)
        sample.itl_p50_ms = percentile(intervals, 50)
        sample.itl_p90_ms = percentile(intervals, 90)
        sample.itl_p99_ms = percentile(intervals, 99)
        if decode_ms > 0:
            decode_tokens = max(completion_tokens - 1, 1)
            sample.tg_tps_client = decode_tokens / (decode_ms / 1e3)

    # server-reported timings, if the engine emits them (llama.cpp `timing`)
    timing = usage.get("timing") or usage.get("timings") or {}
    if isinstance(timing, dict):
        s_pp = _ms(timing, ("t_prompt_ms", "prompt_ms", "pre_ms", "promptEvalMs"))
        s_tg = _ms(timing, ("t_eval_ms", "eval_ms", "g_ms", "evalMs"))
        if s_pp and prompt_tokens:
            sample.pp_tps_server = prompt_tokens / (s_pp / 1e3)
        if s_tg and completion_tokens:
            sample.tg_tps_server = completion_tokens / (s_tg / 1e3)

    sample.raw = {"usage": usage}
    return sample


def _ms(timing: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = timing.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None
