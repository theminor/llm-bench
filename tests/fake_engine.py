#!/usr/bin/env python3
"""A stand-in engine that behaves enough like llama-server to test llm-bench.

Implements:
  GET  /health           -> 200
  POST /v1/completions   -> SSE stream, final usage chunk with a llama.cpp-style
                            `timing` object, and llama.cpp-style print_timing
                            lines on stdout after each completion.

Usage: fake_engine.py --port N [--tg-tps 40] [--pp-tps 800]
"""
import argparse
import asyncio
import random
import time
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=0)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--tg-tps", type=float, default=40.0)
parser.add_argument("--pp-tps", type=float, default=800.0)
parser.add_argument("--model", default="fake-model")
args, _unknown = parser.parse_known_args()

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


def sse(obj: dict) -> str:
    return f"data: {json_dumps(obj)}\n\n"


def json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj)


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()
    prompt = str(body.get("prompt", ""))
    max_tokens = int(body.get("max_tokens", 16))
    n_prompt = max(1, len(prompt.split()))
    rng = random.Random(hash(prompt) & 0x7FFFFFFF)

    pp_noise = 1.0 + rng.uniform(-0.02, 0.02)
    tg_noise = 1.0 + rng.uniform(-0.02, 0.02)
    pp_ms = n_prompt / (args.pp_tps * pp_noise) * 1e3
    tg_ms = max_tokens / (args.tg_tps * tg_noise) * 1e3
    interval = tg_ms / max(max_tokens, 1) / 1e3

    async def gen() -> AsyncIterator[str]:
        await asyncio.sleep(pp_ms / 1e3)  # simulated prefill
        for i in range(max_tokens):
            yield sse({
                "choices": [{"index": 0, "text": "tok " if i else "tok", "finish_reason": None}],
                "model": args.model,
            })
            if i:
                await asyncio.sleep(interval)
        usage = {
            "prompt_tokens": n_prompt,
            "completion_tokens": max_tokens,
            "total_tokens": n_prompt + max_tokens,
            "timing": {"t_prompt_ms": pp_ms, "t_eval_ms": tg_ms},
        }
        yield sse({"choices": [], "usage": usage})
        yield "data: [DONE]\n\n"
        print(
            f"slot print_timing: id  0 | task 0 | prompt eval time = {pp_ms:10.2f} ms /"
            f" {n_prompt:5d} tokens ({pp_ms / n_prompt:7.2f} ms per token,"
            f" {n_prompt / (pp_ms / 1e3):10.2f} tokens per second)",
            flush=True,
        )
        print(
            f"slot print_timing: id  0 | task 0 | eval time = {tg_ms:10.2f} ms /"
            f" {max_tokens:5d} tokens ({tg_ms / max(max_tokens, 1):7.2f} ms per token,"
            f" {max_tokens / (tg_ms / 1e3):10.2f} tokens per second)",
            flush=True,
        )

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
