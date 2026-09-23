"""Engine profiles: how to launch, probe, talk to, and stop any engine.

An engine is deliberately just an executable plus an argument template, so
llama.cpp, vLLM, or any future server works without code changes. Profiles
are stored as JSON (see db.py) and managed from the UI.
"""
from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# llama-server prints one of these per completed completion (INFO level):
#   slot print_timing: id  0 | task 0 | prompt eval time =      1234.56 ms /   512 tokens ( ... 414.56 tokens per second)
#   slot print_timing: id  0 | task 0 | eval time =             876.54 ms /   128 tokens ( ... 146.02 tokens per second)
_LLAMA_PP_RE = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second")
_LLAMA_TG_RE = re.compile(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second")


DEFAULT_ENGINE = "llama.cpp (llama-server)"


def builtin_presets() -> list[dict[str, Any]]:
    return [
        {
            "name": "llama.cpp (llama-server)",
            "executable": "llama-server",
            "args": ["--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"],
            "ready_path": "/health",
            "ready_timeout_s": 300.0,
            "model_required": True,
            "timing": "llamacpp",
            "headers": {},
            "env": {},
            "docs": "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md",
            "note": "llama.cpp HTTP server. Any llama-server flag can be swept.",
        },
        {
            "name": "vLLM",
            "executable": "vllm",
            "args": ["serve", "{model}", "--host", "127.0.0.1", "--port", "{port}"],
            "ready_path": "/health",
            "ready_timeout_s": 600.0,
            "model_required": True,
            "timing": "none",
            "headers": {},
            "env": {},
            "docs": "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
            "note": "vLLM OpenAI-compatible server ({model} = model name or path).",
        },
        {
            "name": "KoboldCpp",
            "executable": "koboldcpp",
            "args": ["-m {model}", "--port {port}", "--device cuda"],
            "ready_path": "/health",
            "ready_timeout_s": 300.0,
            "model_required": True,
            "timing": "llamacpp",
            "headers": {},
            "env": {},
            "docs": "https://github.com/LostRuins/koboldcpp",
            "note": "Single-binary GGUF server, OpenAI-compatible. Drop --device cuda for CPU; "
                   "flags vary a bit between versions — verify for yours.",
        },
        {
            "name": "Ollama",
            "executable": "ollama",
            "args": ["serve"],
            "ready_path": "/v1/models",
            "ready_timeout_s": 120.0,
            "model_required": False,
            "timing": "none",
            "headers": {},
            "env": {"OLLAMA_HOST": "127.0.0.1:{port}"},
            "docs": "https://github.com/ollama/ollama/blob/main/docs/openai.md",
            "note": "Runs `ollama serve` per variant on a free port (model comes from the sweep's "
                   "Models field). `ollama pull <model>` first. For per-variant tuning, sweep "
                   "OLLAMA_* / num_gpu as env dimensions. If a system Ollama already runs, this "
                   "port-bound instance is independent.",
        },
        {
            "name": "SGLang",
            "executable": "python3",
            "args": ["-m sglang.launch_server", "--model-path {model}",
                     "--host 127.0.0.1", "--port {port}"],
            "ready_path": "/health",
            "ready_timeout_s": 600.0,
            "model_required": True,
            "timing": "none",
            "headers": {},
            "env": {},
            "docs": "https://docs.sglang.ai/backend/openai_api_compatibility.html",
            "note": "Fast local server, OpenAI-compatible. {model} = HF repo id or local dir.",
        },
        {
            "name": "TGI (Hugging Face)",
            "executable": "text-generation-server",
            "args": ["--model-id {model}", "--host 127.0.0.1", "--port {port}"],
            "ready_path": "/health",
            "ready_timeout_s": 600.0,
            "model_required": True,
            "timing": "none",
            "headers": {},
            "env": {},
            "docs": "https://huggingface.co/docs/text-generation-inference/conceptual/openai",
            "note": "HF Text Generation Inference; OpenAI-compatible on /v1/completions.",
        },
    ]


@dataclass
class EngineProfile:
    name: str
    executable: str
    args: list[str] = field(default_factory=list)
    ready_path: str = "/health"
    ready_timeout_s: float = 300.0
    model_required: bool = True
    timing: str = "llamacpp"  # "llamacpp" | "none"
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    docs: str = ""  # link to the engine's argument reference
    note: str = ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EngineProfile":
        return EngineProfile(
            name=str(d["name"]),
            executable=str(d["executable"]),
            args=[str(a) for a in d.get("args", [])],
            ready_path=str(d.get("ready_path", "/health")),
            ready_timeout_s=float(d.get("ready_timeout_s", 300.0)),
            model_required=bool(d.get("model_required", True)),
            timing=str(d.get("timing", "llamacpp")),
            headers={str(k): str(v) for k, v in d.get("headers", {}).items()},
            env={str(k): str(v) for k, v in d.get("env", {}).items()},
            docs=str(d.get("docs", "")),
            note=str(d.get("note", "")),
        )

    def render_args(self, model: str, port: int) -> list[str]:
        # Each args line is shell-split, so lines may be written naturally
        # as "--model {model}" (flag + value) instead of one token per line.
        # Quote a value if it may contain spaces: --model "{model}".
        out = []
        for a in self.args:
            line = a.replace("{model}", model).replace("{port}", str(port))
            try:
                out.extend(shlex.split(line))
            except ValueError as e:
                raise ValueError(f"unparseable engine arg line {line!r}: {e}")
        return out

    def render_headers(self, port: int) -> dict[str, str]:
        return {k: v.replace("{port}", str(port)) for k, v in self.headers.items()}


class ServerProcess:
    """Launches an engine, captures its log, waits for readiness, kills cleanly."""

    def __init__(
        self,
        profile: EngineProfile,
        variant_args: list[str],
        model: str,
        port: int,
        log_path: str,
        ready_timeout_s: float | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.profile = profile
        self.variant_args = variant_args
        self.model = model
        self.port = port
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s or profile.ready_timeout_s
        # Sweep-level env (base + per-variant dimension overrides). Applied on
        # top of the engine's own env so sweeps can test/override variables.
        self.extra_env = dict(extra_env or {})
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: asyncio.subprocess.Process | None = None
        self._drainer: asyncio.Task | None = None
        self._log = None
        self.startup_error: str = ""
        self.lines: list[str] = []

    @property
    def command(self) -> list[str]:
        return [self.profile.executable] + self.profile.render_args(
            self.model, self.port
        ) + self.variant_args

    @property
    def merged_env(self) -> dict[str, str]:
        """Full env the engine process runs with: process < engine < sweep.

        Env values support the same {port} / {model} placeholders as the args.
        """
        env = dict(os.environ)
        for src in (self.profile.env, self.extra_env):
            for k, v in src.items():
                env[str(k)] = str(v).replace("{port}", str(self.port)).replace("{model}", self.model)
        return env

    async def start(self) -> None:
        env = self.merged_env
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "w")
        self._log.write("$ " + " ".join(self.command) + "\n")
        if self.extra_env:
            shown = " ".join(f"{k}={v}" for k, v in self.extra_env.items())
            self._log.write(f"env+ {shown}\n")
        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._drainer = asyncio.create_task(self._drain())
        await self._wait_ready()

    async def _drain(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            self.lines.append(text)
            try:
                self._log.write(text + "\n")
                self._log.flush()
            except Exception:
                pass

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        url = self.base_url + self.profile.ready_path
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self.proc and self.proc.returncode is not None:
                    raise RuntimeError(
                        f"engine exited early (code {self.proc.returncode}); "
                        f"see {self.log_path}"
                    )
                try:
                    r = await client.get(url, timeout=2.0)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        raise TimeoutError(
            f"engine not ready at {url} within {self.ready_timeout_s:.0f}s"
        )

    def llamacpp_timings_since(self, since_index: int) -> tuple[list[float], list[float]]:
        """Parsed (pp_tps, tg_tps) log timings emitted since a line index."""
        pp: list[float] = []
        tg: list[float] = []
        for line in self.lines[since_index:]:
            m = _LLAMA_PP_RE.search(line)
            if m:
                pp.append(float(m.group(3)))
                continue
            m = _LLAMA_TG_RE.search(line)
            if m:
                tg.append(float(m.group(3)))
        return pp, tg

    async def stop(self) -> None:
        if not self.proc or self.proc.returncode is not None:
            await self._finish()
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                await self.proc.wait()
        except ProcessLookupError:
            pass
        await self._finish()

    async def _finish(self) -> None:
        if self._drainer:
            try:
                await asyncio.wait_for(self._drainer, timeout=5)
            except asyncio.TimeoutError:
                self._drainer.cancel()
            self._drainer = None
        if self._log:
            self._log.close()
            self._log = None
