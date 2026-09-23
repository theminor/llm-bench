# llm-bench

A web tool for benchmarking **any** LLM inference engine (llama.cpp, vLLM, or
anything else with an OpenAI-compatible HTTP API) across **any** set of
command-line parameters.

It answers the question llama-bench can't: *"does this flag people recommend on
the internet actually make my server faster, end-to-end?"* — because it measures
the **running server over HTTP**, exactly the way you use it, not the engine
library in-process.

## How it works

For every combination ("variant") of the parameters you choose:

1. llm-bench **launches the engine executable** you configured, with your base
   args + the variant's args (e.g. `llama-server -m ... --flash-attn on`).
2. Polls its **readiness endpoint** until it's up (`/health`).
3. Sends a **warmup** request, then N timed requests per workload.
4. Kills the server, cools down, moves to the next variant (strictly
   sequential, so your GPU is only ever used by one server).

Timings are captured by **streaming** (`/v1/completions` with SSE) so every
token arrival is timestamped. You get, mean ± stddev across repetitions:

| Metric | Meaning |
|---|---|
| Prefill t/s | prompt tokens / time-to-first-token (client) or server-reported |
| Decode t/s | generated tokens / decode window (client) or server-reported |
| TTFT | time to first output token (includes queueing + HTTP) |
| E2E | full request latency |
| ITL p50/p90/p99 | inter-token arrival latencies (jitter you actually feel) |

For **llama.cpp** it additionally parses the server's own `prompt eval time` /
`eval time` log lines (and the `timing` object llama.cpp adds to the final SSE
usage chunk), so you can separate real engine speed from HTTP overhead. For
engines without that, client-side numbers stand on their own.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run.py --port 8090            # then open http://127.0.0.1:8090
```

The **Engines** tab ships with presets for `llama-server`, `vllm serve`,
KoboldCpp, Ollama, SGLang, and Hugging Face TGI. An engine is deliberately *just
an executable + argument template*: `{model}` and `{port}` are substituted (in
args **and** env vars), and every other argument you sweep is passed through
verbatim, so any current or future engine flag works with zero code changes. If
your server needs auth, put the header (e.g. `{"Authorization": "Bearer ..."}`)
in the engine definition. Each engine can carry a **docs link** to its argument
reference, shown in the UI and in Markdown exports.

## Defining a sweep

* **Models** — add one or more models. Each is substituted into the engine's
  `{model}` argument and run under identical settings, so a multi-model sweep
  directly compares model speed (this is the cross-product dimension just like
  the others).
* **Workloads** mirror llama-bench's test kinds:
  * `pp` — prefill a prompt of N tokens, generate 1 (→ Prefill t/s, TTFT)
  * `tg` — generate N tokens from a tiny prompt (→ Decode t/s, ITL)
  * `pg` — realistic: prompt of N tokens then generate M (→ both + E2E)
  Every workload runs for every variant (model × dimensions).
* **Base args** (multi-line) — flags held constant across all variants; each
  line is shell-parsed.
* **Base env vars** (multi-line `KEY=value`) — environment variables held
  constant for every variant (e.g. pin a GPU). They override any *default* env
  set on the engine. A bare `KEY` means value `1`.
* **Dimensions** are swept as a cross-product. Each has a *type*, a *name* and
  *values*:
  * **arg** (default) — a command-line flag. Leave the arg template **empty**
    and the name becomes the flag (name `flash-attn`, values `on,off,auto` →
    `--flash-attn on`, …), or use a template with `{v}` (e.g. `--n-gpu-layers
    {v}`). Values accept llama-bench ranges `1-16+4`, `1-32*2`, literals, or
    whole flag-group fragments separated by `;`.
  * **env** — an environment variable. Each value is a `KEY=value` pair
    (separated by `;` so values may contain commas), e.g. name `cuda` with
    values `CUDA_VISIBLE_DEVICES=0;CUDA_VISIBLE_DEVICES=1`. These are set on
    the engine process, not the command line.
* **Repetitions** default to 3; each repetition gets a unique prompt suffix so
  prompt/prefix caching cannot fake-inflate your numbers.
* **Preview plan** renders the *exact* command **and env** for the first
  variant, so mistakes are visible before the sweep starts.
* **Clone** copies any finished sweep's configuration back into the New-sweep
  form so you can tweak and re-run instead of re-entering everything.

## Results, exports & logs

The **Results** tab aggregates any selection of sweeps into a table (with
charts) and exports:

* The table **heat-maps each metric column** green→yellow→red by rank, so the
  best and worst variants are obvious at a glance (t/s columns reward higher;
  latency columns reward lower). The chart draws each bar's mean with
  **±1 stddev whiskers** showing the spread of the repeated runs.
* A **"Best per workload" recommendation** picks the fastest variant by total
  request time and calls out the caveats that matter: when the fastest-on-
  average has much worse **tail latency** (ITL p99) than a consistent rival,
  when decode-speed and prefill-speed winners disagree, and when a variant shows
  **high run-to-run variance** (needs more reps). A **"How to read these
  numbers"** glossary explains every column.
* **Markdown** — a human-readable summary (`/api/export/markdown?sweep_ids=1,2`)
  for one or many sweeps: environment, engine docs link, models, dimensions,
  base env, per-variant mean ± stddev, the recommendation, and the glossary.
  Perfect for posting results back where you found the idea.
* CSV / JSON — aggregated metrics for further analysis
* SQL — `CREATE TABLE llm_bench` + inserts, pipe into `sqlite3` (llama-bench style)

The **Logs** tab is one place for the full stdout+stderr of every variant of
every sweep — where crashes, startup errors, and the engine's per-request
timing lines live. A running variant's log keeps refreshing while open.

## Running on a remote machine / Desktop Rig

Run `run.py` **on the machine that has the GPUs** (it spawns engines as local
subprocesses). Bind it to your LAN with `--host 0.0.0.0` and open the UI from
another machine, e.g. on the rig:

```bash
git clone https://github.com/theminor/llm-bench.git && cd llm-bench
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run.py --host 0.0.0.0 --port 8090
```

All state (engines, sweeps, samples, server logs) lives in `./data/`.

## Development / tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

The suite includes an **end-to-end test against a fake engine**
(`tests/fake_engine.py`) that mimics llama-server (health endpoint, SSE
completions with usage timings, `print_timing` log lines), so the full
pipeline — launch, readiness, warmup, measurement, timing extraction,
aggregation, exports — is verified on machines with no GPU and no models.

## Caveats

* HTTP-measured numbers legitimately differ from `llama-bench`: they include
  tokenization, sampling, and serving overhead. That *is* the user experience.
* Engine startup dominates long matrices; keep dimensions small or split them
  across sweeps.
* Thermals drift over long sweeps; interleave or re-run if a sweep takes hours.
* `ignore_eos`/`min_tokens` (used to keep `tg` runs honest) are supported by
  llama.cpp and vLLM; exotic engines may ignore them.

## License

MIT
