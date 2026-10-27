# Inference-Pareto-Router

### *Sub-millisecond 2D model-tier and reasoning-effort routing for LLM inference*

[![PyPI version](https://img.shields.io/pypi/v/pareto-router.svg)](https://pypi.org/project/pareto-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/pareto-router.svg)](https://pypi.org/project/pareto-router/)
[![License](https://img.shields.io/pypi/l/pareto-router.svg)](LICENSE)
[![CI](https://github.com/Sandeep-Hipparagi/Inference-Pareto-Router/actions/workflows/ci-publish.yml/badge.svg)](https://github.com/Sandeep-Hipparagi/Inference-Pareto-Router/actions/workflows/ci-publish.yml)
[![mypy](https://img.shields.io/badge/mypy-strict-success.svg)](https://mypy.readthedocs.io/)
[![Routing latency](https://img.shields.io/badge/decision%20latency-p99%200.181%20ms-blue.svg)](#benchmarks)

**Developed by [Sandeep Hipparagi](https://www.bluepatterns.ai/).**

Every prompt that reaches an LLM costs two independent resources: the *tier* of the
model you send it to, and the *test-time reasoning effort* you let that model spend.
Almost every router in the wild optimises one dimension — cheap model versus
expensive model — and then throws a fixed reasoning budget on top. That is why a
greeting can burn 8,000 thinking tokens on a frontier model while a proof request
gets a one-shot answer from a mini model.

Inference-Pareto-Router treats routing as a **point in a 2D space**. It scores a
prompt on two orthogonal axes, `logic` (verifiability, algorithmic depth, math/code
structure) and `knowledge` (entity richness, domain nuance, world knowledge), and in
**under 0.2 ms** returns a coordinate: `(ModelTier, ReasoningEffort)` — plus the
exact, provider-ready request body that implements it.

---

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart) · [Constraints and the routing decision](#quickstart)
- [How the 2D decision is computed](#how-the-2d-decision-is-computed)
- [Strategy presets](#strategy-presets)
- [Benchmarks](#benchmarks)
- [Provider routing table](#provider-routing-table)
- [Features](#features)
- [CLI](#cli)
- [What is and is not implemented](#what-is-and-is-not-implemented)
- [Development, testing and CI](#development-testing-and-ci)
- [Attribution and license](#attribution-and-license)

---

## Installation

```bash
pip install pareto-router
```

> The repository is named **Inference-Pareto-Router**; the PyPI distribution is
> **`pareto-router`**. `pip install pareto-router` is the canonical command.

Requires **Python 3.11+**. With `uv` or `pipx`:

```bash
uv add pareto-router
pipx run pareto-router "Why is my asyncio task never awaited?"
```

Install the development toolchain (pytest, pytest-asyncio, mypy):

```bash
pip install "pareto-router[dev]"
```

---

## Quickstart

Fifteen lines: set your cost/latency/quality constraints, score a prompt, and get
both the routing decision and the provider-ready payload.

```python
from pareto_router import (HeuristicEvaluator, ParetoPolicy, ProviderNormalizer,
                           ProviderType, RoutingStrategy)

policy = ParetoPolicy(
    logic_threshold_low=0.15,       # below this: no reasoning at all (STANDARD mode)
    logic_threshold_high=0.75,      # above this: always FRONTIER
    knowledge_threshold_high=0.65,  # above this: always FRONTIER
    overthinking_threshold=0.25,    # clamp reasoning on greetings / trivial asks
    default_strategy=RoutingStrategy.PARETO_COST,
)

scores = HeuristicEvaluator().evaluate_sync("Debug this race condition in a Rust mutex.")
decision = policy.evaluate_coordinates(scores, strategy=RoutingStrategy.PARETO_QUALITY)
payload = ProviderNormalizer().to_provider_payload(
    decision, ProviderType.OPENAI, [{"role": "user", "content": "Debug this race condition."}])

print(decision.tier, decision.effort_level, decision.reasoning_budget_tokens)
print(payload.model, payload.body.get("reasoning_effort"))
# -> ModelTier.FRONTIER EffortLevel.HIGH 25488
# -> o1 high
```

`payload` is a complete, dispatchable request: `payload.model`, `payload.body`
(including the provider-specific reasoning knob) and `payload.headers`. Hand it to
your own client, or use the gateway wrapper below.

### The drop-in gateway wrapper

`ParetoRouter` exposes an OpenAI-shaped surface for exercising routing behaviour in
tests and demos:

```python
from pareto_router import ParetoRouter, RoutingStrategy

router = ParetoRouter(strategy=RoutingStrategy.PARETO_BALANCED)  # reads OPENAI_API_KEY
resp = router.chat.completions.create(
    messages=[{"role": "user", "content": "Summarise this error log in one line."}],
)

print(resp.routing_decision.tier, resp.routing_decision.effort_level, resp.model)
```

Read [What is and is not implemented](#what-is-and-is-not-implemented) before using
this in production — the transport layer in `0.1.x` is a deterministic stub, not a
live provider call.

---

## How the 2D decision is computed

The router is a pure function of two scores and one strategy preset. There is no
model call in the decision path, which is why it costs microseconds rather than
milliseconds.

### 1. The two axes

$$s = (\,l,\;k\,) \quad\text{where}\quad l = \text{logic score} \in [0,1],\;\; k = \text{knowledge score} \in [0,1]$$

`l` measures how much *verifiable structure* a prompt has — proofs, code, multi-step
arithmetic, explicit constraints. `k` measures how much *breadth of knowledge or
nuance* it needs — named entities, domain jargon, cultural or stylistic depth.
Scores come from a pluggable evaluator (`HeuristicEvaluator` ships in-box;
`ONNXEvaluator` and `DecisionModelClient` allow an external classifier).

### 2. Overthinking clamp

Frontier reasoning models measurably degrade on trivial inputs. If the evaluator
flags low-entropy input (`is_low_entropy`, `is_greeting`, `is_aesthetic`) **and**
`l < overthinking_threshold`, effort is pinned to `0.0` and the decision records
`overthinking_penalized = True`.

### 3. Continuous effort

For everything else, effort is a rescaled distance above the low threshold:

$$e = \operatorname{clamp}\!\left(\frac{l - \tau_{\text{low}}}{1 - \tau_{\text{low}}} + \beta + \sigma,\;\; 0.05,\;\; 1.0\right)$$

where $\beta = 0.10$ when `k > knowledge_threshold_high` and `l > 0.50` (a
"hard-and-broad" boost), and $\sigma \in \{-0.10, 0, +0.10\}$ is the strategy
adjustment. Results are rounded to three decimals.

### 4. Discrete buckets and mode

| Effort score | `EffortLevel` | Mode |
|---|---|---|
| `≤ 0.001` | `NONE` | `STANDARD` (thinking disabled) |
| `(0.001, 0.35]` | `LOW` | `REASONING` |
| `(0.35, 0.70]` | `MEDIUM` | `REASONING` |
| `(0.70, 1.0]` | `HIGH` | `REASONING` |

### 5. Tier selection

$$\text{tier} = \begin{cases}
\text{FRONTIER} & k \ge \tau^{k}_{high} \;\lor\; l \ge \tau^{l}_{high} \;\lor\; (k \ge 0.50 \land l \ge 0.50)\\
\text{MID} & k \ge \tau^{k}_{low} \;\lor\; l \ge \tau^{l}_{low}\\
\text{COMPACT} & \text{otherwise}
\end{cases}$$

### 6. Token budget

When mode is `REASONING`, the effort score becomes an integer thinking budget:

$$B = \lfloor 1024 + e \cdot (32000 - 1024) \rfloor \quad\text{(Anthropic, OpenAI)}$$
$$B_{\text{gemini}} = \lfloor 1024 + e \cdot (24576 - 1024) \rfloor$$

---

## Strategy presets

A preset shifts every threshold at once, moving the operating point along the
frontier without changing the algorithm:

| Threshold | `PARETO_BALANCED` | `PARETO_COST` | `PARETO_QUALITY` |
|---|---|---|---|
| `logic_threshold_low` | 0.15 | 0.25 | 0.10 |
| `logic_threshold_high` | 0.75 | 0.85 | 0.60 |
| `knowledge_threshold_low` | 0.35 | 0.45 | 0.25 |
| `knowledge_threshold_high` | 0.65 | 0.75 | 0.50 |
| `overthinking_threshold` | 0.25 | 0.35 | 0.15 |
| Effort bias | — | −0.10 | +0.10 |

`PARETO_COST` needs more evidence before it escalates and trims effort; 
`PARETO_QUALITY` escalates earlier and adds 0.10 to every effort score.

---

## Benchmarks

Measured on the in-repo 100-scenario synthetic suite
(`python -m pareto_router.benchmarks.run_benchmarks`, results in
[`benchmark_results.json`](benchmark_results.json)):

| Metric | Result |
|---|---|
| Scenarios evaluated | 100 |
| Mean decision latency | **0.099 ms** |
| p50 / p90 / p95 / p99 | 0.086 / 0.143 / 0.159 / **0.181 ms** |
| Max observed | 0.404 ms |
| 15 ms latency budget | **Passed** (151× headroom at p99) |
| Tier distribution | compact 21 · mid 41 · frontier 38 |
| Effort distribution | none 63 · low 16 · medium 12 · high 9 |
| Overthinking clamp accuracy | 16 of 20 expected (80%) |
| Estimated cost saving vs. always-frontier-high | **71.5%** |
| Estimated latency saving vs. always-frontier-high | **70.1%** |

The saving figures are modelled against a baseline that sends every prompt to a
frontier model with high reasoning effort. They are estimates from the tier/effort
distribution, not billed invoices — reproduce them with the benchmark module.

---

## Provider routing table

`(tier, mode)` resolves to a concrete model per provider via `ProviderModelCatalog`,
which you can override to track your own deployments:

| Tier | Mode | OpenAI | Anthropic | Gemini |
|---|---|---|---|---|
| COMPACT | standard | `gpt-4o-mini` | `claude-3-5-haiku-20241022` | `gemini-2.5-flash` |
| COMPACT | reasoning | `o3-mini` | `claude-3-7-sonnet-20250219` | `gemini-2.5-flash` |
| MID | standard | `gpt-4o` | `claude-3-5-sonnet-20241022` | `gemini-2.5-flash` |
| MID | reasoning | `o3-mini` | `claude-3-7-sonnet-20250219` | `gemini-2.5-pro` |
| FRONTIER | standard | `gpt-4o` | `claude-3-5-sonnet-20241022` | `gemini-2.5-pro` |
| FRONTIER | reasoning | `o1` | `claude-3-7-sonnet-20250219` | `gemini-2.5-pro` |

---

## Features

- **Sub-millisecond, dependency-light decision path.** Pydantic v2 + NumPy only for
  the core; `httpx`, `openai` and `onnxruntime` are used by the optional transport
  and ONNX evaluator.
- **Pluggable evaluation.** `HeuristicEvaluator` (in-box, no network),
  `ONNXEvaluator` (local classifier), `DecisionModelClient` (external logits
  service) all satisfy the same `BaseEvaluator` contract.
- **Sliding-window context triage.** The last `sliding_window_size` turns inform the
  score without re-scoring the whole transcript.
- **Speculative cascade.** `SpeculativeCascade` plus verifiers
  (`JsonSchemaVerifier`, `PythonSyntaxVerifier`, `RegexVerifier`, `CustomVerifier`)
  escalate a failed compact attempt to a stronger coordinate.
- **Streaming shapes.** `ParetoStream` / `ParetoAsyncStream` expose
  `ChatCompletionChunk`-compatible iteration with the routing decision attached.
- **Prometheus telemetry.** `PrometheusMetricsExporter` records tier dispatches,
  overthinking clamps and evaluation overhead in exposition or JSON form.
- **Strict typing.** `mypy --strict` over both `pareto_router` and `tests` in CI.

---

## CLI

Installed as the `pareto-router` console script:

```bash
pareto-router "Explain why a B-tree beats a hash index for range scans"
pareto-router --provider anthropic "hi there"      # overthinking clamp fires
cat prompt.txt | pareto-router --provider gemini   # reads stdin
```

It prints the two axis scores, the decision latency, the chosen tier, mode, effort
level and token budget, and the exact model plus payload extras that would be
dispatched.

---

## What is and is not implemented

Honest boundaries, so you do not discover them in production:

| Capability | Status |
|---|---|
| Two-axis scoring, decision maths, tier/effort/budget selection | **Implemented and tested** |
| Provider payload translation (model + reasoning knobs + headers) | **Implemented and tested** |
| CLI triage | **Implemented** |
| Benchmark suite and latency budget check | **Implemented** |
| Cascade + verifier escalation logic | **Implemented** (executes against the stub executor) |
| Prometheus/JSON telemetry | **Implemented** |
| **Live provider transport in `ParetoRouter.create()`** | **Not implemented in 0.1.x** — returns a deterministic stub `InferenceResponse` so routing can be tested without spending tokens. Dispatch `payload` with your own client for real inference. |
| Adaptive feedback loop from observed cost/quality | Planned |

---

## Development, testing and CI

```bash
git clone https://github.com/Sandeep-Hipparagi/Inference-Pareto-Router.git
cd Inference-Pareto-Router
uv sync --all-extras --dev

uv run pytest -v                     # test suite
uv run mypy --strict pareto_router tests
uv run python -m pareto_router.benchmarks.run_benchmarks
```

CI (`.github/workflows/ci-publish.yml`) runs on every push and pull request to
`main`: a Python 3.11/3.12 matrix that installs with `uv`, runs `mypy --strict`
over package **and** tests, runs `pytest`, then executes the benchmark suite with
its latency-budget assertion. On a `v*` tag it builds the sdist and wheel and
publishes to PyPI via trusted publishing (OIDC), falling back to `PYPI_API_TOKEN`
when configured.

---

## Attribution and license

Designed and developed by **[Sandeep Hipparagi](https://www.bluepatterns.ai/)**.

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE). You may use,
modify and redistribute it, including commercially, with an express patent grant;
retain the copyright and attribution notices.

Contributions are welcome: open an issue describing the routing behaviour you need,
and include a benchmark scenario that demonstrates it.
