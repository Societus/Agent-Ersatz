# Agent-Ersatz

**er·satz** */ˈerˌzäts, ərˈzats/* — *adjective* — (of a product) made or used as a substitute, typically an inferior one, for something else. From German *Ersatz* "replacement," from *ersetzen* "to replace." First known use: 1875. Synonyms: artificial, mock, synthetic. "The coffee was an **ersatz** substitute, made from roasted acorns."

Agent-Ersatz is a survival toolkit for people running a personal AI agent on hardware that has no business running an AI agent. If you're pointing Hermes Agent at a local inference server hosted on a machine that shares duties with your media server, and your "fast" model is a 1.2B parameter afterthought that can barely format JSON -- you're in the right place. This is the substitute for having the proper resources: a rack of A100s, an OpenAI account with billing enabled, or the good sense not to try this at all.

It handles two problems that anyone running local LLMs through an agent framework will run into eventually: **configuration drift** (updates overwrite your custom settings) and **model performance uncertainty** (you don't know which model on your server will actually survive a real workload without timing out).

## What it does

**Config Survival** (`shelter.py`) watches your Hermes Agent configuration files and patches them back when updates clobber your settings. You declare what your config should look like in `state/baseline.yaml`, write static patches for known changes in `patches/`, and let the tool handle the rest. When a static patch isn't enough, it asks the local LLM to generate the edit, runs verification tests, and auto-reverts if anything breaks. A git post-merge hook means this runs automatically after `hermes update` -- you don't have to think about it.

**Model Benchmark** (`benchmark.py`) discovers every model on your inference server, runs a streaming benchmark against each one, and ranks them by actual measured performance. Not marketing numbers -- real prompt processing speed, real generation throughput, real time-to-first-token. It detects which models emit reasoning tokens (and can filter them out), estimates parameter counts from model names, probes context window limits, and calculates timeout recommendations so your agent framework doesn't kill connections before the model finishes thinking.

## How it works

### Auto-Detection

On first run, `shelter.py setup` discovers your local inference server without any hardcoded configuration:

1. Reads your Hermes `config.yaml` custom_providers, skipping any cloud URLs (api.openai.com, openrouter.ai, etc.)
2. Probes common local ports: LM Studio (1234), Ollama (11434), vLLM (8000), SGLang (30000), TabbyAPI (5000), koboldcpp (5001)
3. Offers an interactive selection if it finds multiple endpoints

No provider URL or model name lives in the repo. Everything goes into `shelter.conf`, which is gitignored and generated per machine.

### The Healing Pipeline

When `shelter.py heal` runs (automatically via post-merge hook, or manually):

1. **Detect** -- compares current file state against `baseline.yaml` declarations
2. **Static Patch** -- applies any matching patches from `patches/` (unified diff format)
3. **LLM Fallback** -- if static patches don't resolve all drift, sends the diff to the local LLM with instructions to generate a surgical edit
4. **Test** -- runs all scripts in `tests/` to verify nothing broke
5. **Commit or Revert** -- if tests pass, keeps changes. If tests fail, restores the pre-heal snapshot

### Benchmark Pipeline

When `benchmark.py` runs (or `shelter.py benchmark`):

1. **Discover** -- queries `/v1/models` on the configured provider
2. **Measure** -- sends a streaming completion request to each model, recording per-token timestamps via SSE
3. **Calculate** -- derives prompt processing speed (prompt_tokens / TTFT), generation throughput ((tokens - 1) / generation_time), and peak 1-second window speed
4. **Detect** -- checks for `reasoning_content` fields in stream deltas to identify reasoning models
5. **Probe** -- optionally tests increasing context sizes (4K, 8K, 16K, 32K) to find the actual context limit
6. **Recommend** -- extrapolates TTFT to max context, estimates generation time for 4096 output tokens, applies a 2.5x safety factor

## Usage

```bash
# First run: detect your local LLM provider
python shelter.py setup

# Check if your config has drifted from baseline
python shelter.py check

# Auto-heal drift (static patch + LLM fallback + test + revert)
python shelter.py heal

# Capture current state as a known-good snapshot
python shelter.py snapshot

# Run verification tests only
python shelter.py test

# Benchmark all models on the provider
python shelter.py benchmark

# Benchmark with additional flags (passed through to benchmark.py)
python shelter.py benchmark --quick
python shelter.py benchmark --full
python shelter.py benchmark --skip-reasoning
python shelter.py benchmark --model gemma
python shelter.py benchmark --recommend-timeouts
python shelter.py benchmark --save results.json
```

## Benchmark flags

| Flag | What it does |
|------|-------------|
| `--quick` | Fast mode: 512-token prompt, 32-token generation, 1 run. Good for a quick sanity check. |
| `--full` | Thorough mode: 2048-token prompt, 128-token generation, 3 runs, probes context at 4K/8K/16K/32K. Expect this to take a while. |
| `--skip-reasoning` | Excludes models with names matching reasoning patterns (qwq, deepseek-r1, o3, etc.) from the benchmark run. |
| `--model <name>` | Benchmarks only models whose name contains the given substring (case-insensitive). |
| `--save <path>` | Saves JSON results to a file. Defaults to `benchmark-YYYYMMDD-HHMMSS.json` in the project directory. |
| `--recommend-timeouts` | Adds gateway_timeout and aux_timeout columns to the output table, calculated from measured TTFT scaled to max context with a 2.5x safety factor. |

Default mode (no flags): 1024-token prompt, 64-token generation, 1 run, probes context at 4K and 16K.

## Project structure

```
Agent-Ersatz/
├── shelter.py            # Main entry: config survival + CLI
├── benchmark.py          # Model benchmark suite
├── shelter.conf          # Generated per-machine (gitignored)
├── state/
│   ├── baseline.yaml     # Expected config state declarations
│   └── snapshots/        # Timestamped state captures
├── patches/              # Static unified-diff patches for known changes
└── tests/                # Verification test scripts
```

## Generating documentation

Agent-Ersatz includes a repowise wiki in `.repowise/` with generated documentation for every file, module, and symbol. If you're working on this project and want to regenerate or update the docs, you can use repowise against the repo:

- **Using a local LLM**: [Societus/repowise](https://github.com/Societus/repowise) -- supports ollama, litellm, and any OpenAI-compatible local endpoint
- **Using cloud models**: [repowise-dev/repowise](https://github.com/repowise-dev/repowise) -- the upstream fork with full cloud provider support

```bash
# With a local inference server (e.g. LM Studio, Ollama)
OLLAMA_BASE_URL="http://your-server:1234" \
repowise init --provider ollama --model your-model-name

# To rebuild just the vector search index (fast, no LLM calls)
OPENAI_API_KEY="sk-local" \
OPENAI_BASE_URL="http://your-server:1234/v1" \
OPENAI_EMBEDDING_MODEL="your-embedding-model" \
repowise reindex --embedder openai

# Search the docs
repowise search "timeout configuration"
```

## Requirements

- Python 3.11+
- [Hermes Agent](https://github.com/h chapo-io/hermes) installed at `~/.hermes/`
- A local inference server (LM Studio, Ollama, vLLM, etc.) with at least one model loaded
- `httpx` and `pyyaml` in your Python environment

## The honest part

This tool exists because running a 26B-parameter MoE model on consumer hardware through an agent framework that expects sub-second API responses is, by any reasonable definition, an ersatz setup. The model takes two minutes to say "OK." The benchmark exists because you need to know which of your locally-hosted models can actually survive a real agent workload before you configure your timeouts. The config survival system exists because Hermes updates will silently rewrite your carefully tuned settings, and manually fixing them every time gets old fast.

It works. That's the bar.
