# Agent-Ersatz

Self-healing configuration manager + model benchmark suite for Hermes Agent.

**Config Survival** — Detects drift from declared baseline state, applies patches (static then LLM-driven), runs verification tests, and auto-reverts on failure. Survives `hermes update` via a git post-merge hook.

**Model Benchmark** — Benchmarks all models on a local inference provider. Measures prompt processing speed (TTFT), token generation speed, context window limits, reasoning token behavior, and estimated parameter count. Ranks models and recommends timeout values.

## Quick Start

```bash
cd ~/projects/Agent-Ersatz

# First run: detect local LLM provider
python shelter.py setup

# Benchmark all models on the provider
python shelter.py benchmark

# Check config state vs baseline
python shelter.py check

# Auto-heal after hermes update
python shelter.py heal
```

## Commands

| Command | Description |
|---------|-------------|
| `setup` | Configure LLM provider (auto-detects local endpoints) |
| `benchmark` | Benchmark all models, rank by speed, recommend timeouts |
| `check` | Report current config state vs baseline |
| `heal` | Detect drift, patch (static + LLM), test, keep/revert |
| `snapshot` | Capture current known-good state |
| `test` | Run verification tests |

## Structure

```
Agent-Ersatz/
├── shelter.py          # Main entry: config survival + CLI
├── benchmark.py        # Model benchmark suite
├── shelter.conf        # User-specific (gitignored): provider, model, timeouts
├── state/
│   ├── baseline.yaml   # Expected config state declarations
│   └── snapshots/      # Timestamped state captures
├── patches/            # Static patches to apply after updates
└── tests/              # Verification test scripts
```

## Auto-Detection

On first run, shelter.py discovers local LLM providers by:

1. Reading Hermes `config.yaml` custom_providers (skips cloud URLs)
2. Probing known ports: LM Studio (1234), Ollama (11434), vLLM (8000), SGLang (30000), TabbyAPI (5000), koboldcpp (5001)
3. Offering interactive selection if multiple found

No provider URL or model name is hardcoded in the repo.

## Post-Merge Hook

The git post-merge hook at `~/.hermes/hermes-agent/.git/hooks/post-merge` calls `shelter.py heal` automatically after `hermes update`.

## Benchmark Details

The benchmark measures for each model:

- **Prompt Processing Speed** (tokens/s) — how fast it ingests context
- **Time to First Token** (TTFT, seconds) — the wait before generation starts
- **Token Generation Speed** (tokens/s) — generation throughput
- **Reasoning Detection** — whether the model emits reasoning/thinking tokens
- **Context Window** — maximum context size discovered from API or probing
- **Estimated Parameters** — from model name heuristics
- **Recommended Timeout** — calculated from TTFT + context scaling

Results are saved to `benchmark-results.json` and printed as a ranked table.

Options:
- `--skip-reasoning` — exclude models that use reasoning tokens
- `--quick` — fast benchmark (1 run, shorter prompts)
- `--full` — thorough benchmark (3 runs, multiple context sizes)
- `--save` — save results to file
- `--recommend-timeouts` — output shelter.conf timeout recommendations
