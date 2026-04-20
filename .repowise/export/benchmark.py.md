# File: benchmark.py



# Agent-Ersatz Benchmark (`benchmark.py`)

## Overview

**Agent-Ersatz Benchmark** is a specialized performance profiling tool designed for local Large Language Model (LLM) providers compatible with the OpenAI API standard. It serves as a diagnostic and ranking utility, measuring speed and capability metrics to help users select or configure models effectively.

This module acts as a **HOTSPOT**—it exhibits high complexity and frequent change frequency. Careful maintenance is required due to its reliance on heuristic parsing for model metadata and complex streaming response handling.

### Key Capabilities
*   **Performance Metrics:** Measures Prompt Processing speed (tokens/sec), Time To First Token (TTFT), and Token Generation speed.
*   **Reasoning Detection:** Identifies if a model emits `reasoning_content` or `reasoning` fields during generation.
*   **Context Window Probing:** Estimates the maximum context window size by testing increasing token counts.
*   **Parameter Estimation:** Heuristically estimates parameter counts (e.g., 7B, 26B) from model names using regex patterns.
*   **Gateway Configuration:** Generates recommended timeout values for reverse proxies or gateways based on worst-case latency extrapolation.

## Public API

The module exposes a comprehensive set of functions and classes for configuration, discovery, benchmarking, and reporting.

### Configuration & Discovery

#### `load_conf() -> dict`
Loads provider configuration from either `shelter.conf` (primary) or `~/.hermes/config.yaml` (fallback). Returns a dictionary containing `llm_url` and `llm_model`.

#### `get_base_url() -> str`
Retrieves the configured base URL for the LLM provider. Exits with an error if no provider is configured.

#### `list_models(base_url: str) -> list[str]`
Performs an HTTP GET request to `{base_url}/models` and returns a sorted list of available model IDs. Handles connection errors gracefully by returning an empty list.

### Heuristics & Utilities

#### `estimate_params(model_name: str) -> Optional[float]`
Estimates the number of parameters (in billions) from a model string.
*   **Logic:** Uses regex to match patterns like `26b`, `1.2b`, `70B`, or written-out numbers (`4 billion`). Also handles MoE active parameter notation (e.g., `a3b`).

#### `is_reasoning_model_name(model_name: str) -> bool`
Determines if a model name suggests it is a "thinking" or reasoning model.
*   **Patterns:** Checks for substrings like `qwq`, `deepseek-r1`, `o1`, `reason`, `think`, `GRPO`.

### Benchmarking Core

#### `BenchmarkResult` (Dataclass)
A structured container for benchmark metrics. Key fields include:
*   `pp_tokens_per_sec`: Prompt processing speed.
*   `ttft_seconds`: Time to first content token.
*   `tg_tokens_per_sec`: Token generation speed.
*   `peak_tg_tokens_per_sec`: Peak generation speed in a 1-second window.
*   `has_reasoning_tokens`: Boolean indicating if reasoning output was detected.

#### `benchmark_single(base_url: str, model: str, pp_tokens=1024, gen_tokens=64, runs=1, probe_context_sizes=None, timeout=300) -> BenchmarkResult`
Executes the benchmark loop for a single model.
*   **Mechanism:** Uses HTTP streaming to capture token timestamps. Calculates TTFT based on the first content chunk arrival relative to request start. Calculates generation speed based on inter-token latency.
*   **Context Probing:** If `probe_context_sizes` is provided, it iteratively tests if the model can handle specific context lengths by attempting to stream a single token back.

#### `_build_prompt(token_target: int) -> str`
Generates a prompt of approximately `token_target` length using repeated English text about computing history (~4 chars per token). Used for consistent benchmarking inputs.

#### `_probe_context(base_url: str, model: str, token_count: int, timeout: int) -> bool`
Tests if a model can accept a specific context size. Sends a request with the generated prompt and `max_tokens=1`. Returns `True` if the response is successful (HTTP 200).

### Reporting & Recommendations

#### `recommend_timeout(result: BenchmarkResult, safety_factor=2.5) -> dict`
Calculates recommended timeout values for gateway configurations.
*   **Logic:** Extrapolates TTFT linearly based on prompt size to estimate worst-case latency at the model's max context window (assumed 32k if unknown). Adds estimated generation time and applies a safety factor.

#### `display_results(results: list[BenchmarkResult], show_timeouts=False)`
Prints a formatted table ranking models by generation speed. Includes columns for parameters, reasoning detection, PP/TG speeds, and optional timeout recommendations.

### Entry Point

#### `run_benchmark(skip_reasoning=False, quick=False, full=False, model_filter=None, save_path=None, recommend=False)`
Orchestrates the benchmark process: loads config, lists models, filters based on arguments, runs benchmarks, displays results, and saves JSON output.

#### `main()`
CLI entry point using `argparse`. Supports flags:
*   `--quick`: Fast mode (1 run, short prompts).
*   `--full`: Thorough mode (3 runs, context probing).
*   `--skip-reasoning`: Exclude reasoning models from results.
*   `--model <name>`: Benchmark a specific model only.
*   `--recommend-timeouts`: Include timeout suggestions in output.

## Dependencies

This module relies on standard library modules and two external packages for HTTP handling and configuration parsing.

### External Libraries
*   **`httpx`**: Used for making asynchronous-compatible HTTP requests, specifically leveraging streaming responses (`httpx.stream`) to capture token-by-token timing data.
*   **`yaml`**: Used via `safe_load` to parse configuration files (`shelter.conf`, `config.yaml`).

### Standard Library Modules
*   **`json`**: Parsing streaming chunks and saving results.
*   **`re`**: Regex patterns for estimating parameter counts and detecting reasoning model names.
*   **`time`**: High-resolution timing (`perf_counter`) for precise speed measurement.
*   **`argparse`**: Command-line argument parsing.
*   **`dataclasses`**: Structuring benchmark results (`BenchmarkResult`).
*   **`pathlib`**: File path manipulation for config loading and result saving.

## Usage Notes

### Configuration Requirements
The tool requires a configured LLM provider endpoint. It checks for configuration in the following order:
1.  `shelter.conf` (in the script's directory).
2.  `~/.hermes/config.yaml` (fallback, looking under `custom_providers`).

If no URL is found, `get_base_url()` will exit with an error message prompting the user to run setup.

### Benchmark Modes
*   **Default:** Runs 1 iteration with 1024 prompt tokens and 64 generation tokens. Probes context sizes at 4k and 16k.
*   **`--quick`:** Reduces overhead. 512 prompt tokens, 32 gen tokens, 1 run. No context probing.
*   **`--full`:** High fidelity. 2048 prompt tokens, 128 gen tokens, 3 runs (averaged). Probes context sizes at 4k, 8k, 16k, and 32k.

### Timeout Recommendation Logic
The `recommend_timeout` function is critical for gateway configuration (e.g., Traefik, Nginx, or custom proxies). It assumes:
*   **TTFT Scaling:** TTFT scales linearly with prompt size.
*   **Generation Time:** Scales linearly with output length (assumed max 4096 tokens).
*   **Safety Factor:** Defaults to 2.5x the calculated worst-case time to prevent premature gateway timeouts during long generations or large context processing.

### Edge Cases & Pitfalls
1.  **Parameter Estimation:** `estimate_params` relies on naming conventions (e.g., `gemma-4-26b`). If a model name deviates significantly from standard patterns (like `70B` or `1.2b`), it may return `None`.
2.  **Reasoning Detection:** The tool detects reasoning models by checking for the presence of `reasoning_content` in the stream delta. It also uses heuristics on model names (`is_reasoning_model_name`). Be aware that some non-reasoning models might emit these fields, or vice versa.
3.  **Streaming Parsing:** The benchmark relies on parsing SSE (Server-Sent Events) lines starting with `data:`. If the provider returns non-standard JSON or error messages in the stream, the parser may skip chunks or fail silently if not handled by the outer try/except blocks.
4.  **Context Probing:** `_probe_context` sends a request and drains it immediately (`for line in resp.iter_lines(): pass`). This is efficient but relies on the server accepting the context size without generating output to confirm validity.