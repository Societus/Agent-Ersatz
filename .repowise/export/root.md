# Module: root



## Overview

The `root` module is a Python-based toolkit designed for managing, patching, and benchmarking local Large Language Model (LLM) providers. It consists of two primary files: `shelter.py`, which serves as the core orchestration layer for setup, file state management, git operations, and CLI command routing; and `benchmark.py`, a specialized performance profiling tool for OpenAI-compatible local LLM endpoints. 

The module exposes 31 public symbols out of 41 total, reflecting a focused API surface with a Mean PageRank of 0.0714. Primary ownership is held by the `societus` team, who have concentrated recent development activity on `shelter.py`, making it the most actively maintained component in the module.

## Public API Summary

The module's public interface is organized across two files. Symbols prefixed with `_` represent internal utilities that support core functionality but are included in the module's overall symbol count.

### `benchmark.py` — Agent-Ersatz Benchmark
Specialized diagnostics and performance profiling for local LLM providers:
- `BenchmarkResult`
- `benchmark_single`
- `display_results`
- `estimate_params`
- `get_base_url`
- `is_reasoning_model_name`
- `list_models`
- `load_conf`
- `main`
- `recommend_timeout`
- `run_benchmark`
- `_build_prompt`
- `_probe_context`

### `shelter.py` — Core Orchestration & CLI Layer
Handles environment setup, static patching, testing, snapshotting, and LLM communication:
- `apply_static_patches`
- `check_file_state`
- `cmd_benchmark`
- `cmd_check`
- `cmd_heal`
- `cmd_setup`
- `cmd_snapshot`
- `cmd_test`
- `detect_drift`
- `file_hash`
- `llm_call`
- `llm_fast`
- `llm_patch_file`
- `load_baseline`
- `load_shelter_conf`
- `log`
- `main`
- `revert_file`
- `run_tests`
- `take_snapshot`
- `_build_llm_instructions`
- `_detect_endpoints`
- `_gather_context`
- `_pick_fast_model`
- `_probe_endpoint`
- `_read_hermes_provider`
- `_resolve_dotpath`
- `_try_git_apply`

## Architecture Notes

The module follows a clear separation of concerns between orchestration and diagnostics:

1. **Orchestration Layer (`shelter.py`)**: Acts as the primary application boundary, routing CLI commands (`cmd_*`) to appropriate handlers. It manages file integrity through hashing and drift detection, applies static patches via git operations, and abstracts LLM interactions behind `llm_call` and `llm_fast`. Configuration is loaded through dedicated loaders (`load_shelter_conf`, `load_baseline`).

2. **Benchmarking Subsystem (`benchmark.py`)**: Provides a decoupled performance profiling pipeline optimized for OpenAI-compatible local providers. It handles model discovery, parameter estimation, timeout recommendations, and result visualization. The benchmark runner operates independently but can be invoked through `shelter.py`'s CLI interface.

3. **Maintenance & Evolution**: Recent commits indicate active development on `shelter.py`, suggesting ongoing improvements to patching logic, endpoint detection, and LLM instruction generation. The module's low Mean PageRank (0.0714) reflects a flat, command-driven architecture where most symbols serve as direct entry points rather than deeply nested dependencies. No external modules are imported beyond standard Python libraries and the internal cross-file references shown above.