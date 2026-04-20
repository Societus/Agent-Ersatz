#!/home/martin/.hermes/hermes-agent/venv/bin/python
"""
Agent-Ersatz Benchmark — model speed and capability ranking for local LLM providers.

Benchmarks all models on an OpenAI-compatible endpoint. Measures:
  - Prompt processing speed (tokens/s via prefill timing)
  - Time to first token (TTFT, seconds)
  - Token generation speed (tokens/s)
  - Reasoning token detection (does the model emit reasoning_content?)
  - Context window estimation
  - Estimated parameter count (from model name heuristic)

Outputs a ranked table and optional JSON results file.

Usage:
    python benchmark.py                          # benchmark all models
    python benchmark.py --quick                  # fast mode (1 run)
    python benchmark.py --skip-reasoning         # exclude reasoning models
    python benchmark.py --recommend-timeouts     # output timeout suggestions
    python benchmark.py --model <name>           # benchmark one model only
"""

import json
import re
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx
import yaml

# ─── Paths ──────────────────────────────────────────────────────────────────

SHELTER_DIR = Path(__file__).resolve().parent
SHELTER_CONF = SHELTER_DIR / "shelter.conf"
HERMES_HOME = Path.home() / ".hermes"
HERMES_CONFIG = HERMES_HOME / "config.yaml"


# ─── Config Loading (mirrors shelter.py) ────────────────────────────────────

def load_conf() -> dict:
    """Load shelter.conf for provider URL."""
    if SHELTER_CONF.exists():
        with open(SHELTER_CONF) as f:
            return yaml.safe_load(f) or {}
    # Fallback: read from Hermes config
    if HERMES_CONFIG.exists():
        with open(HERMES_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        for p in cfg.get("custom_providers", []):
            url = p.get("base_url", "")
            if url and "localhost" in url or "127.0.0" in url or "http://" in url:
                model = p.get("model", "")
                if model:
                    return {"llm_url": url.rstrip("/"), "llm_model": model}
    return {}


def get_base_url() -> str:
    conf = load_conf()
    url = conf.get("llm_url", "")
    if not url:
        print("No provider configured. Run: python shelter.py setup")
        sys.exit(1)
    return url


# ─── Model Discovery ───────────────────────────────────────────────────────

def list_models(base_url: str) -> list[str]:
    """List all models from the provider."""
    try:
        resp = httpx.get(f"{base_url}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return sorted(m["id"] for m in data.get("data", []))
    except Exception as e:
        print(f"Failed to list models: {e}")
        return []


def estimate_params(model_name: str) -> Optional[float]:
    """Estimate parameter count from model name (e.g. 'gemma-4-26b' -> 26)."""
    # Match patterns like 26b, 1.2b, 0.5b, 70B, 405b
    m = re.search(r"(\d+\.?\d*)\s*[bB](?:\s|$|[-_/])", model_name)
    if m:
        return float(m.group(1))
    # Match A3b style (active params in MoE)
    m = re.search(r"a(\d+\.?\d*)[bB]", model_name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Match billions written out
    m = re.search(r"(\d+\.?\d*)\s*(?:billion|B)(?:\s|$|[-_/])", model_name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def is_reasoning_model_name(model_name: str) -> bool:
    """Heuristic: does the model name suggest it's a reasoning/thinking model?"""
    patterns = [
        r"qwq", r"deepseek-r1", r"o1", r"o3", r"reason", r"think",
        r"r1-", r"-r1\b", r" GRPO", r"qwopus", r"marco-o1",
    ]
    name_lower = model_name.lower()
    return any(re.search(p, name_lower) for p in patterns)


# ─── Benchmark Core ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    model: str
    params_b: Optional[float] = None
    is_reasoning: Optional[bool] = None
    has_reasoning_tokens: Optional[bool] = None

    # Prompt processing
    pp_tokens_per_sec: Optional[float] = None  # prefill speed
    ttft_seconds: Optional[float] = None       # time to first content token
    prompt_tokens: int = 0

    # Token generation
    tg_tokens_per_sec: Optional[float] = None  # generation speed
    peak_tg_tokens_per_sec: Optional[float] = None
    total_gen_tokens: int = 0
    gen_time_seconds: float = 0.0

    # Context probing
    max_context_tested: int = 0
    context_ok: bool = True
    error: Optional[str] = None

    # Prompt chain estimation (5-turn agent conversation)
    chain_5turn_seconds: Optional[float] = None
    chain_10turn_seconds: Optional[float] = None

    # Output quality (1-10)
    quality_score: Optional[float] = None
    quality_response: Optional[str] = None
    quality_details: Optional[dict] = None


def _build_prompt(token_target: int) -> str:
    """Build a ~token_target length prompt from repeated known text."""
    # Use a simple repeating pattern. ~4 chars per token for English.
    char_target = token_target * 4
    base = (
        "The history of computing is a story of abstraction layers. "
        "Each generation builds tools that hide the complexity of the layer below. "
        "Machine code gave way to assembly, assembly to FORTRAN, FORTRAN to C, "
        "C to Python, and now Python to natural language prompts. "
        "At each step, more people could instruct machines without understanding "
        "the underlying mechanism. The trend is clear: the interface becomes "
        "more human, the machine does more translation work. "
    )
    repeats = max(1, char_target // len(base))
    text = (base * repeats)[:char_target]
    return text


def benchmark_single(
    base_url: str,
    model: str,
    pp_tokens: int = 1024,
    gen_tokens: int = 64,
    runs: int = 1,
    probe_context_sizes: list[int] = None,
    timeout: int = 300,
) -> BenchmarkResult:
    """Run benchmark for a single model."""

    result = BenchmarkResult(
        model=model,
        params_b=estimate_params(model),
        is_reasoning=is_reasoning_model_name(model),
    )

    prompt_text = _build_prompt(pp_tokens)

    ttft_samples = []
    pp_speed_samples = []
    tg_speed_samples = []
    peak_tg_samples = []
    total_gen = 0
    total_gen_time = 0.0
    detected_reasoning = False

    for run_idx in range(runs):
        try:
            start = time.perf_counter()
            first_content_ts = None
            first_chunk_ts = None
            token_timestamps = []
            prompt_tok_count = 0
            gen_tok_count = 0
            has_reasoning_content = False

            with httpx.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": gen_tokens,
                    "temperature": 0.1,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    result.context_ok = False
                    return result

                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload in ("[DONE]", ""):
                        continue

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    now = time.perf_counter()

                    # Track prompt tokens from usage
                    if "usage" in chunk and chunk["usage"]:
                        prompt_tok_count = chunk["usage"].get("prompt_tokens", prompt_tok_count)

                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})

                        # Check for reasoning content
                        rc = delta.get("reasoning_content") or delta.get("reasoning")
                        if rc:
                            has_reasoning_content = True
                            if first_content_ts is None:
                                first_content_ts = now
                            gen_tok_count += 1
                            token_timestamps.append(now)

                        # Regular content
                        content = delta.get("content")
                        if content:
                            if first_content_ts is None:
                                first_content_ts = now
                            gen_tok_count += 1
                            token_timestamps.append(now)

                end = time.perf_counter()

            # Calculate metrics for this run
            if first_content_ts and gen_tok_count > 1:
                ttft = first_content_ts - start
                ttft_samples.append(ttft)

                # Prompt processing speed = prompt_tokens / TTFT
                if prompt_tok_count > 0 and ttft > 0:
                    pp_speed_samples.append(prompt_tok_count / ttft)

                # Generation speed = (tokens - 1) / (last_token - first_token)
                gen_time = token_timestamps[-1] - token_timestamps[0]
                if gen_time > 0 and gen_tok_count > 1:
                    speed = (gen_tok_count - 1) / gen_time
                    tg_speed_samples.append(speed)
                    total_gen += gen_tok_count
                    total_gen_time += gen_time

                    # Peak speed: best 1-second window
                    if len(token_timestamps) > 2:
                        best_peak = 0
                        for i in range(len(token_timestamps)):
                            # Count tokens in the 1s window starting at token i
                            t_start = token_timestamps[i]
                            t_end = t_start + 1.0
                            count = sum(1 for t in token_timestamps[i:] if t <= t_end)
                            if count > best_peak:
                                best_peak = count
                        peak_tg_samples.append(float(best_peak))

            if has_reasoning_content:
                detected_reasoning = True

        except httpx.TimeoutException:
            result.error = "timeout"
            result.context_ok = False
            break
        except Exception as e:
            result.error = str(e)[:200]
            result.context_ok = False
            break

    # Aggregate results
    if ttft_samples:
        result.ttft_seconds = sum(ttft_samples) / len(ttft_samples)
    if pp_speed_samples:
        result.pp_tokens_per_sec = sum(pp_speed_samples) / len(pp_speed_samples)
    if tg_speed_samples:
        result.tg_tokens_per_sec = sum(tg_speed_samples) / len(tg_speed_samples)
    if peak_tg_samples:
        result.peak_tg_tokens_per_sec = sum(peak_tg_samples) / len(peak_tg_samples)
    result.total_gen_tokens = total_gen
    result.gen_time_seconds = total_gen_time
    result.has_reasoning_tokens = detected_reasoning
    result.prompt_tokens = prompt_tok_count

    # Context probing
    if probe_context_sizes:
        max_ok = pp_tokens
        for ctx_size in sorted(probe_context_sizes):
            ok = _probe_context(base_url, model, ctx_size, timeout)
            if ok:
                max_ok = ctx_size
                result.max_context_tested = ctx_size
                result.context_ok = True
            else:
                result.context_ok = False
                break
        if result.max_context_tested == 0:
            result.max_context_tested = max_ok

    return result


def _probe_context(base_url: str, model: str, token_count: int, timeout: int) -> bool:
    """Test if the model can handle a given context size."""
    prompt = _build_prompt(token_count)
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": True,
            },
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                return False
            for line in resp.iter_lines():
                pass  # drain the response
        return True
    except (httpx.TimeoutException, Exception):
        return False


# ─── Quality Scoring ──────────────────────────────────────────────────────

# A structured prompt that tests instruction following, reasoning, and formatting.
# The model must: follow format rules, solve a logic problem, write a function,
# and produce valid structured output.
QUALITY_PROMPT = """\
You are being evaluated. Respond to ALL four tasks below. Follow every instruction precisely.

Task 1 — Sort these numbers in descending order, as a JSON array: [42, 7, 13, 99, 3, 28]

Task 2 — A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer with just the dollar amount.

Task 3 — Write a Python function called "merge_sorted" that takes two sorted lists and returns a single merged sorted list. No imports allowed. Include a docstring.

Task 4 — In exactly 3 bullet points, summarize what makes a good API design.

Format your entire response as:
=== TASK 1 ===
(your answer)
=== TASK 2 ===
(your answer)
=== TASK 3 ===
(your answer)
=== TASK 4 ===
(your answer)
"""

def _score_quality(response_text: str) -> tuple[float, dict]:
    """Score a model's response to the quality prompt on a 1-10 scale.

    Returns (score, details_dict).
    Scoring rubric (10 points total):
      - Task 1: JSON array, correct order, all 6 numbers (2.5 pts)
      - Task 2: Correct answer $0.05 (1.5 pts)
      - Task 3: Valid function with correct name, docstring, no imports (3 pts)
      - Task 4: Exactly 3 bullet points (2 pts)
      - Overall format: Used the === separators correctly (1 pt)
    """
    text = response_text.strip()
    details = {}
    total = 0.0

    # ── Task 1: Sort descending ──────────────────────────────────────────
    t1_score = 0.0
    # Find the task 1 section
    t1 = _extract_task(text, 1)
    if t1:
        # Check for JSON array
        import ast
        try:
            # Try to find a JSON array in the response
            arr_match = re.search(r'\[[\d,\s.]+\]', t1)
            if arr_match:
                parsed = json.loads(arr_match.group())
                expected = [99, 42, 28, 13, 7, 3]
                if parsed == expected:
                    t1_score = 2.5
                elif sorted(parsed, reverse=True) == expected:
                    # Correct values but wrong order
                    t1_score = 1.0
                elif set(parsed) == set(expected):
                    # Has all numbers but wrong order
                    t1_score = 0.75
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: check if all numbers appear in roughly descending order
        if t1_score == 0:
            numbers = re.findall(r'\b(\d+)\b', t1)
            if set(int(n) for n in numbers) >= {3, 7, 13, 28, 42, 99}:
                t1_score = 0.5
    details["task1_sort"] = t1_score
    total += t1_score

    # ── Task 2: Bat and ball ─────────────────────────────────────────────
    t2_score = 0.0
    t2 = _extract_task(text, 2)
    if t2:
        # The correct answer is $0.05
        if re.search(r'0\.05', t2):
            t2_score = 1.5
        elif re.search(r'5\s*cents?', t2, re.IGNORECASE):
            t2_score = 1.5
        elif re.search(r'5\b', t2):
            t2_score = 1.0
        elif re.search(r'0\.10|10\s*cents?', t2, re.IGNORECASE):
            # Common wrong answer (the whole $1.10 trap)
            t2_score = 0.25
    details["task2_logic"] = t2_score
    total += t2_score

    # ── Task 3: merge_sorted function ────────────────────────────────────
    t3_score = 0.0
    t3 = _extract_task(text, 3)
    if t3:
        has_def = bool(re.search(r'def\s+merge_sorted', t3))
        has_docstring = bool(re.search(r'"""|\'\'\'', t3))
        has_return = bool(re.search(r'return', t3))
        has_import = bool(re.search(r'import\s', t3))
        has_two_lists = len(re.findall(r'(?:for|in|append|len|range)', t3)) >= 1

        if has_def:
            t3_score += 1.0  # correct function name
        if has_docstring:
            t3_score += 0.5
        if has_return:
            t3_score += 0.5
        if has_two_lists:
            t3_score += 0.5  # some actual logic
        if has_import:
            t3_score -= 0.5  # penalty for imports
        t3_score = max(0, t3_score)

        # Bonus: check if it looks like it would actually work
        if t3_score >= 2.0 and re.search(r'while.*(?:i|j|a|b)\s*[<>=]', t3):
            t3_score = min(3.0, t3_score + 0.5)  # has merge loop logic
    details["task3_code"] = t3_score
    total += t3_score

    # ── Task 4: 3 bullet points ──────────────────────────────────────────
    t4_score = 0.0
    t4 = _extract_task(text, 4)
    if t4:
        bullets = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+[.)])\s', t4)
        if len(bullets) == 3:
            t4_score = 2.0
        elif len(bullets) >= 2:
            t4_score = 1.0
        elif len(bullets) >= 1:
            t4_score = 0.5
        # Partial credit if it wrote 3 things even without bullets
        if t4_score == 0 and len(t4.strip().split('\n')) >= 3:
            t4_score = 0.5
    details["task4_format"] = t4_score
    total += t4_score

    # ── Overall format: separators ───────────────────────────────────────
    fmt_score = 0.0
    found_separators = len(re.findall(r'===\s*TASK\s+\d\s*===', text, re.IGNORECASE))
    if found_separators >= 4:
        fmt_score = 1.0
    elif found_separators >= 2:
        fmt_score = 0.5
    details["format_separators"] = fmt_score
    total += fmt_score

    details["total"] = round(total, 1)
    return round(total, 1), details


def _extract_task(text: str, task_num: int) -> str:
    """Extract the text between === TASK N === and === TASK N+1 === (or end)."""
    pattern = rf'===\s*TASK\s+{task_num}\s*===(.*?)(?====\s*TASK\s+{task_num+1}\s*===|$)'
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if no separators at all, return the whole text
    return text


def benchmark_quality(
    base_url: str,
    model: str,
    timeout: int = 300,
) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    """Run the quality evaluation prompt and score the response."""
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": QUALITY_PROMPT}],
                "max_tokens": 2048,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        score, details = _score_quality(content)
        return score, content, details
    except Exception as e:
        return None, None, {"error": str(e)[:200]}


# ─── Prompt Chain Estimation ─────────────────────────────────────────────

def _estimate_chain_time(result: BenchmarkResult, turns: int = 5) -> Optional[float]:
    """Estimate wall-clock time for a multi-turn agent conversation.

    Models a realistic agent prompt chain:
    - System prompt: ~500 tokens (present every turn)
    - Turn 1: User message ~200 tokens, response ~300 tokens
    - Turn N: Growing context from prior turns + tool result ~400 tokens,
              response ~400 tokens

    The context accumulates: each turn adds prior messages to the prompt.
    """
    if not result.pp_tokens_per_sec or not result.tg_tokens_per_sec:
        if not result.ttft_seconds or not result.prompt_tokens:
            return None
        # Rough fallback: use TTFT directly
        pp_speed = result.prompt_tokens / result.ttft_seconds
        tg_speed = result.tg_tokens_per_sec or 10.0
    else:
        pp_speed = result.pp_tokens_per_sec
        tg_speed = result.tg_tokens_per_sec

    system_prompt = 500
    total_time = 0.0
    accumulated_context = system_prompt

    for turn in range(1, turns + 1):
        # Each turn adds a user message and tool result to context
        user_msg = 200
        tool_result = 400
        expected_response = 350  # average response length

        # This turn's prompt = everything accumulated so far + new input
        prompt_tokens = accumulated_context + user_msg + tool_result

        # Time = prompt processing + generation
        pp_time = prompt_tokens / pp_speed
        gen_time = expected_response / tg_speed
        total_time += pp_time + gen_time

        # Context grows by the exchange (user + assistant + tool messages)
        accumulated_context += user_msg + expected_response + tool_result

    return round(total_time, 1)


# ─── Timeout Recommendation ─────────────────────────────────────────────────

def recommend_timeout(result: BenchmarkResult, safety_factor: float = 2.5) -> dict:
    """Calculate recommended timeout values based on benchmark results.

    The key insight: TTFT scales roughly linearly with prompt size, so
    we can extrapolate from the measured TTFT at pp_tokens to estimate
    TTFT at the model's max context.
    """
    if not result.ttft_seconds or not result.prompt_tokens:
        return {
            "gateway_timeout": 7200,
            "aux_timeout": 600,
            "note": "insufficient data, using safe defaults",
        }

    ttft_per_token = result.ttft_seconds / result.prompt_tokens

    # Estimate TTFT at max context (conservative: assume max context is 32k if unknown)
    max_ctx = result.max_context_tested if result.max_context_tested > 0 else 32768
    est_max_ttft = ttft_per_token * max_ctx

    # Generation time estimate: assume max output of 4096 tokens
    gen_speed = result.tg_tokens_per_sec or 10  # conservative fallback
    est_gen_time = 4096 / gen_speed

    # Total estimated worst-case call time
    worst_case = est_max_ttft + est_gen_time

    # Apply safety factor
    recommended_gateway = int(worst_case * safety_factor)
    recommended_aux = int(est_max_ttft * safety_factor)

    # Floor values
    recommended_gateway = max(recommended_gateway, 1800)
    recommended_aux = max(recommended_aux, 300)

    return {
        "gateway_timeout": recommended_gateway,
        "aux_timeout": recommended_aux,
        "ttft_per_token_ms": round(ttft_per_token * 1000, 2),
        "est_max_ttft_s": round(est_max_ttft, 1),
        "est_gen_time_s": round(est_gen_time, 1),
        "worst_case_s": round(worst_case, 1),
        "safety_factor": safety_factor,
        "note": f"based on TTFT={result.ttft_seconds:.1f}s at {result.prompt_tokens} tokens",
    }


# ─── Display ────────────────────────────────────────────────────────────────

def display_results(results: list[BenchmarkResult], show_timeouts: bool = False):
    """Print a ranked results table."""

    # Sort: fastest generation speed first, successful ones at top
    good = [r for r in results if r.tg_tokens_per_sec is not None]
    failed = [r for r in results if r.tg_tokens_per_sec is None]
    good.sort(key=lambda r: r.tg_tokens_per_sec, reverse=True)

    if not good and not failed:
        print("No results to display.")
        return

    # Header
    print(f"\n{'Model':<30} {'Params':>7} {'R':>2} {'PP t/s':>8} {'TTFT s':>8} "
          f"{'TG t/s':>8} {'Peak':>6} {'Quality':>7} {'5-chain':>8} {'10-chain':>9}")
    print("─" * 110)

    for r in good:
        params_str = f"{r.params_b:.0f}B" if r.params_b else "?"
        reasoning_str = "Y" if r.has_reasoning_tokens else "-"
        is_r_marker = "*" if r.is_reasoning else " "

        pp_str = f"{r.pp_tokens_per_sec:.0f}" if r.pp_tokens_per_sec else "-"
        ttft_str = f"{r.ttft_seconds:.1f}" if r.ttft_seconds else "-"
        tg_str = f"{r.tg_tokens_per_sec:.1f}" if r.tg_tokens_per_sec else "-"
        peak_str = f"{r.peak_tg_tokens_per_sec:.0f}" if r.peak_tg_tokens_per_sec else "-"

        # Quality score with visual indicator
        if r.quality_score is not None:
            q = r.quality_score
            if q >= 8:
                q_bar = "●"
            elif q >= 6:
                q_bar = "◐"
            elif q >= 4:
                q_bar = "○"
            else:
                q_bar = "◌"
            quality_str = f"{q_bar} {q:.1f}/10"
        else:
            quality_str = "  -"

        # Chain estimates formatted as human-readable time
        chain5_str = _format_duration(r.chain_5turn_seconds) if r.chain_5turn_seconds else "-"
        chain10_str = _format_duration(r.chain_10turn_seconds) if r.chain_10turn_seconds else "-"

        print(f"{is_r_marker}{r.model:<29} {params_str:>7} {reasoning_str:>2} "
              f"{pp_str:>8} {ttft_str:>8} {tg_str:>8} {peak_str:>6} "
              f"{quality_str:>7} {chain5_str:>8} {chain10_str:>9}")

    for r in failed:
        err = r.error or "unknown"
        print(f"  {r.model:<29} {'FAIL':>7} {'':>2} {'':>8} {'':>8} {'':>8} {'':>6} "
              f"{'':>7} {'':>8} {'':>9}  {err[:30]}")

    print()
    print("  R = reasoning tokens detected during generation")
    print("  * = model name suggests reasoning/thinking model")
    print("  PP t/s = prompt processing speed (tokens/sec)")
    print("  TTFT s = time to first content token (seconds)")
    print("  TG t/s = token generation speed (tokens/sec)")
    print("  Peak = peak generation speed in best 1s window")
    print("  Quality = output quality score from structured evaluation (●>=8 ◐>=6 ○>=4 ◌<4)")
    print("  5-chain = estimated time for a 5-turn agent conversation")
    print("  10-chain = estimated time for a 10-turn agent conversation")

    if show_timeouts:
        print()
        print("  ── Timeout Recommendations ──")
        print(f"  {'Model':<30} {'Gateway (s)':>12} {'Aux (s)':>10} {'Max TTFT est.':>14}")
        print("  " + "─" * 70)
        for r in good:
            rec = recommend_timeout(r)
            print(f"  {r.model:<30} {rec['gateway_timeout']:>12} {rec['aux_timeout']:>10} "
                  f"{rec.get('est_max_ttft_s', '?'):>13}s")


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


# ─── Main ───────────────────────────────────────────────────────────────────

def run_benchmark(
    skip_reasoning: bool = False,
    quick: bool = False,
    full: bool = False,
    model_filter: str = None,
    save_path: str = None,
    recommend: bool = False,
):
    base_url = get_base_url()
    print(f"═══ Agent-Ersatz — Model Benchmark ═══")
    print(f"  Provider: {base_url}")

    models = list_models(base_url)
    if not models:
        print("  No models found.")
        return

    if model_filter:
        models = [m for m in models if model_filter.lower() in m.lower()]
        if not models:
            print(f"  No models matching '{model_filter}'")
            return

    print(f"  Found {len(models)} model(s)")

    # Filter reasoning models if requested
    if skip_reasoning:
        reasoning_names = [m for m in models if is_reasoning_model_name(m)]
        models = [m for m in models if not is_reasoning_model_name(m)]
        if reasoning_names:
            print(f"  Skipping {len(reasoning_names)} reasoning model(s): {', '.join(reasoning_names)}")

    # Benchmark parameters
    if quick:
        pp_tokens = 512
        gen_tokens = 32
        runs = 1
        probe_ctx = None
    elif full:
        pp_tokens = 2048
        gen_tokens = 128
        runs = 3
        probe_ctx = [4096, 8192, 16384, 32768]
    else:
        pp_tokens = 1024
        gen_tokens = 64
        runs = 1
        probe_ctx = [4096, 16384]

    print(f"  Config: pp={pp_tokens} tg={gen_tokens} runs={runs}"
          + (f" ctx_probe={probe_ctx}" if probe_ctx else ""))
    print()

    results = []
    for i, model in enumerate(models, 1):
        print(f"  [{i}/{len(models)}] {model} ...", end=" ", flush=True)
        r = benchmark_single(
            base_url, model,
            pp_tokens=pp_tokens,
            gen_tokens=gen_tokens,
            runs=runs,
            probe_context_sizes=probe_ctx,
        )

        if r.error:
            print(f"FAILED ({r.error[:60]})")
        else:
            tg = f"{r.tg_tokens_per_sec:.1f} t/s" if r.tg_tokens_per_sec else "?"
            ttft = f"{r.ttft_seconds:.1f}s TTFT" if r.ttft_seconds else "? TTFT"
            print(f"OK — {tg}, {ttft}")

            # Estimate prompt chain times
            r.chain_5turn_seconds = _estimate_chain_time(r, turns=5)
            r.chain_10turn_seconds = _estimate_chain_time(r, turns=10)

            # Run quality evaluation
            print(f"  [{i}/{len(models)}] {model} quality ...", end=" ", flush=True)
            q_score, q_response, q_details = benchmark_quality(base_url, model, timeout=600)
            r.quality_score = q_score
            r.quality_response = q_response
            r.quality_details = q_details
            if q_score is not None:
                print(f"{q_score:.1f}/10")
            else:
                print("failed")

        results.append(r)

    # Display
    display_results(results, show_timeouts=recommend)

    # Save
    if save_path:
        save_file = Path(save_path)
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        save_file = SHELTER_DIR / f"benchmark-{ts}.json"

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": base_url,
        "config": {"pp_tokens": pp_tokens, "gen_tokens": gen_tokens, "runs": runs},
        "results": [],
    }
    for r in results:
        entry = asdict(r)
        if recommend:
            entry["timeout_recommendation"] = recommend_timeout(r)
        output["results"].append(entry)

    with open(save_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {save_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Agent-Ersatz Benchmark — rank local LLM models by speed"
    )
    parser.add_argument("--quick", action="store_true", help="Fast benchmark (1 run, short prompts)")
    parser.add_argument("--full", action="store_true", help="Thorough benchmark (3 runs, context probing)")
    parser.add_argument("--skip-reasoning", action="store_true", help="Exclude reasoning/thinking models")
    parser.add_argument("--model", type=str, help="Benchmark a specific model only")
    parser.add_argument("--save", type=str, help="Save results to file")
    parser.add_argument("--recommend-timeouts", action="store_true",
                        help="Include timeout recommendations in output")
    args = parser.parse_args()

    run_benchmark(
        skip_reasoning=args.skip_reasoning,
        quick=args.quick,
        full=args.full,
        model_filter=args.model,
        save_path=args.save,
        recommend=args.recommend_timeouts,
    )


if __name__ == "__main__":
    main()
