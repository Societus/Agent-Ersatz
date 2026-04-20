#!/home/martin/.hermes/hermes-agent/venv/bin/python
"""
Agent-Ersatz — self-healing config manager + benchmark suite for Hermes Agent

Detects drift between expected and actual state of Hermes config files,
uses the local LLM to generate targeted edits, tests the result, and
auto-reverts on failure.

Usage:
    python shelter.py check     # Report current state vs baseline
    python shelter.py heal      # Detect drift, patch, test, keep/revert
    python shelter.py snapshot  # Capture current known-good state
    python shelter.py test      # Run tests against current state only
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

# ─── Paths ──────────────────────────────────────────────────────────────────

SHELTER_DIR = Path(__file__).resolve().parent
BASELINE_FILE = SHELTER_DIR / "state" / "baseline.yaml"
SNAPSHOTS_DIR = SHELTER_DIR / "state" / "snapshots"
PATCHES_DIR = SHELTER_DIR / "patches"
TESTS_DIR = SHELTER_DIR / "tests"
SHELTER_CONF = SHELTER_DIR / "shelter.conf"
LOG_FILE = SHELTER_DIR / "shelter.log"

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_CONFIG = HERMES_HOME / "config.yaml"
HERMES_REPO = HERMES_HOME / "hermes-agent"
WEB_TOOLS_PY = HERMES_REPO / "tools" / "web_tools.py"
TOOLSETS_PY = HERMES_REPO / "toolsets.py"

# ─── LLM Provider Auto-Detection ──────────────────────────────────────────

# Common local inference server ports and their /v1 endpoints
KNOWN_ENDPOINTS = [
    {"name": "LM Studio",    "url": "http://localhost:1234/v1",  "probe": "/v1/models"},
    {"name": "Ollama",       "url": "http://localhost:11434/v1", "probe": "/v1/models"},
    {"name": "vLLM",         "url": "http://localhost:8000/v1",  "probe": "/v1/models"},
    {"name": "SGLang",       "url": "http://localhost:30000/v1", "probe": "/v1/models"},
    {"name": "TabbyAPI",     "url": "http://localhost:5000/v1",  "probe": "/v1/models"},
    {"name": "koboldcpp",    "url": "http://localhost:5001/v1",  "probe": "/v1/models"},
]

# shelter.conf keys that matter (no hardcoded defaults for model/url)
_CONF_KEYS = ["llm_url", "llm_model", "fast_model", "llm_timeout", "fast_timeout"]


def _probe_endpoint(base_url: str, probe_path: str, timeout: float = 3.0) -> list | None:
    """Try to reach an OpenAI-compatible /v1/models endpoint. Return model list or None."""
    try:
        url = base_url.rstrip("/") + probe_path.replace("/v1", "", 1)
        if not url.startswith("http"):
            url = base_url.rstrip("/") + probe_path
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
    except (httpx.ConnectError, httpx.TimeoutException, Exception):
        pass
    return None


def _read_hermes_provider() -> dict | None:
    """Read local model/provider from Hermes config.yaml.

    Checks custom_providers first (user-defined local servers), then the
    main model section. Skips any provider with a cloud-like URL.
    """
    if not HERMES_CONFIG.exists():
        return None
    try:
        with open(HERMES_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}

        # Cloud URL patterns to skip
        cloud_patterns = [
            "api.openai.com", "openrouter.ai", "api.anthropic.com",
            "api.z.ai", "groq.cloud", "mistral.ai", "deepseek.com",
            "together.ai", "fireworks.ai", "replicate.com",
        ]

        # Check custom_providers — prefer local ones
        providers = cfg.get("custom_providers", [])
        for p in providers:
            url = p.get("base_url", "")
            if not url:
                continue
            # Skip cloud URLs
            if any(pat in url for pat in cloud_patterns):
                continue
            model_name = p.get("model", "")
            if not model_name:
                # Try probing the URL to find models
                models = _probe_endpoint(url, "/v1/models", timeout=3)
                if models:
                    model_name = models[0]
            if model_name:
                return {"llm_url": url.rstrip("/"), "llm_model": model_name}

        # Check main model section
        model_cfg = cfg.get("model", {})
        base_url = model_cfg.get("base_url", "")
        model_name = model_cfg.get("default", "")
        if base_url and not any(pat in base_url for pat in cloud_patterns):
            return {"llm_url": base_url.rstrip("/"), "llm_model": model_name}

    except Exception:
        pass
    return None


def _detect_endpoints() -> list[dict]:
    """Probe all known local endpoints + Hermes custom_providers.

    Returns those that respond with model info.
    """
    found = []
    seen_urls = set()

    # Start with known endpoints (localhost ports)
    for ep in KNOWN_ENDPOINTS:
        models = _probe_endpoint(ep["url"], ep["probe"])
        if models is not None:
            found.append({**ep, "models": models})
            seen_urls.add(ep["url"].rstrip("/"))

    # Also check Hermes custom_providers for additional local servers
    if HERMES_CONFIG.exists():
        try:
            with open(HERMES_CONFIG) as f:
                cfg = yaml.safe_load(f) or {}
            for p in cfg.get("custom_providers", []):
                url = p.get("base_url", "").rstrip("/")
                if not url or url in seen_urls:
                    continue
                models = _probe_endpoint(url, "/v1/models", timeout=3)
                if models is not None:
                    name = p.get("name", p.get("provider", url))
                    found.append({
                        "name": name,
                        "url": url,
                        "probe": "/v1/models",
                        "models": models,
                    })
                    seen_urls.add(url)
        except Exception:
            pass

    return found


def _pick_fast_model(models: list[str]) -> str | None:
    """Heuristic: pick the smallest model as the fast model."""
    if not models:
        return None
    # Prefer models with size hints (e.g. "1.2b", "0.5b", "3b")
    import re
    sized = []
    for m in models:
        match = re.search(r"(\d+\.?\d*)[bB]", m)
        if match:
            sized.append((float(match.group(1)), m))
    if sized:
        sized.sort(key=lambda x: x[0])
        return sized[0][1]
    # Fallback: first model
    return models[0]


def cmd_setup():
    """Interactive first-run setup: detect local LLM and configure shelter."""
    print("═══ Agent-Ersatz — Setup ═══\n")

    conf = {}
    changed = False

    # Step 1: Check if shelter.conf already has answers
    if SHELTER_CONF.exists():
        with open(SHELTER_CONF) as f:
            conf = yaml.safe_load(f) or {}
        if conf.get("llm_url") and conf.get("llm_model"):
            print(f"  Current config: {conf['llm_url']} with {conf['llm_model']}")
            answer = input("  Reconfigure? [y/N] ").strip().lower()
            if answer != "y":
                print("  Keeping existing config.")
                return

    # Step 2: Try reading from Hermes config first
    hermes_provider = _read_hermes_provider()
    if hermes_provider:
        print(f"  Found Hermes provider: {hermes_provider['llm_url']}")
        print(f"  Model: {hermes_provider['llm_model']}")
        answer = input("  Use this? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            conf.update(hermes_provider)
            changed = True

    # Step 3: Probe local endpoints if Hermes config didn't work
    if not conf.get("llm_url"):
        print("\n  Probing local inference servers...")
        found = _detect_endpoints()

        if not found:
            print("  No local endpoints detected.")
            print("  Please enter manually:")
            conf["llm_url"] = input("    URL [http://localhost:1234/v1]: ").strip() or "http://localhost:1234/v1"
            conf["llm_model"] = input("    Model name: ").strip()
        elif len(found) == 1:
            ep = found[0]
            print(f"  Found: {ep['name']} at {ep['url']} ({len(ep['models'])} model(s))")
            conf["llm_url"] = ep["url"]
            # Pick model
            if len(ep["models"]) == 1:
                conf["llm_model"] = ep["models"][0]
                print(f"  Model: {conf['llm_model']}")
            else:
                print("  Available models:")
                for i, m in enumerate(ep["models"], 1):
                    print(f"    {i}. {m}")
                choice = input(f"  Select [1-{len(ep['models'])}]: ").strip()
                try:
                    idx = int(choice) - 1
                    conf["llm_model"] = ep["models"][idx]
                except (ValueError, IndexError):
                    conf["llm_model"] = ep["models"][0]
            changed = True
        else:
            print(f"  Found {len(found)} endpoint(s):")
            for i, ep in enumerate(found, 1):
                print(f"    {i}. {ep['name']} ({ep['url']}) — {len(ep['models'])} model(s)")
            choice = input(f"  Select endpoint [1-{len(found)}]: ").strip()
            try:
                idx = int(choice) - 1
                ep = found[idx]
            except (ValueError, IndexError):
                ep = found[0]
            conf["llm_url"] = ep["url"]
            if len(ep["models"]) == 1:
                conf["llm_model"] = ep["models"][0]
            else:
                print(f"  Models on {ep['name']}:")
                for i, m in enumerate(ep["models"], 1):
                    print(f"    {i}. {m}")
                choice = input(f"  Select model [1-{len(ep['models'])}]: ").strip()
                try:
                    idx = int(choice) - 1
                    conf["llm_model"] = ep["models"][idx]
                except (ValueError, IndexError):
                    conf["llm_model"] = ep["models"][0]
            changed = True

    if not conf.get("llm_model"):
        conf["llm_model"] = input("  Model name: ").strip()

    # Step 4: Fast model
    print("\n  Configuring fast model (for quick decisions)...")
    # Probe chosen endpoint for all models to pick fast one
    models = _probe_endpoint(conf["llm_url"], "/v1/models", timeout=5) or []
    fast_pick = _pick_fast_model(models)
    if fast_pick:
        print(f"  Auto-detected fast model: {fast_pick}")
        answer = input(f"  Use this? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            conf["fast_model"] = fast_pick
    if not conf.get("fast_model"):
        conf["fast_model"] = input(f"  Fast model name [{conf['llm_model']}]: ").strip() or conf["llm_model"]

    # Step 5: Timeouts
    conf.setdefault("llm_timeout", 300)
    conf.setdefault("fast_timeout", 60)

    # Step 6: Validate connection
    print(f"\n  Validating: {conf['llm_url']} with {conf['llm_model']}...")
    try:
        resp = httpx.post(
            f"{conf['llm_url']}/chat/completions",
            json={"model": conf["llm_model"], "messages": [{"role": "user", "content": "Reply: OK"}], "max_tokens": 5, "temperature": 0},
            timeout=conf["llm_timeout"],
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        print(f"  ✓ Connection valid — model replied: {text.strip()[:50]}")
    except Exception as e:
        print(f"  ⚠ Connection test failed: {e}")
        print("  Saving config anyway (you can fix later with: python shelter.py setup)")

    # Step 7: Write shelter.conf
    with open(SHELTER_CONF, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)
    print(f"\n  ✓ Config saved to {SHELTER_CONF}")
    changed = True


def load_shelter_conf() -> dict:
    """Load shelter.conf. On first run with no config, trigger setup."""
    if not SHELTER_CONF.exists():
        print("  No shelter.conf found. Running first-time setup...\n")
        cmd_setup()

    if SHELTER_CONF.exists():
        with open(SHELTER_CONF) as f:
            user_conf = yaml.safe_load(f) or {}
        if user_conf.get("llm_url") and user_conf.get("llm_model"):
            return user_conf

    # Last resort: try to read from Hermes config
    hermes = _read_hermes_provider()
    if hermes:
        return {**hermes, "llm_timeout": 300, "fast_timeout": 60}

    print("  ✗ No LLM provider configured. Run: python shelter.py setup")
    sys.exit(1)


# ─── Logging ────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] [{level}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if level in ("INFO", "OK"):
        print(f"  {msg}")
    elif level == "WARN":
        print(f"  ⚠ {msg}")
    elif level == "FAIL":
        print(f"  ✗ {msg}")
    elif level == "OK":
        print(f"  ✓ {msg}")


# ─── LLM Client ─────────────────────────────────────────────────────────────

def llm_call(prompt: str, model: str = None, max_tokens: int = 4096,
             timeout: int = None) -> str:
    """Call the local LLM and return the response text."""
    conf = load_shelter_conf()
    url = conf["llm_url"]
    model = model or conf["llm_model"]
    timeout = timeout or conf["llm_timeout"]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    log(f"Calling LLM ({model}, timeout={timeout}s)...", "INFO")
    resp = httpx.post(
        f"{url}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    log(f"LLM responded ({len(content)} chars)", "INFO")
    return content


def llm_fast(prompt: str, max_tokens: int = 512) -> str:
    """Call the fast (small) model for quick decisions."""
    conf = load_shelter_conf()
    return llm_call(prompt, model=conf["fast_model"],
                    max_tokens=max_tokens, timeout=conf["fast_timeout"])


# ─── File Snapshots ─────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA256 of a file's contents."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def take_snapshot(name: str = None) -> dict:
    """Snapshot all tracked files, return {path: {hash, size}}."""
    if name is None:
        name = datetime.now(timezone.utc).strftime("snap-%Y%m%d-%H%M%S")

    baseline = load_baseline()
    tracked = baseline.get("tracked_files", [])
    # Always include core files
    core = [str(HERMES_CONFIG), str(WEB_TOOLS_PY)]
    for f in core:
        if f not in tracked:
            tracked.append(f)

    snap = {"name": name, "timestamp": datetime.now(timezone.utc).isoformat(), "files": {}}
    for fpath in tracked:
        p = Path(fpath)
        snap["files"][fpath] = {
            "hash": file_hash(p),
            "size": p.stat().st_size if p.exists() else 0,
            "exists": p.exists(),
        }

    # Save snapshot
    snap_path = SNAPSHOTS_DIR / f"{name}.yaml"
    with open(snap_path, "w") as f:
        yaml.dump(snap, f, default_flow_style=False)

    log(f"Snapshot saved: {snap_path.name}", "OK")
    return snap


# ─── Baseline ───────────────────────────────────────────────────────────────

def load_baseline() -> dict:
    """Load the baseline.yaml state declarations."""
    if not BASELINE_FILE.exists():
        return {"tracked_files": [], "checks": [], "patches": []}
    with open(BASELINE_FILE) as f:
        return yaml.safe_load(f) or {}


# ─── State Detection ────────────────────────────────────────────────────────

def check_file_state(filepath: str, expected: dict) -> dict:
    """Check a single file against expected state. Returns {ok, details}."""
    p = Path(filepath)
    result = {"path": filepath, "ok": True, "issues": []}

    if not p.exists():
        result["ok"] = False
        result["issues"].append("file missing")
        return result

    content = p.read_text()

    # Check for required content patterns
    for pattern in expected.get("must_contain", []):
        if pattern not in content:
            result["ok"] = False
            result["issues"].append(f"missing pattern: {pattern[:80]}")

    # Check for forbidden content patterns
    for pattern in expected.get("must_not_contain", []):
        if pattern in content:
            result["ok"] = False
            result["issues"].append(f"forbidden pattern present: {pattern[:80]}")

    # Check YAML values if config file
    if filepath.endswith(".yaml") and expected.get("yaml_values"):
        try:
            data = yaml.safe_load(content) or {}
            for dotpath, expected_val in expected["yaml_values"].items():
                actual = _resolve_dotpath(data, dotpath)
                if actual != expected_val:
                    result["ok"] = False
                    result["issues"].append(
                        f"config drift: {dotpath} = {actual!r}, expected {expected_val!r}"
                    )
        except yaml.YAMLError:
            result["ok"] = False
            result["issues"].append("YAML parse error")

    return result


def _resolve_dotpath(data: dict, path: str):
    """Resolve a dotted path like 'web.backend' in a nested dict."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def detect_drift() -> list:
    """Check all baseline declarations against current files. Return list of issues."""
    baseline = load_baseline()
    issues = []

    for check in baseline.get("checks", []):
        filepath = check["file"]
        # Expand ~ and handle relative paths
        p = Path(filepath).expanduser().resolve()
        result = check_file_state(str(p), check)
        if not result["ok"]:
            issues.append(result)

    return issues


# ─── Patching ───────────────────────────────────────────────────────────────

def apply_static_patches() -> list:
    """Apply all .patch files from patches/ dir. Return list of results."""
    results = []
    baseline = load_baseline()

    for patch_conf in baseline.get("patches", []):
        patch_file = PATCHES_DIR / patch_conf["file"]
        target = Path(patch_conf["target"]).expanduser().resolve()
        repo_dir = Path(patch_conf.get("repo_dir", str(HERMES_REPO))).resolve()

        if not patch_file.exists():
            log(f"Patch file not found: {patch_file}", "WARN")
            results.append({"patch": patch_conf["file"], "applied": False, "error": "file not found"})
            continue

        # Try applying with git apply
        ok = _try_git_apply(patch_file, repo_dir)
        if ok:
            log(f"Patch applied cleanly: {patch_conf['file']}", "OK")
            results.append({"patch": patch_conf["file"], "applied": True})
        else:
            log(f"Patch has conflicts: {patch_conf['file']}", "WARN")
            results.append({
                "patch": patch_conf["file"],
                "applied": False,
                "needs_llm": True,
                "target": str(target),
            })

    return results


def _try_git_apply(patch_file: Path, repo_dir: Path) -> bool:
    """Try git apply, return True if successful."""
    # First check if already applied (reverse applies cleanly)
    r = subprocess.run(
        ["git", "apply", "-R", "--check", str(patch_file)],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if r.returncode == 0:
        log(f"Already applied: {patch_file.name}", "INFO")
        return True

    # Try forward apply --check
    r = subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if r.returncode == 0:
        # Check passed, actually apply
        r = subprocess.run(
            ["git", "apply", str(patch_file)],
            cwd=repo_dir, capture_output=True, text=True,
        )
        return r.returncode == 0

    return False


# ─── LLM-Driven Patching ───────────────────────────────────────────────────

def llm_patch_file(filepath: str, instructions: str, context: str = "") -> bool:
    """Use the LLM to edit a file based on instructions. Returns True on success."""
    p = Path(filepath)
    if not p.exists():
        log(f"File not found for LLM patching: {filepath}", "FAIL")
        return False

    original = p.read_text()
    backup_path = p.with_suffix(p.suffix + ".shelter-backup")

    prompt = f"""You are a precise code editor. Your task is to modify the file below according to the instructions.

RULES:
- Output ONLY the complete modified file content, nothing else.
- Do NOT add any explanation, markdown formatting, or code fences.
- Preserve all existing structure, comments, and formatting.
- Make ONLY the changes described in the instructions.

FILE: {p.name}
TARGET LOCATION: {filepath}

INSTRUCTIONS:
{instructions}

{f"ADDITIONAL CONTEXT:{chr(10)}{context}{chr(10)}" if context else ""}

CURRENT FILE CONTENT:
```
{original}
```

OUTPUT THE COMPLETE MODIFIED FILE NOW:"""

    try:
        response = llm_call(prompt, max_tokens=min(len(original) + 4096, 32000))
    except Exception as e:
        log(f"LLM call failed: {e}", "FAIL")
        return False

    # Strip any markdown code fences the LLM might add despite instructions
    content = response.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```python or similar)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    # Backup original
    shutil.copy2(p, backup_path)

    # Write new content
    p.write_text(content)
    log(f"LLM patched {p.name} ({len(original)} -> {len(content)} chars)", "INFO")
    return True


def revert_file(filepath: str) -> bool:
    """Revert a file from its .shelter-backup."""
    p = Path(filepath)
    backup = p.with_suffix(p.suffix + ".shelter-backup")
    if backup.exists():
        shutil.copy2(backup, p)
        backup.unlink()
        log(f"Reverted {p.name} from backup", "OK")
        return True
    return False


# ─── Testing ────────────────────────────────────────────────────────────────

def run_tests() -> dict:
    """Run all test scripts. Returns {passed: int, failed: int, results: list}."""
    baseline = load_baseline()
    test_scripts = baseline.get("test_scripts", [])
    results = []
    passed = 0
    failed = 0

    for test_conf in test_scripts:
        script = TESTS_DIR / test_conf["script"]
        name = test_conf.get("name", test_conf["script"])
        timeout = test_conf.get("timeout", 30)

        if not script.exists():
            log(f"Test script not found: {script}", "WARN")
            results.append({"name": name, "passed": False, "error": "script not found"})
            failed += 1
            continue

        log(f"Running test: {name}...", "INFO")
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout,
        )

        ok = r.returncode == 0
        if ok:
            passed += 1
            log(f"Test passed: {name}", "OK")
        else:
            failed += 1
            log(f"Test failed: {name} — {r.stderr[:200]}", "FAIL")

        results.append({
            "name": name,
            "passed": ok,
            "stdout": r.stdout[:500],
            "stderr": r.stderr[:500],
        })

    return {"passed": passed, "failed": failed, "results": results}


# ─── Main Orchestration ─────────────────────────────────────────────────────

def cmd_check():
    """Report current state vs baseline."""
    print("═══ Agent-Ersatz — State Check ═══")
    baseline = load_baseline()

    if not baseline.get("checks"):
        print("  No checks defined in baseline.yaml")
        return

    issues = detect_drift()
    if not issues:
        print("  ✓ All checks pass — no drift detected")
    else:
        print(f"  ✗ {len(issues)} check(s) failing:")
        for issue in issues:
            print(f"    • {issue['path']}")
            for detail in issue["issues"]:
                print(f"      — {detail}")


def cmd_heal():
    """Detect drift, apply patches (static then LLM), test, keep/revert."""
    print("═══ Agent-Ersatz — Heal ═══")
    heal_start = time.time()

    # Step 1: Take a pre-heal snapshot
    snap = take_snapshot("pre-heal")

    # Step 2: Detect drift
    issues = detect_drift()
    if not issues:
        print("  ✓ No drift detected — nothing to heal")
        return

    print(f"  Found {len(issues)} issue(s):")
    for issue in issues:
        print(f"    • {issue['path']}: {'; '.join(issue['issues'])}")

    # Step 3: Try static patches first
    print("\n  → Applying static patches...")
    patch_results = apply_static_patches()

    # Step 4: Re-check after static patches
    remaining = detect_drift()
    if remaining:
        print(f"\n  → {len(remaining)} issue(s) remain after static patches")
        print("  → Engaging LLM for live edits...")

        baseline = load_baseline()
        for issue in remaining:
            # Build LLM instructions from the issue details
            instructions = _build_llm_instructions(issue, baseline)
            context = _gather_context(issue)

            success = llm_patch_file(issue["path"], instructions, context)
            if not success:
                log(f"LLM patching failed for {issue['path']}", "FAIL")

    # Step 5: Run tests
    print("\n  → Running verification tests...")
    test_results = run_tests()

    # Step 6: Keep or revert
    if test_results["failed"] == 0:
        print(f"\n  ✓ All {test_results['passed']} test(s) passed — keeping changes")
        take_snapshot("post-heal-ok")
        # Clean up backups
        for f in HERMES_REPO.rglob("*.shelter-backup"):
            f.unlink()
        if HERMES_CONFIG.with_suffix(".yaml.shelter-backup").exists():
            HERMES_CONFIG.with_suffix(".yaml.shelter-backup").unlink()
    else:
        print(f"\n  ✗ {test_results['failed']} test(s) failed — reverting all changes")
        # Revert all modified files
        for snap_file_data in snap.get("files", {}).items():
            filepath = Path(snap_file_data[0])
            revert_file(str(filepath))
        take_snapshot("post-heal-reverted")

    elapsed = time.time() - heal_start
    print(f"\n  Heal completed in {elapsed:.1f}s")


def _build_llm_instructions(issue: dict, baseline: dict) -> str:
    """Build LLM editing instructions from a drift issue."""
    instructions = []
    for detail in issue["issues"]:
        if "missing pattern" in detail:
            # Extract what pattern is missing and find the baseline check for context
            for check in baseline.get("checks", []):
                if check["file"] in issue["path"] or str(Path(check["file"]).expanduser()) == issue["path"]:
                    patterns = check.get("must_contain", [])
                    for p in patterns:
                        if p[:80] in detail:
                            instructions.append(f"Add code that includes this exact text/pattern: {p}")

        elif "forbidden pattern" in detail:
            instructions.append(f"Remove or replace the content matching: {detail}")

        elif "config drift" in detail:
            # Parse the drift message for expected value
            import re
            m = re.match(r"config drift: (.+?) = (.+?), expected (.+)", detail)
            if m:
                key, actual, expected = m.groups()
                instructions.append(f"Set YAML key '{key}' to {expected} (currently {actual})")

        else:
            instructions.append(f"Fix: {detail}")

    return "\n".join(instructions)


def _gather_context(issue: dict) -> str:
    """Gather surrounding context for the LLM from the affected file."""
    p = Path(issue["path"])
    if not p.exists():
        return ""
    content = p.read_text()
    # For large files, give first/last 2000 chars
    if len(content) > 6000:
        return (
            f"[FILE START]\n{content[:2000]}\n...\n"
            f"[FILE END]\n{content[-2000:]}"
        )
    return content


def cmd_snapshot():
    """Take a snapshot of current state."""
    print("═══ Agent-Ersatz — Snapshot ═══")
    take_snapshot()


def cmd_test():
    """Run tests against current state."""
    print("═══ Agent-Ersatz — Test ═══")
    results = run_tests()
    print(f"\n  Results: {results['passed']} passed, {results['failed']} failed")
    if results["failed"] > 0:
        for r in results["results"]:
            if not r["passed"]:
                print(f"    ✗ {r['name']}: {r.get('stderr', 'unknown error')[:200]}")


# ─── Benchmark Command ───────────────────────────────────────────────────

def cmd_benchmark():
    """Run model benchmark via benchmark.py."""
    import subprocess
    bench_script = SHELTER_DIR / "benchmark.py"
    if not bench_script.exists():
        print("  benchmark.py not found")
        return
    # Forward all args after 'benchmark' to benchmark.py
    bench_args = sys.argv[2:]
    r = subprocess.run(
        [sys.executable, str(bench_script)] + bench_args,
    )
    sys.exit(r.returncode)


# ─── CLI ────────────────────────────────────────────────────────────────────

COMMANDS = {
    "setup": ("Configure LLM provider (detect local endpoints)", cmd_setup),
    "benchmark": ("Benchmark models, rank by speed, recommend timeouts", cmd_benchmark),
    "check": ("Report current state vs baseline", cmd_check),
    "heal": ("Detect drift, patch via static + LLM, test, keep/revert", cmd_heal),
    "snapshot": ("Capture current known-good state", cmd_snapshot),
    "test": ("Run verification tests", cmd_test),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Agent-Ersatz — self-healing config manager + benchmark suite for Hermes Agent\n")
        print("Usage: python shelter.py <command> [args]\n")
        print("Commands:")
        for name, (desc, _) in COMMANDS.items():
            print(f"  {name:12s} {desc}")
        print("\n  'benchmark' accepts additional flags: --quick, --full, --skip-reasoning,")
        print("  --model <name>, --save <file>, --recommend-timeouts")
        sys.exit(1)

    command = sys.argv[1]
    _, fn = COMMANDS[command]
    fn()


if __name__ == "__main__":
    main()
