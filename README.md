# hermes-shelter

Self-healing configuration manager for Hermes Agent. Detects drift from declared baseline state, applies patches (static then LLM-driven), runs verification tests, and auto-reverts on failure.

## How It Works

```
hermes update (git pull)
        |
        v
  post-merge hook
        |
        v
  shelter.py heal
        |
        ├── 1. Snapshot current state
        ├── 2. Detect drift vs baseline.yaml
        ├── 3. Apply static patches (git apply)
        ├── 4. Re-check drift
        ├── 5. LLM edits for remaining issues
        ├── 6. Run verification tests
        └── 7. All pass → keep | Any fail → revert
```

## Commands

```
python shelter.py check     # Report drift vs baseline (no changes)
python shelter.py heal      # Full detect → patch → test → keep/revert cycle
python shelter.py snapshot  # Capture current known-good state
python shelter.py test      # Run verification tests only
```

## Structure

```
hermes-shelter/
├── shelter.py              # Main orchestrator
├── shelter.conf            # LLM endpoint config
├── state/
│   ├── baseline.yaml       # Expected state declarations
│   └── snapshots/          # Timestamped state snapshots
├── patches/
│   └── searxng-backend.patch
├── tests/
│   ├── test_searxng_backend.py
│   ├── test_searxng_search.py
│   ├── test_searxng_extract.py
│   ├── test_timeouts.py
│   ├── test_no_cloud_keys.py
│   └── test_mcp_config.py
└── README.md
```

## Baseline Declarations

`state/baseline.yaml` declares the expected state using three mechanisms:

- **checks** — each declares a file, content it `must_contain`, content it `must_not_contain`, and YAML key-value pairs that must hold
- **patches** — git-format patches that shelter tries to apply before falling back to LLM
- **test_scripts** — Python scripts that verify the config actually works end-to-end

## LLM-Driven Patching

When static patches fail (upstream refactored the target file), shelter sends the file + instructions to the local LLM, asking it to produce the modified file. The LLM response is written to disk, then tested. If tests fail, all changes are reverted from backups.

Uses the local OmenLM instance at `omenofdoom:1234` — no cloud calls.

## Adding New Checks

1. Add a check entry to `state/baseline.yaml`
2. Write a test script in `tests/`
3. Add the test to the `test_scripts` list in baseline
4. Run `python shelter.py check` and `python shelter.py test` to verify

## Integration with Hermes Updates

The git post-merge hook at `~/.hermes/hermes-agent/.git/hooks/post-merge` delegates to `shelter.py heal`. This fires automatically after every `hermes update`.
