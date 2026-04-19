#!/usr/bin/env python3
"""Test: Config timeouts are set high enough for slow local LLM."""
import sys
import yaml

with open("/home/martin/.hermes/config.yaml") as f:
    cfg = yaml.safe_load(f)

errors = []

gt = cfg.get("agent", {}).get("gateway_timeout", 0)
if gt < 3600:
    errors.append(f"gateway_timeout={gt}, need >=3600")

tt = cfg.get("terminal", {}).get("timeout", 0)
if tt < 300:
    errors.append(f"terminal.timeout={tt}, need >=300")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
    sys.exit(1)

print(f"OK: gateway_timeout={gt}, terminal.timeout={tt}")
