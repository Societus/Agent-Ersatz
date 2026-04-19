#!/usr/bin/env python3
"""Test: No cloud API keys are set in environment."""
import sys

cloud_keys = [
    "FIRECRAWL_API_KEY",
    "PARALLEL_API_KEY",
    "TAVILY_API_KEY",
    "EXA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]

leaked = []
for key in cloud_keys:
    val = __import__("os").environ.get(key, "").strip()
    if val:
        leaked.append(f"{key}={val[:8]}...")

if leaked:
    for l in leaked:
        print(f"FAIL: cloud key detected: {l}")
    sys.exit(1)

print("OK: no cloud API keys in environment")
