#!/usr/bin/env python3
"""Test: SearXNG extract returns page content."""
import sys

sys.path.insert(0, "/home/martin/.hermes/hermes-agent")
from tools.web_tools import _searxng_extract

results = _searxng_extract(["https://www.python.org"])

if not results:
    print("FAIL: extract returned empty list")
    sys.exit(1)

r = results[0]
if r.get("error"):
    print(f"FAIL: extract error: {r['error']}")
    sys.exit(1)

if not r.get("content") or len(r["content"]) < 100:
    print(f"FAIL: content too short ({len(r.get('content', ''))} chars)")
    sys.exit(1)

print(f"OK: extracted {len(r['content'])} chars from '{r.get('title', 'N/A')}'")
