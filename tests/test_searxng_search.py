#!/usr/bin/env python3
"""Test: SearXNG search returns results."""
import sys
import json

sys.path.insert(0, "/home/martin/.hermes/hermes-agent")
from tools.web_tools import _searxng_search

result = _searxng_search("python programming", limit=3)
data = json.loads(result) if isinstance(result, str) else result

if not data.get("success"):
    print(f"FAIL: search returned success=false: {data}")
    sys.exit(1)

web = data.get("data", {}).get("web", [])
if len(web) == 0:
    print("FAIL: search returned 0 results")
    sys.exit(1)

print(f"OK: {len(web)} results returned")
for r in web:
    print(f"  [{r['position']}] {r['title'][:50]}")
