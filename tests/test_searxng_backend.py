#!/usr/bin/env python3
"""Test: SearXNG is the configured web backend."""
import sys
sys.path.insert(0, "/home/martin/.hermes/hermes-agent")
from tools.web_tools import _get_backend, _is_backend_available

backend = _get_backend()
if backend != "searxng":
    print(f"FAIL: backend is '{backend}', expected 'searxng'")
    sys.exit(1)

if not _is_backend_available("searxng"):
    print("FAIL: searxng backend not available")
    sys.exit(1)

print(f"OK: backend={backend}, available=True")
