#!/usr/bin/env python3
"""Test: MCP servers are configured in config.yaml."""
import sys
import yaml

with open("/home/martin/.hermes/config.yaml") as f:
    cfg = yaml.safe_load(f)

errors = []
mcp = cfg.get("mcp_servers", {})

if "fetch" not in mcp:
    errors.append("mcp_servers.fetch not configured")
elif "mcp-server-fetch" not in str(mcp["fetch"]):
    errors.append("mcp_servers.fetch does not reference mcp-server-fetch")

if "filesystem" not in mcp:
    errors.append("mcp_servers.filesystem not configured")
elif "/home/martin/.hermes/webcache" not in str(mcp["filesystem"]):
    errors.append("mcp_servers.filesystem not pointing to webcache dir")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
    sys.exit(1)

print(f"OK: MCP servers configured (fetch, filesystem)")
