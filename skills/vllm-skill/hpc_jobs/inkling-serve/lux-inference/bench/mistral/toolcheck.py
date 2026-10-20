#!/usr/bin/env python3
"""One chat call with a tool; passes if the server returns a parsed tool call."""
import json, sys, urllib.request

base = sys.argv[1].rstrip("/")
model = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=60))["data"][0]["id"]
body = {
    "model": model, "max_tokens": 256, "temperature": 0,
    "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}],
    "tool_choice": "auto",
}
req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
try:
    msg = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
except Exception as e:
    print(f"TOOL FAIL: {e}")
    sys.exit(1)
calls = msg.get("tool_calls") or []
if calls:
    f = calls[0]["function"]
    print(f"TOOL OK: {f['name']}({f['arguments']})")
else:
    print(f"TOOL FAIL: no tool_calls; content={(msg.get('content') or '')[:200]!r}")
