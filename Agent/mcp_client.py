"""Minimal MCP client for local MCP server.

Provides mcp_complete(prompt, timeout, context) -> str
Tries to POST to MCP server and return the completion text. Raises on errors so caller can fallback.
"""
import os
import json
import requests
from typing import Any, Dict, Optional

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8701/complete")


def mcp_complete(prompt: str, timeout: float = 30.0, context: Optional[Dict[str, Any]] = None) -> str:
    """Send a completion request to the local MCP server and return text.

    Raises requests.RequestException on transport errors or ValueError if MCP returns error.
    """
    payload = {
        "prompt": prompt,
        "context": context or {},
        "timeout": timeout
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(MCP_URL, data=json.dumps(payload), headers=headers, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    # Expecting {'text': '...'} from MCP server
    if not isinstance(body, dict) or 'text' not in body:
        raise ValueError(f"Unexpected MCP response: {body}")
    return body['text']


def mcp_call_tool(prompt: str, timeout: float = 60.0, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the MCP /complete endpoint and return the full JSON response.

    This is useful when asking MCP to run a registered tool; the tool result will
    appear under the `tool_result` key in the returned dict.
    """
    payload = {
        "prompt": prompt,
        "context": context or {},
        "timeout": timeout,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(MCP_URL, data=json.dumps(payload), headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
