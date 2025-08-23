"""Model Context Protocol (MCP) integration.

While FastA2A targets the A2A protocol, many organisations have
existing infrastructure built on the Model Context Protocol.  To ease
migration, FastA2A includes a lightweight proxy that can forward
requests from an A2A client to an MCP server.  The interface
exposed by :class:`MCPProxy` mirrors the A2A client so that callers
can swap between A2A and MCP servers transparently.

This integration is intentionally minimal and does not attempt to
provide full coverage of the MCP SDK.  Instead it focuses on
invoking tools on a remote MCP server and translating the result
format.  Developers are encouraged to extend or replace this proxy
when deeper integration is needed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx


class MCPProxy:
    """Proxy for calling tools exposed via the Model Context Protocol.

    Parameters
    ----------
    base_url:
        The URL of the MCP server, e.g. ``https://my-mcp.example.com``.
    api_key:
        Optional API key used for authentication.  If provided, it will
        be included as a bearer token.
    timeout:
        Request timeout in seconds.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=self.timeout)

    async def call_tool(self, tool_name: str, args: Dict[str, Any], *, card_id: Optional[str] = None) -> Dict[str, Any]:
        """Invoke a tool on the MCP server.

        MCP uses a JSON‑RPC style request body.  The tool name is
        specified as the ``method`` and arguments are passed via
        ``params``.  The response is returned as a dict with a
        ``result`` key.  Errors from the server result in an
        exception.

        Parameters
        ----------
        tool_name:
            The name of the tool to call.
        args:
            A dictionary of arguments to pass to the tool.
        card_id:
            Optional ID of the agent card to target.  If provided,
            appended to the URL as a query parameter.
        """
        url = f"{self.base_url}/call"
        if card_id:
            url += f"?card_id={card_id}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_body = {
            "jsonrpc": "2.0",
            "method": tool_name,
            "params": args,
            "id": 1,
        }
        response = await self.http_client.post(url, json=request_body, headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(f"MCP server error {response.status_code}: {response.text}")
        data = response.json()
        if "error" in data and data["error"]:
            raise RuntimeError(f"MCP call error: {data['error']}")
        return data.get("result", {})