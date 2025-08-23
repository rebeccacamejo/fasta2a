"""Client for interacting with A2A servers.

The :class:`A2AClient` simplifies making requests to A2A remote
agents.  It handles fetching the agent card for capability discovery,
constructing properly shaped requests for tools, and streaming event
and message responses.  Authentication tokens can be supplied via
constructor parameters or environment variables.

This client is intentionally thin; it is not tied to any specific
framework and can be used in synchronous or asynchronous contexts
through its async API.  Example usage:

.. code-block:: python

    from fasta2a.client import A2AClient

    async def main():
        client = A2AClient("https://remote-agent.example.com", bearer_token="...")
        card = await client.get_card()
        result = await client.call_tool("add", {"x": 1, "y": 2})
        print(result)  # {"result": 3}

    import asyncio
    asyncio.run(main())

"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, Optional

import httpx


class A2AClientError(Exception):
    """Base exception for client errors."""
    pass


class A2AClient:
    """Client for a remote A2A agent.

    Parameters
    ----------
    base_url:
        The base URL of the A2A server, e.g. ``https://agent.example.com``.
    bearer_token:
        Optional bearer token for authentication.  If provided,
        included in the ``Authorization`` header for every request.
    timeout:
        Request timeout in seconds.  Default is 30 seconds.
    """

    def __init__(self, base_url: str, bearer_token: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token or os.getenv("A2A_BEARER_TOKEN")
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=self.timeout)
        self._card_cache: Optional[Dict[str, Any]] = None

    async def get_card(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Retrieve the agent card.

        The card is cached for the lifetime of the client unless
        ``force_refresh`` is True.  See A2A specification for details
        about the card structure.
        """
        if self._card_cache is not None and not force_refresh:
            return self._card_cache
        url = f"{self.base_url}/agent-card"
        headers = self._make_headers()
        resp = await self.http_client.get(url, headers=headers)
        if resp.status_code >= 400:
            raise A2AClientError(f"Failed to fetch card: {resp.status_code} {resp.text}")
        self._card_cache = resp.json()
        return self._card_cache

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool on the remote agent.

        Parameters
        ----------
        tool_name:
            The name of the tool as advertised in the agent card.
        args:
            Arguments for the tool.  The keys and types must match
            those declared in the tool's input schema.

        Returns
        -------
        dict
            The parsed JSON response from the server.  Typically
            contains a ``result`` key.
        """
        url = f"{self.base_url}/tools/{tool_name}"
        headers = self._make_headers()
        resp = await self.http_client.post(url, json=args, headers=headers)
        if resp.status_code >= 400:
            raise A2AClientError(f"Tool call failed: {resp.status_code} {resp.text}")
        return resp.json()

    async def send_message(self, handler_name: str, payload: Dict[str, Any]) -> Any:
        """Send a message to a message handler on the remote agent.

        Messages enable custom workflows beyond simple tool calls.
        """
        url = f"{self.base_url}/messages/{handler_name}"
        headers = self._make_headers()
        resp = await self.http_client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise A2AClientError(f"Message send failed: {resp.status_code} {resp.text}")
        return resp.json()

    async def subscribe_event(self, event_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to a server‑sent event stream.

        Returns an async generator that yields JSON objects from the
        stream.  The caller is responsible for cancelling the
        subscription when finished.  Example::

            async for update in client.subscribe_event("progress"):
                print(update)

        """
        url = f"{self.base_url}/events/{event_name}"
        headers = self._make_headers()
        # Use httpx streaming to consume SSE
        async with self.http_client.stream("GET", url, headers=headers) as resp:
            if resp.status_code >= 400:
                raise A2AClientError(f"Event subscription failed: {resp.status_code} {await resp.aread()}" )
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # SSE format: "data: <json>"
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        # yield raw text on parse error
                        yield {"raw": data_str}

    def _make_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers