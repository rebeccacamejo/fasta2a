"""Top level package for FastA2A.

This module exposes the primary classes and functions necessary for
building and interacting with A2A servers.  FastA2A is inspired by the
FastMCP project but targets the [Agent‑to‑Agent (A2A) protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/).
It aims to make it easy for developers to expose their tools via the
protocol while providing a rich client and a suite of integrations with
popular large language model (LLM) providers.  The package also
includes an authentication subsystem, deployment helpers, and
integration hooks for the Model Context Protocol (MCP).

Key concepts from the A2A protocol include:

* **Capability discovery**: A2A agents advertise their capabilities via a
  card, allowing other agents to determine if they can satisfy a
  request【234393459519911†L234-L243】.
* **Task management**: Communication is oriented around tasks which may
  complete immediately or run for an extended period of time【234393459519911†L245-L249】.
* **Collaboration via messages**: Agents exchange messages to provide
  context, artifacts and updates【234393459519911†L252-L253】.

FastA2A encapsulates these concepts in a Pythonic API built on top of
FastAPI.  See the individual submodules for more details.

"""

from .server import A2AApp, tool, event, message, card, skill
from .client import A2AClient
from .auth import APIKeyAuthBackend, OAuth2AuthBackend, JWTAuthBackend
from .llm import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    ClaudeProvider,
    BedrockProvider,
    OllamaProvider,
)
from .mcp import MCPProxy
from .deployment import run

__all__ = [
    "A2AApp",
    "tool",
    "event",
    "message",
    "card",
    "skill",
    "A2AClient",
    "APIKeyAuthBackend",
    "OAuth2AuthBackend",
    "JWTAuthBackend",
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "ClaudeProvider",
    "BedrockProvider",
    "OllamaProvider",
    "MCPProxy",
    "run",
]