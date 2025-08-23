"""LLM provider abstractions.

FastA2A provides a common interface for invoking a variety of large
language model (LLM) providers.  Each provider exposes an async
``complete`` method which takes a prompt and returns a generated
response.  A base class implements caching and error handling;
concrete subclasses implement provider‑specific logic.

Agents built on A2A can select the desired provider at runtime or
compose responses from multiple providers.  This flexibility aligns
with the protocol's goal of modality independence and agent
interoperability.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx


class LLMError(Exception):
    """Base class for LLM related errors."""
    pass


@dataclass
class BaseLLMProvider:
    """Base class for all LLM providers.

    Child classes must implement :meth:`_complete`.  The public
    :meth:`complete` method wraps :meth:`_complete` with retry logic,
    optional caching, and timeout support.

    Parameters
    ----------
    timeout:
        Maximum number of seconds to wait for a response.  Defaults to
        30 seconds.
    max_retries:
        Number of times to retry a failed request before raising an
        exception.  A simple exponential backoff is used between
        attempts.
    cache_size:
        Maximum number of prompt/response pairs to cache in memory.
    """

    timeout: float = 30.0
    max_retries: int = 2
    cache_size: int = 128
    _response_cache: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize an LRU cache at instance level.  Using @lru_cache on
        # an instance method is problematic because each instance would
        # share the same cache.  Instead we create a closure that
        # captures a dedicated cache.
        self._response_cache = lru_cache(maxsize=self.cache_size)(self._uncached_complete)

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt.

        This wrapper handles caching, retries and timeouts.  Provider
        implementations should override :meth:`_complete` to send the
        actual request.
        """
        # Use the caching layer only if no non‑hashable kwargs are
        # provided.  If kwargs are present and all values are hashable
        # then they will be included in the cache key.
        try:
            key_kwargs = tuple(sorted(kwargs.items()))
            result = await asyncio.wait_for(
                self._response_cache(prompt, key_kwargs), self.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise LLMError(f"Request timed out after {self.timeout} seconds")

    async def _uncached_complete(self, prompt: str, key_kwargs: Any) -> str:
        # key_kwargs is a tuple of sorted (k, v) pairs; convert back to dict
        kwargs = dict(key_kwargs)
        attempt = 0
        delay = 1.0
        while True:
            try:
                return await self._complete(prompt, **kwargs)
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise LLMError(f"LLM provider error after {attempt} attempts: {exc}") from exc
                await asyncio.sleep(delay)
                delay *= 2

    async def _complete(self, prompt: str, **kwargs: Any) -> str:
        """Concrete providers must implement this method."""
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    """LLM provider for OpenAI's chat and completion endpoints.

    This class supports both chat and completion endpoints.  By
    default it will call the chat/completions endpoint using the
    ``gpt-3.5-turbo`` model.  If you supply a ``model`` kwarg then
    that model will be used.  You may also pass additional parameters
    such as ``temperature`` or ``max_tokens``.
    """

    api_base: str = "https://api.openai.com/v1"

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via constructor or OPENAI_API_KEY env var")
        self.http_client = httpx.AsyncClient(timeout=self.timeout)

    async def _complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.pop("model", "gpt-3.5-turbo")
        endpoint = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        payload.update(kwargs)
        response = await self.http_client.post(endpoint, json=payload, headers=headers)
        if response.status_code >= 400:
            raise LLMError(f"OpenAI API error {response.status_code}: {response.text}")
        data = response.json()
        # Response format: {choices: [{message: {content: ...}}], ...}
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            raise LLMError(f"Unexpected OpenAI response: {data}")


class AnthropicProvider(BaseLLMProvider):
    """LLM provider for Anthropic's models.

    Uses the messages API for models like Claude.  See
    https://docs.anthropic.com/claude/reference/messages_post for
    details.  An API key must be provided via the constructor or the
    ``ANTHROPIC_API_KEY`` environment variable.
    """

    api_base: str = "https://api.anthropic.com/v1"

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided via constructor or ANTHROPIC_API_KEY env var")
        self.http_client = httpx.AsyncClient(timeout=self.timeout)

    async def _complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.pop("model", "claude-3-opus-20240229")
        endpoint = f"{self.api_base}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": kwargs.pop("max_tokens", 1024),
        }
        payload.update(kwargs)
        response = await self.http_client.post(endpoint, json=payload, headers=headers)
        if response.status_code >= 400:
            raise LLMError(f"Anthropic API error {response.status_code}: {response.text}")
        data = response.json()
        # Response format: {content: [...], ...}
        try:
            # content is a list of blocks; join into a single string
            parts = [block.get("text", "") for block in data.get("content", [])]
            return "".join(parts).strip()
        except Exception:
            raise LLMError(f"Unexpected Anthropic response: {data}")


class ClaudeProvider(AnthropicProvider):
    """Alias for AnthropicProvider to support backward compatibility."""
    pass


class BedrockProvider(BaseLLMProvider):
    """LLM provider for AWS Bedrock.

    This provider requires boto3 to be installed and configured with
    credentials.  If boto3 is not available or no region is provided
    then it raises an error.  In practice, you should install boto3
    separately and configure AWS credentials via environment variables
    or IAM roles.  The provider uses the `bedrock-runtime` client to
    invoke models.
    """

    def __init__(self, region_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.region_name = region_name or os.getenv("AWS_REGION")
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError("boto3 is required for BedrockProvider but is not installed")
        if not self.region_name:
            raise ValueError("AWS region must be provided via constructor or AWS_REGION env var")
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.region_name)

    async def _complete(self, prompt: str, **kwargs: Any) -> str:
        # boto3 is synchronous; wrap in thread executor
        import concurrent.futures
        model_id = kwargs.pop("model_id", "anthropic.claude-v2")
        body = {
            "prompt": prompt,
            # Pass through any supported parameters
            **kwargs,
        }

        def _invoke() -> str:
            response = self.bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode("utf-8"),
            )
            # The response body is bytes; decode JSON
            result = json.loads(response["body"].read())
            # The structure depends on the model; attempt common pattern
            return result.get("completion", result.get("message", "")).strip()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _invoke)


class OllamaProvider(BaseLLMProvider):
    """LLM provider for local models served by Ollama.

    Ollama exposes a simple HTTP API on ``localhost:11434`` by default.
    See https://github.com/ollama/ollama for details.  The ``model``
    argument selects which model to use.  Additional parameters are
    passed through to the API.  No authentication is performed.
    """

    base_url: str = "http://localhost:11434"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.http_client = httpx.AsyncClient(timeout=self.timeout)

    async def _complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.pop("model", "llama2")
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
        }
        payload.update(kwargs)
        response = await self.http_client.post(endpoint, json=payload)
        if response.status_code >= 400:
            raise LLMError(f"Ollama API error {response.status_code}: {response.text}")
        # Streaming API returns newline separated JSON; but the /generate
        # endpoint returns full JSON by default
        data = response.json()
        return data.get("response", "").strip()