# FastA2A

FastA2A is a Python framework for building and consuming Agent‑to‑Agent (A2A) servers.  It draws inspiration from the popular FastMCP project but targets the [A2A protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) directly, making it easy to expose your tools and services to other agents.  The framework provides a simple decorator‑based API, robust authentication, flexible LLM integrations, and helpers for deployment and testing.

## Key Concepts

A2A facilitates communication between a **client agent** and a **remote agent**.  The client formulates tasks and messages, while the remote agent acts on those requests and returns artifacts.  The protocol emphasizes:

- **Capability discovery** via an *agent card* that lists available tools.
- **Task management** with support for both short and long‑running jobs.
- **Collaboration through messages** that convey context, artifacts and updates.

FastA2A implements these concepts using idiomatic Python and FastAPI.

## Features

- ☑️ **Declarative tool registration** – Decorators auto‑generate JSON schemas from type hints and docstrings.
- 🔐 **Pluggable authentication** – API keys, OAuth2 and JWT backends are built in.
- 🤖 **LLM provider flexibility** – Native integrations for OpenAI, Anthropic/Claude, AWS Bedrock, and Ollama with retry, caching and timeout support.
- 🔄 **Event and message streaming** – Easily push state updates or implement chat‑like workflows using server‑sent events and message handlers.
- 🔗 **MCP proxy** – Bridge calls to existing Model Context Protocol servers for gradual migration.
- 🗂️ **Agent card & skill decorators** – Add custom metadata to your agent card and group tools into skills for richer capability descriptions.
- 🚀 **Deployment helpers** – Sensible defaults for running with Uvicorn and guidance for containerisation.
- ✅ **Testing utilities** – Example `pytest` tests demonstrate how to validate your tools and authentication.

## Installation

FastA2A is currently distributed as source.  To install it into your project:

```bash
pip install fastapi uvicorn httpx  # prerequisites

# copy the `fasta2a` package into your project or install from a wheel
```

If you build and publish the package to PyPI, users can install via `pip install fasta2a`.

## Getting Started

The following walkthrough shows how to define an agent with a single tool, run the server, and invoke it from a client.

### 1. Define your server

```python
from fasta2a.server import A2AApp

app = A2AApp(title="CalculatorAgent", description="Performs basic arithmetic")

@app.tool()
async def add(x: int, y: int) -> int:
    """Add two integers.
    x: first integer
    y: second integer
    """
    return x + y

if __name__ == "__main__":
    from fasta2a.deployment import run
    run(app, host="0.0.0.0", port=8000)

You can enrich your agent card with additional metadata using the `@app.card()` decorator or the `set_card_info()` method:

```python
@app.card()
def card_meta():
    # dynamic values can be computed here
    return {
        "contact": {"email": "admin@example.com"},
        "tags": ["calculator", "demo"]
    }

# Alternatively update fields imperatively
app.set_card_info(icon="🧮")
```

To organise tools into skill categories, use the `@app.skill(category, description)` decorator in combination with `@app.tool()`:

```python
@app.skill("math", "Basic arithmetic operations")
@app.tool()
async def subtract(x: int, y: int) -> int:
    return x - y
```

The agent card will now include a `skills` section grouping related tools.

Running this script will start a FastAPI server exposing:

- `GET /agent-card` – returns the agent card with the `add` tool schema.
- `POST /tools/add` – accepts JSON `{ "x": 1, "y": 2 }` and returns `{ "result": 3 }`.

### 2. Invoke the tool from a client

```python
import asyncio
from fasta2a.client import A2AClient

async def main():
    client = A2AClient("http://localhost:8000")
    card = await client.get_card()
    print("Available tools:", [t["name"] for t in card["tools"]])
    result = await client.call_tool("add", {"x": 7, "y": 5})
    print(result)  # {"result": 12}

asyncio.run(main())
```

## LLM Providers

FastA2A includes native integrations with several large language model providers.  Each provider implements a unified `complete(prompt, **kwargs)` method with retry and caching logic.

```python
from fasta2a.llm import OpenAIProvider, AnthropicProvider, BedrockProvider, OllamaProvider

# OpenAI
openai_llm = OpenAIProvider(api_key="sk-...")
response = await openai_llm.complete("Tell me a joke")

# Anthropic/Claude
anthropic_llm = AnthropicProvider(api_key="anthropic-...")
response = await anthropic_llm.complete("Explain photosynthesis")

# AWS Bedrock (requires boto3 and AWS credentials)
bedrock_llm = BedrockProvider(region_name="us-east-1")
response = await bedrock_llm.complete("Summarise the A2A protocol")

# Ollama (runs locally)
ollama_llm = OllamaProvider()
response = await ollama_llm.complete("Translate 'hello' to Spanish", model="mistral")
```

These providers can be composed with your tools to generate dynamic responses or offload expensive reasoning.

## Authentication

To secure your server, attach an authentication backend:

```python
from fasta2a.auth import APIKeyAuthBackend

app = A2AApp()
app.use_authentication(APIKeyAuthBackend({"my-secret-key"}))

@app.tool()
async def confidential() -> str:
    return "top secret"
```

Clients must include `X-API-Key: my-secret-key` in their request headers.  See `fasta2a.auth` for OAuth2 and JWT backends.

## Events and Messages

Long‑running tasks or streaming data can be implemented using events.  An event handler returns an async generator; clients subscribe via Server‑Sent Events:

```python
from fasta2a.server import A2AApp
import asyncio

app = A2AApp()

@app.event()
async def progress():
    for i in range(5):
        yield {"step": i}
        await asyncio.sleep(1)
```

Message handlers accept arbitrary JSON and can implement custom workflows:

```python
@app.message()
async def echo(payload: dict) -> dict:
    return {"you_sent": payload}
```

The client can subscribe to events with `subscribe_event()` and send messages via `send_message()`.

## MCP Integration

If you need to forward calls to a Model Context Protocol server, use the `MCPProxy`:

```python
from fasta2a.mcp import MCPProxy

mcp = MCPProxy("https://mcp.example.com", api_key="mcp-secret")
result = await mcp.call_tool("search_documents", {"query": "A2A"})
```

This allows you to bridge existing MCP services into an A2A ecosystem during migration.

## Deployment

The `fasta2a.deployment` module exposes a `run()` helper that wraps Uvicorn.  For production use, disable autoreload and configure multiple workers:

```python
from fasta2a.deployment import run
run(app, host="0.0.0.0", port=8080, workers=4)
```

To containerise your server, add a `Dockerfile` that installs dependencies and executes your script using `uvicorn`.  The underlying FastAPI app is available as `app.app` on the `A2AApp` instance, so you can mount it in other ASGI servers as needed.

## Testing

The repository includes example tests in the `tests/` directory.  Install `pytest` and run:

```bash
pip install pytest
pytest -q
```

These tests demonstrate tool invocation and authentication flows.

## Contributing

FastA2A is a reference implementation meant to jump‑start development.  Feel free to extend it with additional protocol features, new LLM providers, or richer authentication schemes.  Pull requests and suggestions are welcome!