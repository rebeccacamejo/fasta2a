"""Server implementation for FastA2A.

This module provides the classes and decorators needed to build a
fully‑featured A2A server.  The design draws inspiration from
FastMCP—tools are defined with simple Python functions and decorated
to expose them as A2A tasks.  Under the hood, FastAPI handles HTTP
requests, Pydantic validates inputs, and the tool registry generates
an agent card for capability discovery.

The core of the module is :class:`A2AApp`, which encapsulates a
FastAPI application and exposes methods to register tools, events and
messages.  Decorators (:func:`tool`, :func:`event`, :func:`message`)
are thin wrappers around these registration methods, enabling a
declarative programming style.  At runtime, A2AApp automatically
constructs JSON schemas from type hints and docstrings, ensuring that
client agents know how to invoke each tool and what to expect in
return.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Type, TypeVar, get_type_hints

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model

from .auth import AuthenticationBackend
from .utils import parse_docstring

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn)


@dataclass
class ToolDefinition:
    """Metadata for a registered tool.

    Attributes
    ----------
    name:
        The canonical name of the tool as exposed over A2A.
    description:
        A short human‑readable description of the tool.
    input_model:
        A Pydantic model describing the inputs.
    output_model:
        A Pydantic model describing the output.
    func:
        The original Python callable implementing the tool.
    """

    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    func: Callable[..., Awaitable[Any]]

    def to_card_entry(self) -> Dict[str, Any]:
        """Convert the tool definition into an entry for the agent card."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.schema(),
            "output_schema": self.output_model.schema(),
        }


class A2AApp:
    """Main application class for building A2A servers.

    An instance of A2AApp wraps a FastAPI application and manages a
    registry of tools, events and message handlers.  Developers
    interact with the instance primarily through decorators (see
    :func:`tool`) to declare endpoints.  Authentication backends can
    be attached globally and will apply to all registered routes.
    """

    def __init__(self, *, title: str = "FastA2A", version: str = "0.1.0", description: str = "") -> None:
        self.app = FastAPI(title=title, version=version, description=description)
        self._tools: Dict[str, ToolDefinition] = {}
        self._auth_backend: Optional[AuthenticationBackend] = None
        # Additional metadata for the agent card (e.g. contact info, icon)
        self._card_extras: Dict[str, Any] = {}
        # Mapping of skill categories to lists of tool names
        self._skills: Dict[str, Dict[str, Any]] = {}
        # Pending skill metadata for functions not yet registered as tools
        self._pending_skill_info: Dict[Callable[..., Any], list[tuple[str, str]]] = {}
        # Expose the card endpoint
        @self.app.get("/agent-card", summary="Retrieve the agent card")
        async def get_card() -> Dict[str, Any]:
            return self._build_card()

    def use_authentication(self, backend: AuthenticationBackend) -> None:
        """Attach an authentication backend to the application.

        The backend will be invoked for every tool, event and message
        handler.  If authentication fails, the request will be
        rejected with an appropriate HTTP status code.
        """
        self._auth_backend = backend

    # ------------------------------------------------------------------
    # Agent card metadata registration
    # ------------------------------------------------------------------
    def set_card_info(self, **extras: Any) -> None:
        """Update extra metadata fields on the agent card.

        Any key/value pairs provided here will be merged into the
        top‑level agent card dictionary returned by the ``/agent-card``
        endpoint.  Use this to advertise icons, contact information,
        or other non‑standard fields.
        """
        self._card_extras.update(extras)

    def card(self) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
        """Decorator to supply additional agent card metadata.

        The decorated function must return a dictionary.  At import
        time its return value is merged into the agent card.  This
        approach allows dynamic computation of metadata while keeping
        declaration syntax simple.

        Example
        -------

        >>> app = A2AApp()
        >>> @app.card()
        ... def meta():
        ...     return {"contact": {"email": "agent@example.com"}, "tags": ["demo"]}
        >>> # The agent card will now include these fields.
        """
        def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
            extras = func()
            if not isinstance(extras, dict):
                raise ValueError("card decorator function must return a dict")
            self._card_extras.update(extras)
            return func
        return decorator

    # ------------------------------------------------------------------
    # Skill registration
    # ------------------------------------------------------------------
    def register_skill(
        self,
        tool_name: str,
        category: str,
        category_description: str,
        tool_description: Optional[str] = None,
    ) -> None:
        """Associate a tool with a skill category.

        Parameters
        ----------
        tool_name:
            Name of the tool being associated.
        category:
            Skill category name.
        category_description:
            Description of the skill category.
        tool_description:
            Optional description for the tool. Defaults to
            ``category_description``.
        """
        cat = self._skills.get(category)
        if cat is None:
            cat = {"description": category_description, "tools": []}
            self._skills[category] = cat
        # Append the tool if not already present
        if tool_name not in [t["name"] for t in cat["tools"]]:
            cat["tools"].append(
                {"name": tool_name, "description": tool_description or category_description}
            )

    def skill(self, category: str, description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to categorise a tool under a skill.

        Use this decorator in combination with :func:`tool` to assign
        your tool to a named skill.  The decorator stores the skill
        metadata until the tool is registered, at which point
        :meth:`register_skill` is invoked.  Order of decorators does
        not matter.

        Example
        -------

        >>> app = A2AApp()
        >>> @app.skill("math", "Basic math operations")
        ... @app.tool()
        ... async def multiply(x: int, y: int) -> int:
        ...     return x * y
        >>> # The agent card will include a skills entry for "math".
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # If the tool is already registered, associate immediately
            for name, td in self._tools.items():
                if td.func is func:
                    self.register_skill(name, category, description, td.description)
                    break
            else:
                # Otherwise store metadata until the tool is registered
                self._pending_skill_info.setdefault(func, []).append((category, description))
            return func
        return decorator

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def register_tool(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> ToolDefinition:
        """Register a function as an A2A tool.

        The function's type hints and docstring are used to construct
        Pydantic models for input and output.  If either model is
        explicitly provided it will be used instead.  The tool is
        exposed as a POST endpoint at ``/tools/{name}`` on the FastAPI
        application.
        """
        tool_name = name or func.__name__
        if tool_name in self._tools:
            raise ValueError(f"Tool {tool_name} is already registered")

        # Extract description and parameter docs
        doc_desc, param_docs = parse_docstring(func)
        tool_desc = description or doc_desc or f"Tool {tool_name}"

        # Determine input types via type hints
        hints = get_type_hints(func)
        # Exclude return annotation
        return_hint = hints.pop("return", None)
        fields: Dict[str, tuple] = {}
        for arg_name, arg_type in hints.items():
            # Parameter description from docstring if available
            field_desc = param_docs.get(arg_name, "")
            default = ...  # Required by default
            # Inspect the function signature to see if there is a default
            sig = inspect.signature(func)
            param = sig.parameters[arg_name]
            if param.default is not inspect.Parameter.empty:
                default = param.default
            fields[arg_name] = (arg_type, default, field_desc)

        if input_model is None:
            # Dynamically create a Pydantic model for inputs
            model_fields = {
                name: (typ, default) for name, (typ, default, _) in fields.items()
            }
            input_model = create_model(f"{tool_name.capitalize()}Input", **model_fields)  # type: ignore
            # Add descriptions to the field schema
            for field_name, (_, _, desc) in fields.items():
                if not desc:
                    continue
                if hasattr(input_model, "model_fields"):
                    input_model.model_fields[field_name].description = desc  # type: ignore[attr-defined]
                else:  # Pydantic v1
                    input_model.__fields__[field_name].field_info.description = desc

        # Determine output model
        if output_model is None:
            if return_hint is None or return_hint is type(None):  # noqa: E721
                # Default to an empty object
                output_model = create_model(f"{tool_name.capitalize()}Output")  # type: ignore
            else:
                # Wrap the return type in a model
                output_model = create_model(
                    f"{tool_name.capitalize()}Output", result=(return_hint, ...)
                )  # type: ignore

        # Ensure the function is async; wrap if necessary
        async_func: Callable[..., Awaitable[Any]]
        if _is_async_callable(func):
            async_func = func  # type: ignore
        else:
            async def wrapper(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)
            async_func = wrapper

        definition = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            input_model=input_model,
            output_model=output_model,
            func=async_func,
        )
        self._tools[tool_name] = definition

        # If the function was annotated with skill metadata, register it now
        pending = self._pending_skill_info.pop(func, [])
        for category, cat_desc in pending:
            self.register_skill(tool_name, category, cat_desc, tool_desc)

        # Register the FastAPI route
        async def endpoint(
            payload: input_model,  # type: ignore[valid-type]
            request: Request,
            auth_ctx: Optional[dict] = Depends(self._auth_backend) if self._auth_backend else None,
        ) -> JSONResponse:
            """Dynamically generated endpoint for the tool."""
            try:
                result = await definition.func(**payload.dict())
                # Validate result via output model
                output = definition.output_model.parse_obj({"result": result} if definition.output_model.__fields__ else {})
                return JSONResponse(content=output.dict())
            except HTTPException:
                # Re‑raise FastAPI HTTP exceptions directly
                raise
            except Exception as exc:
                logger.exception("Error executing tool %s", tool_name)
                raise HTTPException(status_code=500, detail=str(exc))

        # Mount at /tools/{tool_name}
        self.app.post(f"/tools/{tool_name}", name=tool_name, summary=tool_desc)(endpoint)

        return definition

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------
    def tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[[Callable[..., Any]], Any]]:
        """Decorator to register a function as an A2A tool.

        This decorator can be used with or without arguments.  When used
        without arguments it simply registers the function under its
        own name.  When used with arguments, you may override the
        default name, description or models.

        Example
        -------

        >>> app = A2AApp()
        >>> @app.tool()
        ... async def add(x: int, y: int) -> int:
        ...     Add two integers.
        ...     x: first integer
        ...     y: second integer
        ...     
        ...     return x + y
        >>> # The tool is now available at POST /tools/add
        
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register_tool(
                func,
                name=name,
                description=description,
                input_model=input_model,
                output_model=output_model,
            )
            return func
        return decorator

    def event(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering event handlers.

        Events allow agents to stream state updates or notifications to
        clients.  This is implemented using Server‑Sent Events (SSE)
        under the hood.  When a handler yields values, each value will
        be sent to the client as a JSON message.  If the handler is
        asynchronous it will be awaited on each iteration.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            event_name = name or func.__name__
            doc_desc, _ = parse_docstring(func)
            event_desc = description or doc_desc or f"Event {event_name}"

            async def event_endpoint(request: Request, auth_ctx: Optional[dict] = Depends(self._auth_backend) if self._auth_backend else None):  # type: ignore
                async def event_generator():
                    try:
                        # If the function is async iterate accordingly
                        if _is_async_callable(func):
                            async for item in func():
                                yield f"data: {json.dumps(item)}\n\n"
                        else:
                            for item in func():
                                yield f"data: {json.dumps(item)}\n\n"
                    except Exception as exc:
                        logger.exception("Event handler %s error", event_name)
                        # Yield an error message before closing
                        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

                return StreamingResponse(event_generator(), media_type="text/event-stream")

            from fastapi.responses import StreamingResponse  # imported here to avoid circular import
            import json

            self.app.get(f"/events/{event_name}", name=event_name, summary=event_desc)(event_endpoint)
            return func
        return decorator

    def message(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for message handlers.

        Message handlers process arbitrary JSON messages sent from client
        agents.  They are useful for implementing chat‑like workflows
        or custom protocols layered on top of A2A.  Messages are sent
        via POST to ``/messages/{name}`` and must include a JSON
        payload.  The handler function should accept a dictionary and
        return a dictionary or any JSON serialisable structure.
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            message_name = name or func.__name__
            doc_desc, _ = parse_docstring(func)
            message_desc = description or doc_desc or f"Message handler {message_name}"

            async def message_endpoint(payload: Dict[str, Any], auth_ctx: Optional[dict] = Depends(self._auth_backend) if self._auth_backend else None):  # type: ignore
                try:
                    if _is_async_callable(func):
                        result = await func(payload)
                    else:
                        result = func(payload)
                    return result
                except Exception as exc:
                    logger.exception("Message handler %s error", message_name)
                    raise HTTPException(status_code=500, detail=str(exc))

            self.app.post(f"/messages/{message_name}", name=message_name, summary=message_desc)(message_endpoint)
            return func
        return decorator

    # ------------------------------------------------------------------
    # Card generation
    # ------------------------------------------------------------------
    def _build_card(self) -> Dict[str, Any]:
        """Assemble the agent card as a dictionary.

        The card contains metadata about the agent and its registered
        tools.  Clients use the card for capability discovery.
        """
        card = {
            "name": self.app.title,
            "version": self.app.version,
            "description": self.app.description,
            "tools": [td.to_card_entry() for td in self._tools.values()],
        }
        # Merge in extras provided via set_card_info or card decorator
        card.update(self._card_extras)
        # Include skills if any are defined
        if self._skills:
            card["skills"] = {
                category: {
                    "description": meta["description"],
                    "tools": meta["tools"],
                }
                for category, meta in self._skills.items()
            }
        return card


# Expose decorators as module level functions for convenience
def tool(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Shortcut to :meth:`A2AApp.tool` on a default application.

    This function allows users to write ``@tool`` instead of
    ``@app.tool()`` when a global application instance is used.  If
    called before an :class:`A2AApp` is instantiated, a new
    application is created implicitly.
    """
    global _default_app
    if '_default_app' not in globals() or _default_app is None:
        _default_app = A2AApp()
    return _default_app.tool(*args, **kwargs)


def event(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    global _default_app
    if '_default_app' not in globals() or _default_app is None:
        _default_app = A2AApp()
    return _default_app.event(*args, **kwargs)


def message(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    global _default_app
    if '_default_app' not in globals() or _default_app is None:
        _default_app = A2AApp()
    return _default_app.message(*args, **kwargs)


__all__ = [
    "A2AApp",
    "tool",
    "event",
    "message",
]