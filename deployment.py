"""Deployment utilities for FastA2A.

Running an A2A server in production involves more than just writing
handler functions.  This module provides helpers to run your
FastA2A application with sensible defaults and includes comments
describing best practices for containerisation and cloud deployment.

Example
-------

To run an A2A server defined in ``server.py`` with an
``A2AApp`` instance called ``app``::

    from fasta2a.deployment import run
    from myproject.server import app
    run(app, host="0.0.0.0", port=8000, reload=True)

This will start a Uvicorn server with a production‑grade ASGI
implementation.  The ``reload`` flag enables automatic reloading of
code changes in development.
"""

from __future__ import annotations

import multiprocessing
from typing import Optional

import uvicorn

from .server import A2AApp


def run(app: A2AApp, host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False, workers: Optional[int] = None, ssl_keyfile: Optional[str] = None, ssl_certfile: Optional[str] = None) -> None:
    """Run a FastA2A server using Uvicorn.

    Parameters
    ----------
    app:
        The :class:`~fasta2a.server.A2AApp` instance to serve.
    host:
        The host interface to bind to.  Defaults to ``127.0.0.1``.  Use
        ``0.0.0.0`` to listen on all interfaces.
    port:
        The TCP port to listen on.  Defaults to ``8000``.
    reload:
        Whether to enable autoreload.  Useful in development but should
        be disabled in production.
    workers:
        Number of worker processes to start.  If ``None``, a sensible
        default of ``min(4, multiprocessing.cpu_count() + 1)`` is used.
    ssl_keyfile / ssl_certfile:
        Optional paths to SSL key and certificate files.  If both are
        provided the server will run under HTTPS.
    """
    if workers is None:
        # Uvicorn suggests using one worker per CPU core plus one
        workers = min(4, multiprocessing.cpu_count() + 1)
    uvicorn.run(
        app.app,  # underlying FastAPI instance
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


__all__ = ["run"]