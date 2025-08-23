"""Internal utilities for FastA2A.

This module exposes helper functions used by the server and client
implementations.  They are not part of the public API and may change
without notice.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Dict, Tuple


def parse_docstring(func) -> Tuple[str, Dict[str, str]]:
    """Parse the docstring of a function.

    Returns a tuple of the description and a mapping of parameter
    names to descriptions.  The parsing is intentionally simple and
    based on common conventions: the first paragraph of the docstring
    becomes the description, and any lines starting with the
    parameter name followed by a colon are captured as parameter
    descriptions.

    Examples
    --------

    >>> def foo(x, y):
    ...     """Add two numbers.
    ...
    ...     x: the first number
    ...     y: the second number
    ...     """
    ...     return x + y
    >>> parse_docstring(foo)
    ('Add two numbers.', {'x': 'the first number', 'y': 'the second number'})

    """
    doc = inspect.getdoc(func) or ""
    lines = doc.strip().splitlines()
    description = lines[0] if lines else ""
    params: Dict[str, str] = {}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, desc = line.split(":", 1)
            params[name.strip()] = desc.strip()
    return description, params