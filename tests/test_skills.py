import types
import sys
import pathlib

import pytest

# Ensure the root package can be imported as 'fasta2a'
package = types.ModuleType("fasta2a")
package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1])]
sys.modules.setdefault("fasta2a", package)

from fasta2a.server import A2AApp


def test_skill_tool_descriptions():
    app = A2AApp()

    @app.skill("math", "Basic math operations")
    @app.tool()
    async def add(x: int, y: int) -> int:
        """Add two numbers.
        x: first number
        y: second number
        """
        return x + y

    @app.skill("math", "Basic math operations")
    @app.tool()
    async def multiply(x: int, y: int) -> int:
        """Multiply numbers.
        x: first number
        y: second number
        """
        return x * y

    card = app._build_card()
    skill = card["skills"]["math"]
    assert skill["description"] == "Basic math operations"
    tool_entries = {t["name"]: t["description"] for t in skill["tools"]}
    assert tool_entries == {"add": "Add two numbers.", "multiply": "Multiply numbers."}
