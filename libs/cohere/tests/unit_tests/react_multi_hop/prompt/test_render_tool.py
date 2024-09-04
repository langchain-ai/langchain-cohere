from pathlib import Path
from typing import Any, Dict, Type

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_cohere.react_multi_hop.prompt import render_tool

DATA_DIR = Path(__file__).parents[1] / "data" / "tools"


class ExampleToolWithArgs(BaseTool):
    class _args_schema(BaseModel):
        foo: str = Field(description="A description of foo")
        bar: int = Field(description="A description of bar", default=None)

    name = "example_tool"
    description = "A description of example_tool"
    args_schema: Type[BaseModel] = _args_schema  # type: ignore

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


example_tool = ExampleToolWithArgs()

json_schema = {
    "name": "example_tool",
    "description": "A description of example_tool",
    "parameters": {
        "type": "object",
        "properties": {
            "foo": {"type": "string", "description": "A description of foo"},
            "bar": {"type": "integer", "description": "A description of bar"},
        },
        "required": ["foo"],
    },
}

without_description = ExampleToolWithArgs()
without_description.description = ""


class ExampleToolWithoutArgs(BaseTool):
    class _args_schema(BaseModel):
        pass

    name = "example_tool"
    description = "A description of example_tool"
    args_schema: Type[BaseModel] = _args_schema  # type: ignore

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


example_tool_without_args = ExampleToolWithoutArgs()

json_schema_without_args = {
    "name": "example_tool",
    "description": "A description of example_tool",
}


@pytest.mark.parametrize(
    "expected_contents,tool,json_schema",
    [
        pytest.param(
            DATA_DIR / "default.txt", example_tool, None, id="basetool - default"
        ),
        pytest.param(
            DATA_DIR / "default.txt",
            None,
            json_schema,
            id="json schema - default",
        ),
        pytest.param(
            DATA_DIR / "without_description.txt",
            without_description,
            None,
            id="basetool - without description",
        ),
        pytest.param(
            DATA_DIR / "without_args.txt",
            None,
            json_schema_without_args,
            id="json schema - without args",
        ),
        pytest.param(
            DATA_DIR / "without_args.txt",
            example_tool_without_args,
            None,
            id="basetool - without args",
        ),
    ],
)
def test_render_tool(
    expected_contents: Path,
    tool: BaseTool,
    json_schema: Dict,
) -> None:
    with open(expected_contents, "r") as f:
        expected = f.read().rstrip("\n")

    actual = render_tool(
        tool=tool,
        json_schema=json_schema,
    )

    assert actual == expected
