import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytest
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseModel, BaseTool, Field

from langchain_cohere.cohere_agent import (
    _format_to_cohere_tools,
    _format_to_cohere_tools_messages,
)


class _TestToolSchema(BaseModel):
    arg_1: str = Field(description="Arg1 description")
    arg_2: int = Field(description="Arg2 description")
    optional_arg_3: Optional[str] = Field(description="Arg3 description", default="3")


class _TestTool(BaseTool):
    name = "test_tool"
    description = "test_tool description"
    args_schema: Type[_TestToolSchema] = _TestToolSchema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


class test_tool_base_model(BaseModel):
    """test_tool description"""

    arg_1: str = Field(description="Arg1 description")
    arg_2: int = Field(description="Arg2 description")
    optional_arg_3: Optional[str] = Field(description="Arg3 description", default="3")


def tool_callable(arg_1: str, arg_2: int, optional_arg_3: Optional[str]) -> None:
    """test_tool description"""


test_tool_as_dict = {
    "title": "test_tool",
    "description": "test_tool description",
    "properties": {
        "arg_1": {"description": "Arg1 description", "type": "string"},
        "arg_2": {"description": "Arg2 description", "type": "integer"},
        "optional_arg_3": {
            "description": "Arg3 description",
            "type": "string",
            "default": "3",
        },
    },
}


@pytest.mark.parametrize(
    "tool,expected_name,has_parameter_descriptions",
    [
        pytest.param(_TestTool(), _TestTool().name, True, id="tool from BaseTool"),
        pytest.param(
            test_tool_base_model, test_tool_base_model.__name__, True, id="BaseModel"
        ),
        pytest.param(
            test_tool_as_dict, test_tool_as_dict["title"], True, id="JSON schema dict"
        ),
        pytest.param(
            tool_callable,
            "tool_callable",
            False,
            id="Callable",
            # langchain_core.utils.function_calling.convert_to_openai_function has a bug
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_format_to_cohere_tools(
    tool: Union[Dict[str, Any], BaseTool, Type[BaseModel]],
    expected_name: str,
    has_parameter_descriptions: bool,
) -> None:
    expected_test_tool_definition = {
        "description": "test_tool description",
        "name": expected_name,
        "parameter_definitions": {
            "arg_1": {
                "description": "Arg1 description"
                if has_parameter_descriptions
                else None,
                "required": True,
                "type": "str",
            },
            "arg_2": {
                "description": "Arg2 description"
                if has_parameter_descriptions
                else None,
                "required": True,
                "type": "int",
            },
            "optional_arg_3": {
                "description": "Arg3 description"
                if has_parameter_descriptions
                else None,
                "required": False,
                "type": "str",
            },
        },
    }

    actual = _format_to_cohere_tools([tool])

    assert [expected_test_tool_definition] == actual


@pytest.mark.parametrize(
    "intermediate_step,expected",
    [
        pytest.param(
            (
                AgentAction(tool="tool_name", tool_input={"arg1": "value1"}, log=""),
                "result",
            ),
            {
                "call": {"name": "tool_name", "parameters": {"arg1": "value1"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input as dict",
        ),
        pytest.param(
            (
                AgentAction(
                    tool="tool_name", tool_input=json.dumps({"arg1": "value1"}), log=""
                ),
                "result",
            ),
            {
                "call": {"name": "tool_name", "parameters": {"arg1": "value1"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input as serialized dict",
        ),
        pytest.param(
            (AgentAction(tool="tool_name", tool_input="foo", log=""), "result"),
            {
                "call": {"name": "tool_name", "parameters": {"input": "foo"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input as string",
        ),
        pytest.param(
            (AgentAction(tool="tool_name", tool_input="['foo']", log=""), "result"),
            {
                "call": {"name": "tool_name", "parameters": {"input": "['foo']"}},
                "outputs": [{"answer": "result"}],
            },
            id="tool_input unrelated JSON",
        ),
    ],
)
def test_format_to_cohere_tools_messages(
    intermediate_step: Tuple[AgentAction, str], expected: List[Dict[str, Any]]
) -> None:
    actual = _format_to_cohere_tools_messages(intermediate_steps=[intermediate_step])

    assert [expected] == actual
