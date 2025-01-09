import json
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, Union

from cohere.types import (
    Tool,
    ToolCall,
    ToolParameterDefinitionsValue,
    ToolResult,
    ToolV2,
    ToolV2Function,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from pydantic import BaseModel

from langchain_cohere.utils import JSON_TO_PYTHON_TYPES


def _format_to_cohere_tools(
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> List[Dict[str, Any]]:
    return [_convert_to_cohere_tool(tool) for tool in tools]


def _format_to_cohere_tools_v2(
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> List[ToolV2]:
    return [_convert_to_cohere_tool_v2(tool) for tool in tools]


def _format_to_cohere_tools_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[Dict[str, Any]]:
    """Convert (AgentAction, tool output) tuples into tool messages."""
    if len(intermediate_steps) == 0:
        return []
    tool_results = []
    for agent_action, observation in intermediate_steps:
        # agent_action.tool_input can be a dict, serialised dict, or string.
        # Cohere API only accepts a dict.
        tool_call_parameters: Dict[str, Any]
        if isinstance(agent_action.tool_input, dict):
            # tool_input is a dict, use as-is.
            tool_call_parameters = agent_action.tool_input
        else:
            try:
                # tool_input is serialised dict.
                tool_call_parameters = json.loads(agent_action.tool_input)
                if not isinstance(tool_call_parameters, dict):
                    raise ValueError()
            except ValueError:
                # tool_input is a string, last ditch attempt at having something useful.
                tool_call_parameters = {"input": agent_action.tool_input}
        tool_results.append(
            ToolResult(
                call=ToolCall(
                    name=agent_action.tool,
                    parameters=tool_call_parameters,
                ),
                outputs=[{"answer": observation}],
            ).dict()
        )

    return tool_results


def _convert_to_cohere_tool(
    tool: Union[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> Dict[str, Any]:
    """
    Convert a BaseTool instance, JSON schema dict, or BaseModel type to a Cohere tool.
    """
    if isinstance(tool, dict):
        if not all(k in tool for k in ("title", "description", "properties")):
            raise ValueError(
                "Unsupported dict type. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
            )
        return Tool(
            name=tool.get("title"),
            description=tool.get("description"),
            parameter_definitions={
                param_name: ToolParameterDefinitionsValue(
                    description=param_definition.get("description"),
                    type=JSON_TO_PYTHON_TYPES.get(
                        param_definition.get("type"), param_definition.get("type")
                    ),
                    required="default" not in param_definition,
                )
                for param_name, param_definition in tool.get("properties", {}).items()
            },
        ).dict()
    elif (
        (isinstance(tool, type) and issubclass(tool, BaseModel))
        or callable(tool)
        or isinstance(tool, BaseTool)
    ):
        as_json_schema_function = convert_to_openai_function(tool)
        parameters = as_json_schema_function.get("parameters", {})
        properties = parameters.get("properties", {})
        parameter_definitions = {}
        for param_name, param_definition in properties.items():
            if "type" in param_definition:
                _type_str = param_definition.get("type")
                _type = JSON_TO_PYTHON_TYPES.get(_type_str)
            elif "anyOf" in param_definition:
                _type_str = next(
                    (
                        t.get("type")
                        for t in param_definition.get("anyOf", [])
                        if t.get("type") != "null"
                    ),
                    param_definition.get("type"),
                )
                _type = JSON_TO_PYTHON_TYPES.get(_type_str)
            else:
                _type = None
            tool_definition = ToolParameterDefinitionsValue(
                description=param_definition.get("description"),
                type=_type,
                required=param_name in parameters.get("required", []),
            )
            parameter_definitions[param_name] = tool_definition
        return Tool(
            name=as_json_schema_function.get("name"),
            description=as_json_schema_function.get(
                # The Cohere API requires the description field.
                "description",
                as_json_schema_function.get("name"),
            ),
            parameter_definitions=parameter_definitions,
        ).dict()
    else:
        raise ValueError(
            f"Unsupported tool type {type(tool)}. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
        )


def _convert_to_cohere_tool_v2(
    tool: Union[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> ToolV2:
    """
    Convert a BaseTool instance, JSON schema dict,
    or BaseModel type to a V2 Cohere tool.
    """
    if isinstance(tool, dict):
        if not all(k in tool for k in ("title", "description", "properties")):
            raise ValueError(
                "Unsupported dict type. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
            )
        return ToolV2(
            type="function",
            function=ToolV2Function(
                name=tool.get("title"),
                description=tool.get("description"),
                parameters={
                    "type": "object",
                    "properties": {
                        param_name: {
                            "description": param_definition.get("description"),
                            "type": param_definition.get("type"),
                        }
                        for param_name, param_definition in tool.get(
                            "properties", {}
                        ).items()
                    },
                    "required": [
                        param_name
                        for param_name, param_definition in tool.get(
                            "properties", {}
                        ).items()
                        if "default" not in param_definition
                    ],
                },
            ),
        )
    elif (
        (isinstance(tool, type) and issubclass(tool, BaseModel))
        or callable(tool)
        or isinstance(tool, BaseTool)
    ):
        as_json_schema_function = convert_to_openai_function(tool)
        parameters = as_json_schema_function.get("parameters", {})
        properties = parameters.get("properties", {})
        parameter_definitions = {}
        required_params = []
        for param_name, param_definition in properties.items():
            if "type" in param_definition:
                _type = param_definition.get("type")
            elif "anyOf" in param_definition:
                _type = next(
                    (
                        t.get("type")
                        for t in param_definition.get("anyOf", [])
                        if t.get("type") != "null"
                    ),
                    param_definition.get("type"),
                )
            else:
                _type = None
            tool_definition = {
                "type": _type,
                "description": param_definition.get("description"),
            }
            parameter_definitions[param_name] = tool_definition
            if param_name in parameters.get("required", []):
                required_params.append(param_name)
        return ToolV2(
            type="function",
            function=ToolV2Function(
                name=as_json_schema_function.get("name"),
                description=as_json_schema_function.get(
                    # The Cohere API requires the description field.
                    "description",
                    as_json_schema_function.get("name"),
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        **parameter_definitions,
                    },
                    "required": required_params,
                },
            ),
        )
    else:
        raise ValueError(
            f"Unsupported tool type {type(tool)}. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
        )


class _CohereToolsAgentOutputParser(
    BaseOutputParser[Union[List[AgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError(f"Expected ChatGeneration, got {type(result)}")
        if "tool_calls" in result[0].message.additional_kwargs:
            actions = []
            for tool in result[0].message.additional_kwargs["tool_calls"]:
                function = tool.get("function", {})
                actions.append(
                    AgentAction(
                        tool=function.get("name"),
                        tool_input=function.get("arguments"),
                        log=function.get("name"),
                    )
                )
            return actions
        else:
            return AgentFinish(
                return_values={
                    "text": result[0].message.content,
                    "additional_info": result[0].message.additional_kwargs,
                },
                log="",
            )

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")
