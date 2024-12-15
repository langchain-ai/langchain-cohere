from typing import Any, Callable, Dict, List, Sequence, Type, Union

from cohere.types import (
    ToolV2,
    ToolV2Function,
)
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core._api.deprecation import deprecated
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import ToolCall
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from pydantic import BaseModel


@deprecated(
    since="0.1.7",
    removal="0.4.0",
    alternative="""Use the 'tool calling agent' 
    or 'Langgraph agent' with the ChatCohere class instead.
    See https://docs.cohere.com/docs/cohere-and-langchain for more information.""",
)
def create_cohere_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    if "agent_scratchpad" not in (
        prompt.input_variables + list(prompt.partial_variables)
    ):
        raise ValueError(
            "Prompt must have input variable `agent_scratchpad`, but wasn't found. "
            f"Found {prompt.input_variables} instead."
        )
    llm_with_tools = llm.bind(tools=_format_to_cohere_tools(tools))
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | _CohereToolsAgentOutputParser()
    )
    return agent


def _format_to_cohere_tools(
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> List[Dict[str, Any]]:
    return [_convert_to_cohere_tool(tool) for tool in tools]


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
        return ToolV2(
            type="function",
            function=ToolV2Function(
                name=tool.get("title"),
                description=tool.get("description"),
                parameters={"type": "object", "properties": tool.get("properties", {})},
            ),
        ).dict()
    elif (
        (isinstance(tool, type) and issubclass(tool, BaseModel))
        or callable(tool)
        or isinstance(tool, BaseTool)
    ):
        as_json_schema_function = convert_to_openai_function(tool)
        parameters = as_json_schema_function.get("parameters", {})
        return ToolV2(
            type="function",
            function=ToolV2Function(
                name=as_json_schema_function.get("name"),
                description=as_json_schema_function.get(
                    # The Cohere API requires the description field.
                    "description",
                    as_json_schema_function.get("name"),
                ),
                parameters=parameters,
            ),
        ).dict()
    else:
        raise ValueError(
            f"Unsupported tool type {type(tool)}. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
        )


class _CohereToolsAgentOutputParser(
    BaseOutputParser[Union[List[ToolAgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[ToolAgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError(f"Expected ChatGeneration, got {type(result)}")
        message = result[0].message
        if (
            hasattr(result[0].message, "tool_call_chunks")
            and result[0].message.tool_call_chunks
        ):
            tool_calls = result[0].message.tool_call_chunks
        elif hasattr(result[0].message, "tool_calls") and result[0].message.tool_calls:
            tool_calls = result[0].message.tool_calls
        else:
            return AgentFinish(
                return_values={
                    "text": result[0].message.content,
                    "additional_info": result[0].message.additional_kwargs,
                },
                log="",
            )
        actions = []
        tool_calls_list = []
        for tool in tool_calls:
            tool_calls_list.append(
                ToolCall(
                    name=tool.get("name"), args=tool.get("args", {}), id=tool.get("id")
                )
            )
        for tool_call in tool_calls_list:
            log = f"\nInvoking: `{tool_call["name"]}` with `{tool_call["args"]}`\
                \n{message.content}\n"
            actions.append(
                ToolAgentAction(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"],
                    log=log,
                    message_log=[message],
                    tool_call_id=tool_call["id"],
                )
            )
        return actions

    def parse(self, text: str) -> Union[List[ToolAgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")
