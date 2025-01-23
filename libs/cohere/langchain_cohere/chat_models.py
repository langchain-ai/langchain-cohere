import copy
import json
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from cohere.types import (
    AssistantChatMessageV2,
    ChatMessageV2,
    ChatResponse,
    DocumentToolContent,
    NonStreamedChatResponse,
    SystemChatMessageV2,
    ToolCall,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    UserChatMessageV2,
)
from cohere.types import Document as DocumentV2
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.messages import (
    ToolCall as LC_ToolCall,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, PrivateAttr

from langchain_cohere.cohere_agent import (
    _convert_to_cohere_tool,
    _format_to_cohere_tools_v2,
)
from langchain_cohere.llms import BaseCohere
from langchain_cohere.react_multi_hop.prompt import convert_to_documents

LC_TOOL_CALL_TEMPLATE = {
    "id": "",
    "type": "function",
    "function": {
        "name": "",
        "arguments": "",
    },
}


def _message_to_cohere_tool_results(
    messages: List[BaseMessage], tool_message_index: int
) -> List[Dict[str, Any]]:
    """Get tool_results from messages."""
    tool_results = []
    tool_message = messages[tool_message_index]
    if not isinstance(tool_message, ToolMessage):
        raise ValueError(
            "The message index does not correspond to an instance of ToolMessage"
        )

    messages_until_tool = messages[:tool_message_index]
    previous_ai_message = [
        message
        for message in messages_until_tool
        if isinstance(message, AIMessage) and message.tool_calls
    ][-1]
    tool_results.extend(
        [
            {
                "call": ToolCall(
                    name=lc_tool_call["name"],
                    parameters=lc_tool_call["args"],
                ),
                "outputs": convert_to_documents(tool_message.content),
            }
            for lc_tool_call in previous_ai_message.tool_calls
            if lc_tool_call["id"] == tool_message.tool_call_id
        ]
    )
    return tool_results


def _get_curr_chat_turn_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Get the messages for the current chat turn."""
    current_chat_turn_messages = []
    for message in messages[::-1]:
        current_chat_turn_messages.append(message)
        if isinstance(message, HumanMessage):
            break
    return current_chat_turn_messages[::-1]


def _messages_to_cohere_tool_results_curr_chat_turn(
    messages: List[BaseMessage],
) -> List[Dict[str, Any]]:
    """Get tool_results from messages."""
    tool_results = []
    curr_chat_turn_messages = _get_curr_chat_turn_messages(messages)
    for message in curr_chat_turn_messages:
        if isinstance(message, ToolMessage):
            tool_message = message
            previous_ai_msgs = [
                message
                for message in curr_chat_turn_messages
                if isinstance(message, AIMessage) and message.tool_calls
            ]
            if previous_ai_msgs:
                previous_ai_msg = previous_ai_msgs[-1]
                tool_results.extend(
                    [
                        {
                            "call": ToolCall(
                                name=lc_tool_call["name"],
                                parameters=lc_tool_call["args"],
                            ),
                            "outputs": convert_to_documents(tool_message.content),
                        }
                        for lc_tool_call in previous_ai_msg.tool_calls
                        if lc_tool_call["id"] == tool_message.tool_call_id
                    ]
                )

    return tool_results


def get_role(message: BaseMessage) -> str:
    """Get the role of the message.

    Args:
        message: The message.

    Returns:
        The role of the message.

    Raises:
        ValueError: If the message is of an unknown type.
    """
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return "User"
    elif isinstance(message, AIMessage):
        return "Chatbot"
    elif isinstance(message, SystemMessage):
        return "System"
    elif isinstance(message, ToolMessage):
        return "Tool"
    else:
        raise ValueError(f"Got unknown type {type(message).__name__}")


def _get_message_cohere_format(
    message: BaseMessage, tool_results: Optional[List[Dict[Any, Any]]]
) -> Dict[
    str,
    Union[
        str,
        List[LC_ToolCall],
        List[ToolCall],
        List[Union[str, Dict[Any, Any]]],
        List[Dict[Any, Any]],
        None,
    ],
]:
    """Get the formatted message as required in cohere's api.

    Args:
        message: The BaseMessage.
        tool_results: The tool results if any

    Returns:
        The formatted message as required in cohere's api.
    """

    if isinstance(message, AIMessage):
        return {
            "role": get_role(message),
            "message": message.content,
            "tool_calls": _get_tool_call_cohere_format(message.tool_calls),
        }
    elif isinstance(message, HumanMessage) or isinstance(message, SystemMessage):
        return {"role": get_role(message), "message": message.content}
    elif isinstance(message, ToolMessage):
        return {"role": get_role(message), "tool_results": tool_results}
    else:
        raise ValueError(f"Got unknown type {message}")


def _get_tool_call_cohere_format(tool_calls: List[LC_ToolCall]) -> List[ToolCall]:
    """Convert LangChain tool calls into Cohere's format"""
    cohere_tool_calls = []
    for lc_tool_call in tool_calls:
        name = lc_tool_call.get("name")
        parameters = lc_tool_call.get("args")
        id = lc_tool_call.get("id")
        cohere_tool_calls.append(ToolCall(name=name, parameters=parameters, id=id))
    return cohere_tool_calls


def get_cohere_chat_request(
    messages: List[BaseMessage],
    *,
    documents: Optional[List[Document]] = None,
    connectors: Optional[List[Dict[str, str]]] = None,
    stop_sequences: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the request for the Cohere chat API.

    Args:
        messages: The messages.
        connectors: The connectors.
        **kwargs: The keyword arguments.

    Returns:
        The request for the Cohere chat API.
    """
    if connectors or "connectors" in kwargs:
        warn_deprecated(
            since="0.3.3",
            message=(
                "The 'connectors' parameter is deprecated as of version 0.3.3.\n"
                "Please use the 'tools' parameter instead."
            ),
            removal="0.4.0",
        )
    additional_kwargs = messages[-1].additional_kwargs

    # cohere SDK will fail loudly if both connectors and documents are provided
    if additional_kwargs.get("documents", []) and documents and len(documents) > 0:
        raise ValueError(
            "Received documents both as a keyword argument and as an prompt additional keyword argument. Please choose only one option."  # noqa: E501
        )

    parsed_docs: Optional[Union[List[Document], List[Dict]]] = None
    if "documents" in additional_kwargs:
        parsed_docs = (
            additional_kwargs["documents"]
            if len(additional_kwargs.get("documents", []) or []) > 0
            else None
        )
    elif (documents is not None) and (len(documents) > 0):
        parsed_docs = documents

    formatted_docs: Optional[List[Dict[str, Any]]] = None
    if parsed_docs:
        formatted_docs = []
        for i, parsed_doc in enumerate(parsed_docs):
            if isinstance(parsed_doc, Document):
                formatted_docs.append(
                    {
                        "text": parsed_doc.page_content,
                        "id": parsed_doc.metadata.get("id") or f"doc-{str(i)}",
                    }
                )
            elif isinstance(parsed_doc, dict):
                formatted_docs.append(parsed_doc)

    # by enabling automatic prompt truncation, the probability of request failure is
    # reduced with minimal impact on response quality
    prompt_truncation = (
        "AUTO" if formatted_docs is not None or connectors is not None else None
    )
    tool_results: Optional[List[Dict[str, Any]]] = (
        _messages_to_cohere_tool_results_curr_chat_turn(messages)
        or kwargs.get("tool_results")
    )
    if not tool_results:
        tool_results = None
    # check if the last message is a tool message or human message
    if not (
        isinstance(messages[-1], ToolMessage) or isinstance(messages[-1], HumanMessage)
    ):
        raise ValueError("The last message is not an ToolMessage or HumanMessage")

    chat_history = []
    temp_tool_results = []
    # if force_single_step is set to False, then only message is empty in request if there is tool call  # noqa: E501
    if not kwargs.get("force_single_step"):
        for i, message in enumerate(messages[:-1]):
            # If there are multiple tool messages, then we need to aggregate them into one single tool message to pass into chat history  # noqa: E501
            if isinstance(message, ToolMessage):
                temp_tool_results += _message_to_cohere_tool_results(messages, i)

                if (i == len(messages) - 1) or not (
                    isinstance(messages[i + 1], ToolMessage)
                ):
                    cohere_message = _get_message_cohere_format(
                        message, temp_tool_results
                    )
                    chat_history.append(cohere_message)
                    temp_tool_results = []
            else:
                chat_history.append(_get_message_cohere_format(message, None))

        message_str = "" if tool_results else messages[-1].content

    else:
        message_str = ""
        # if force_single_step is set to True, then message is the last human message in the conversation  # noqa: E501
        for i, message in enumerate(messages[:-1]):
            if isinstance(message, AIMessage) and message.tool_calls:
                continue

            # If there are multiple tool messages, then we need to aggregate them into one single tool message to pass into chat history  # noqa: E501
            if isinstance(message, ToolMessage):
                temp_tool_results += _message_to_cohere_tool_results(messages, i)

                if (i == len(messages) - 1) or not (
                    isinstance(messages[i + 1], ToolMessage)
                ):
                    cohere_message = _get_message_cohere_format(
                        message, temp_tool_results
                    )
                    chat_history.append(cohere_message)
                    temp_tool_results = []
            else:
                chat_history.append(_get_message_cohere_format(message, None))
        # Add the last human message in the conversation to the message string
        for message in messages[::-1]:
            if (isinstance(message, HumanMessage)) and (message.content):
                message_str = message.content
                break

    req = {
        "message": message_str,
        "chat_history": chat_history,
        "tool_results": tool_results,
        "documents": formatted_docs,
        "connectors": connectors,
        "prompt_truncation": prompt_truncation,
        "stop_sequences": stop_sequences,
        **kwargs,
    }

    return {k: v for k, v in req.items() if v is not None}


def get_role_v2(message: BaseMessage) -> str:
    """Get the role of the message (V2).
    Args:
        message: The message.
    Returns:
        The role of the message.
    Raises:
        ValueError: If the message is of an unknown type.
    """
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, ToolMessage):
        return "tool"
    else:
        raise ValueError(f"Got unknown type {type(message).__name__}")


def _get_message_cohere_format_v2(
    message: BaseMessage, tool_results: Optional[List[MutableMapping]] = None
) -> ChatMessageV2:
    """Get the formatted message as required in cohere's api (V2).
    Args:
        message: The BaseMessage.
        tool_results: The tool results if any
    Returns:
        The formatted message as required in cohere's api.
    """
    if isinstance(message, AIMessage):
        if message.tool_calls:
            return AssistantChatMessageV2(
                role=get_role_v2(message),
                tool_plan=message.content
                if message.content
                else "I will assist you using the tools provided.",
                tool_calls=[
                    ToolCallV2(
                        id=tool_call.get("id"),
                        type="function",
                        function=ToolCallV2Function(
                            name=tool_call.get("name"),
                            arguments=json.dumps(tool_call.get("args")),
                        ),
                    )
                    for tool_call in message.tool_calls
                ],
            )
        return AssistantChatMessageV2(
            role=get_role_v2(message),
            content=message.content,
        )
    elif isinstance(message, HumanMessage):
        return UserChatMessageV2(
            role=get_role_v2(message),
            content=message.content,
        )
    elif isinstance(message, SystemMessage):
        return SystemChatMessageV2(
            role=get_role_v2(message),
            content=message.content,
        )
    elif isinstance(message, ToolMessage):
        if tool_results is None:
            raise ValueError("Tool results are required for ToolMessage")

        content = [
            DocumentToolContent(
                type="document",
                document=DocumentV2(
                    data=dict(tool_result),
                ),
            )
            for tool_result in tool_results
        ]

        if not content:
            content = [
                DocumentToolContent(
                    type="document", document=DocumentV2(data={"output": ""})
                )
            ]

        return ToolChatMessageV2(
            role=get_role_v2(message),
            tool_call_id=message.tool_call_id,
            content=content,
        )
    else:
        raise ValueError(f"Got unknown type {message}")


def get_cohere_chat_request_v2(
    messages: List[BaseMessage],
    *,
    documents: Optional[List[Document]] = None,
    stop_sequences: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the request for the Cohere chat API (V2).
    Args:
        messages: The messages.
        **kwargs: The keyword arguments.
    Returns:
        The request for the Cohere chat API.
    """
    additional_kwargs = messages[-1].additional_kwargs

    # cohere SDK will fail loudly if both connectors and documents are provided
    if additional_kwargs.get("documents", []) and documents and len(documents) > 0:
        raise ValueError(
            "Received documents both as a keyword argument and as an prompt additional keyword argument. Please choose only one option."  # noqa: E501
        )

    parsed_docs: Optional[Union[List[Document], List[Dict]]] = None
    if "documents" in additional_kwargs:
        parsed_docs = (
            additional_kwargs["documents"]
            if len(additional_kwargs.get("documents", []) or []) > 0
            else None
        )
    elif (documents is not None) and (len(documents) > 0):
        parsed_docs = documents

    formatted_docs: Optional[List[Dict[str, Any]]] = None
    if parsed_docs:
        formatted_docs = []
        for i, parsed_doc in enumerate(parsed_docs):
            if isinstance(parsed_doc, Document):
                formatted_docs.append(
                    {
                        "id": parsed_doc.metadata.get("id") or f"doc-{str(i)}",
                        "data": {
                            "text": parsed_doc.page_content,
                        },
                    }
                )
            elif isinstance(parsed_doc, dict):
                if "data" not in parsed_doc:
                    formatted_docs.append(
                        {
                            "id": parsed_doc.get("id") or f"doc-{str(i)}",
                            "data": {
                                **parsed_doc,
                            },
                        }
                    )
                else:
                    formatted_docs.append(parsed_doc)

    # check if the last message is a tool message or human message
    if not (
        isinstance(messages[-1], ToolMessage) or isinstance(messages[-1], HumanMessage)
    ):
        raise ValueError("The last message is not an ToolMessage or HumanMessage")

    if kwargs.get("preamble"):
        messages = [SystemMessage(content=str(kwargs.get("preamble")))] + messages
        del kwargs["preamble"]

    if kwargs.get("connectors"):
        warn_deprecated(
            "0.4.0",
            message=(
                "The 'connectors' parameter is deprecated as of version 0.4.0.\n"
                "Please use the 'tools' parameter instead."
            ),
            removal="0.4.0",
        )
        raise ValueError(
            "The 'connectors' parameter is deprecated as of version 0.4.0."
        )

    chat_history_with_curr_msg = []
    for message in messages:
        if isinstance(message, ToolMessage):
            tool_output = convert_to_documents(message.content)
            chat_history_with_curr_msg.append(
                _get_message_cohere_format_v2(message, tool_output)
            )
        else:
            chat_history_with_curr_msg.append(
                _get_message_cohere_format_v2(message, None)
            )

    req = {
        "messages": chat_history_with_curr_msg,
        "documents": formatted_docs,
        "stop_sequences": stop_sequences,
        **kwargs,
    }

    return {k: v for k, v in req.items() if v is not None}


class ChatCohere(BaseChatModel, BaseCohere):
    """
    Implements the BaseChatModel (and BaseLanguageModel) interface with Cohere's large
    language models.

    Find out more about us at https://cohere.com and https://huggingface.co/CohereForAI

    This implementation uses the Chat API - see https://docs.cohere.com/reference/chat

    To use this you'll need to a Cohere API key - either pass it to cohere_api_key
    parameter or set the COHERE_API_KEY environment variable.

    API keys are available on https://cohere.com - it's free to sign up and trial API
    keys work with this implementation.

    Basic Example:
        .. code-block:: python

            from langchain_cohere import ChatCohere
            from langchain_core.messages import HumanMessage

            llm = ChatCohere(cohere_api_key="{API KEY}")

            message = [HumanMessage(content="Hello, can you introduce yourself?")]

            print(llm.invoke(message).content)
    """

    preamble: Optional[str] = None

    _default_model_name: Optional[str] = PrivateAttr(
        default=None
    )  # Used internally to cache API calls to list models.

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = _format_to_cohere_tools_v2(tools)
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        method: Literal[
            "function_calling", "tool_calling", "json_mode", "json_schema"
        ] = "json_schema",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.
        Given schema can be a Pydantic class or a dict.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict.
            method: The method for steering model generation, one of:
                - "function_calling" or "tool_calling":
                    Uses Cohere's tool-calling (formerly called function calling)
                    API: https://docs.cohere.com/v2/docs/tool-use
                - "json_schema":
                    Uses Cohere's Structured Output API: https://docs.cohere.com/docs/structured-outputs
                    Allows the user to pass a json schema (or pydantic)
                        to the model for structured output.
                    This is the default method.
                    Supported for "command-r", "command-r-plus", and later
                    models.
                - "json_mode":
                    Uses Cohere's Structured Output API: https://docs.cohere.com/docs/structured-outputs
                    Supported for "command-r", "command-r-plus", and later
                    models.

        Returns:
            A Runnable that takes any ChatModel input and returns either a dict or
            Pydantic class as output.
        """
        if (not schema) and (method != "json_mode"):
            raise ValueError(
                "schema must be specified when method is not 'json_mode'. "
                f"Received {schema}."
            )
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        if method == "function_calling" or method == "tool_calling":
            llm = self.bind_tools([schema], **kwargs)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                key_name = _convert_to_cohere_tool(schema)["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_mode":
            # Refers to Cohere's `json_object` mode
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            response_format = (
                dict(
                    schema.model_json_schema().items()  # type: ignore[union-attr]
                )
                if is_pydantic_schema
                else schema
            )
            cohere_response_format: Dict[Any, Any] = {"type": "json_object"}
            cohere_response_format["schema"] = {
                k: v
                for k, v in response_format.items()  # type: ignore[union-attr]
            }
            llm = self.bind(response_format=cohere_response_format)
            if is_pydantic_schema:
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"or 'json_schema' or 'json_mode'. Received: '{method}'"
            )

        return llm | output_parser

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "cohere-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        base_params = {
            "model": self.model,
            "temperature": self.temperature,
            "preamble": self.preamble,
        }
        return {k: v for k, v in base_params.items() if v is not None}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="cohere",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens"):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _stream_v1(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        if hasattr(self.client, "chat_stream"):  # detect and support sdk v5
            stream = self.client.chat_stream(**request)
        else:
            stream = self.client.chat(**request, stream=True)
        for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            if data.event_type == "tool-calls-chunk":
                if data.tool_call_delta:
                    delta = data.tool_call_delta
                    cohere_tool_call_chunk = _format_cohere_tool_calls([delta])[0]
                    message = AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            ToolCallChunk(
                                name=cohere_tool_call_chunk["function"].get("name"),
                                args=cohere_tool_call_chunk["function"].get(
                                    "arguments"
                                ),
                                id=cohere_tool_call_chunk.get("id"),
                                index=delta.index,
                            )
                        ],
                    )
                    chunk = ChatGenerationChunk(message=message)
                else:
                    delta = data.text
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            elif data.event_type == "stream-end":
                generation_info = self._get_generation_info(data.response)
                message = AIMessageChunk(
                    content="",
                    additional_kwargs=generation_info,
                )
                yield ChatGenerationChunk(
                    message=message,
                    generation_info=generation_info,
                )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Workaround to allow create_cohere_react_agent to work with the
        # current implementation. create_cohere_react_agent relies on the
        # 'raw_prompting' parameter to be set, which is only available
        # in the v1 API.
        # TODO: Remove this workaround once create_cohere_react_agent is
        # updated to work with the v2 API.
        if kwargs.get("raw_prompting"):
            for value in self._stream_v1(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield value
            return

        request = get_cohere_chat_request_v2(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        stream = self.client.v2.chat_stream(**request)
        curr_tool_call: Dict[str, Any] = copy.deepcopy(LC_TOOL_CALL_TEMPLATE)
        tool_calls = []
        for data in stream:
            if data.type == "content-delta":
                delta = data.delta.message.content.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            elif data.type in {
                "tool-call-start",
                "tool-call-delta",
                "tool-plan-delta",
                "tool-call-end",
            }:
                # tool-call-start: Contains the name of the tool function.
                #                  No arguments are included
                # tool-call-delta: Contains the arguments of the tool function.
                #                  The function name is not included
                # tool-plan-delta: Contains a chunk of the tool-plan message
                # tool-call-end:   End of tool call streaming
                if data.type in {"tool-call-start", "tool-call-delta"}:
                    index = data.index
                    delta = data.delta.message

                    # To construct the current tool call you need
                    # to buffer all the deltas
                    if data.type == "tool-call-start":
                        curr_tool_call["id"] = delta.tool_calls.id
                        curr_tool_call["function"][
                            "name"
                        ] = delta.tool_calls.function.name
                    elif data.type == "tool-call-delta":
                        curr_tool_call["function"][
                            "arguments"
                        ] += delta.tool_calls.function.arguments

                    # If the current stream event is a tool-call-start,
                    # then the ToolCallV2 object will only contain the function
                    # name. If the current stream event is a tool-call-delta,
                    # then the ToolCallV2 object will only contain the arguments.
                    tool_call_v2 = ToolCallV2(
                        function=ToolCallV2Function(
                            name=delta.tool_calls.function.name
                            if hasattr(delta.tool_calls.function, "name")
                            else None,
                            arguments=delta.tool_calls.function.arguments
                            if hasattr(delta.tool_calls.function, "arguments")
                            else None,
                        )
                    )

                    cohere_tool_call_chunk = _format_cohere_tool_calls_v2(
                        [tool_call_v2]
                    )[0]
                    message = AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            ToolCallChunk(
                                name=cohere_tool_call_chunk["function"].get("name"),
                                args=cohere_tool_call_chunk["function"].get(
                                    "arguments"
                                ),
                                id=cohere_tool_call_chunk.get("id"),
                                index=index,
                            )
                        ],
                    )
                    chunk = ChatGenerationChunk(message=message)
                elif data.type == "tool-plan-delta":
                    delta = data.delta.message.tool_plan
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                elif data.type == "tool-call-end":
                    # Maintain a list of all of the tool calls seen during streaming
                    tool_calls.append(curr_tool_call)
                    curr_tool_call = copy.deepcopy(LC_TOOL_CALL_TEMPLATE)
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            elif data.type == "message-end":
                delta = data.delta
                generation_info = self._get_stream_info_v2(
                    delta, documents=request.get("documents"), tool_calls=tool_calls
                )
                message = AIMessageChunk(
                    content="",
                    additional_kwargs=generation_info,
                )
                yield ChatGenerationChunk(
                    message=message,
                    generation_info=generation_info,
                )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        request = get_cohere_chat_request_v2(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        stream = self.async_client.v2.chat_stream(**request)
        curr_tool_call: Dict[str, Any] = copy.deepcopy(LC_TOOL_CALL_TEMPLATE)
        tool_plan_deltas = []
        tool_calls = []
        async for data in stream:
            if data.type == "content-delta":
                delta = data.delta.message.content.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            elif data.type in {
                "tool-call-start",
                "tool-call-delta",
                "tool-plan-delta",
                "tool-call-end",
            }:
                # tool-call-start: Contains the name of the tool function.
                #                  No arguments are included
                # tool-call-delta: Contains the arguments of the tool function.
                #                  The function name is not included
                # tool-plan-delta: Contains a chunk of the tool-plan message
                # tool-call-end:   End of tool call streaming
                if data.type in {"tool-call-start", "tool-call-delta"}:
                    index = data.index
                    delta = data.delta.message

                    # To construct the current tool call you
                    # need to buffer all the deltas
                    if data.type == "tool-call-start":
                        curr_tool_call["id"] = delta.tool_calls.id
                        curr_tool_call["function"][
                            "name"
                        ] = delta.tool_calls.function.name
                    elif data.type == "tool-call-delta":
                        curr_tool_call["function"][
                            "arguments"
                        ] += delta.tool_calls.function.arguments

                    # If the current stream event is a tool-call-start,
                    # then the ToolCallV2 object will only contain the
                    # function name. If the current stream event is a
                    # tool-call-delta, then the ToolCallV2 object will
                    # only contain the arguments.
                    tool_call_v2 = ToolCallV2(
                        function=ToolCallV2Function(
                            name=delta.tool_calls.function.name
                            if hasattr(delta.tool_calls.function, "name")
                            else None,
                            arguments=delta.tool_calls.function.arguments
                            if hasattr(delta.tool_calls.function, "arguments")
                            else None,
                        )
                    )

                    cohere_tool_call_chunk = _format_cohere_tool_calls_v2(
                        [tool_call_v2]
                    )[0]
                    message = AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            ToolCallChunk(
                                name=cohere_tool_call_chunk["function"].get("name"),
                                args=cohere_tool_call_chunk["function"].get(
                                    "arguments"
                                ),
                                id=cohere_tool_call_chunk.get("id"),
                                index=index,
                            )
                        ],
                    )
                    chunk = ChatGenerationChunk(message=message)
                elif data.type == "tool-plan-delta":
                    delta = data.delta.message.tool_plan
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                    tool_plan_deltas.append(delta)
                elif data.type == "tool-call-end":
                    # Maintain a list of all of the tool calls seen during streaming
                    tool_calls.append(curr_tool_call)
                    curr_tool_call = copy.deepcopy(LC_TOOL_CALL_TEMPLATE)
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
            elif data.type == "message-end":
                delta = data.delta
                generation_info = self._get_stream_info_v2(
                    delta, documents=request.get("documents"), tool_calls=tool_calls
                )

                tool_call_chunks = []
                if tool_calls:
                    content = "".join(tool_plan_deltas)
                    try:
                        tool_call_chunks = [
                            {
                                "name": tool_call["function"].get("name"),
                                "args": tool_call["function"].get("arguments"),
                                "id": tool_call.get("id"),
                                "index": tool_call.get("index"),
                            }
                            for tool_call in tool_calls
                        ]
                    except KeyError:
                        pass
                else:
                    content = ""

                message = AIMessageChunk(
                    content=content,
                    additional_kwargs=generation_info,
                    tool_call_chunks=tool_call_chunks,
                    usage_metadata=generation_info.get("token_count"),
                )
                yield ChatGenerationChunk(
                    message=message,
                    generation_info=generation_info,
                )

    def _get_generation_info(self, response: NonStreamedChatResponse) -> Dict[str, Any]:
        """Get the generation info from cohere API response."""
        generation_info: Dict[str, Any] = {
            "documents": response.documents,
            "citations": response.citations,
            "search_results": response.search_results,
            "search_queries": response.search_queries,
            "is_search_required": response.is_search_required,
            "generation_id": response.generation_id,
        }
        if response.tool_calls:
            # Only populate tool_calls when 1) present on the response and
            #  2) has one or more calls.
            generation_info["tool_calls"] = _format_cohere_tool_calls(
                response.tool_calls
            )
        if hasattr(response, "token_count"):
            generation_info["token_count"] = response.token_count
        elif hasattr(response, "meta") and response.meta is not None:
            if hasattr(response.meta, "tokens") and response.meta.tokens is not None:
                generation_info["token_count"] = response.meta.tokens.dict()
        return generation_info

    def _get_generation_info_v2(
        self, response: ChatResponse, documents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Get the generation info from cohere API response (V2)."""
        generation_info: Dict[str, Any] = {
            "id": response.id,
            "finish_reason": response.finish_reason,
        }

        if documents:
            generation_info["documents"] = documents

        if response.message:
            if response.message.tool_plan:
                generation_info["tool_plan"] = response.message.tool_plan
            if response.message.tool_calls:
                generation_info["tool_calls"] = _format_cohere_tool_calls_v2(
                    response.message.tool_calls
                )
            if response.message.content:
                generation_info["content"] = response.message.content[0].text
            if response.message.citations:
                generation_info["citations"] = response.message.citations

        if response.usage:
            if response.usage.tokens:
                generation_info["token_count"] = response.usage.tokens.dict()

        return generation_info

    def _get_stream_info_v2(
        self,
        final_delta: Any,
        documents: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Get the stream info from cohere API response (V2)."""
        input_tokens = final_delta.usage.billed_units.input_tokens
        output_tokens = final_delta.usage.billed_units.output_tokens
        total_tokens = input_tokens + output_tokens
        stream_info = {
            "finish_reason": final_delta.finish_reason,
            "token_count": {
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        if documents:
            stream_info["documents"] = documents
        if tool_calls:
            stream_info["tool_calls"] = tool_calls
        return stream_info

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        request = get_cohere_chat_request_v2(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        response = self.client.v2.chat(**request)

        generation_info = self._get_generation_info_v2(
            response, request.get("documents")
        )
        if "tool_calls" in generation_info:
            content = response.message.tool_plan if response.message.tool_plan else ""
            tool_calls = [
                lc_tool_call
                for tool_call in response.message.tool_calls
                if (
                    lc_tool_call := _convert_cohere_v2_tool_call_to_langchain(tool_call)
                )
            ]
        else:
            content = (
                response.message.content[0].text if response.message.content else ""
            )
            tool_calls = []
        usage_metadata = _get_usage_metadata_v2(response)
        message = AIMessage(
            content=content,
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
        )
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        request = get_cohere_chat_request_v2(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )

        response = await self.async_client.v2.chat(**request)

        generation_info = self._get_generation_info_v2(
            response, request.get("documents")
        )
        if "tool_calls" in generation_info:
            content = response.message.tool_plan if response.message.tool_plan else ""
            tool_calls = [
                lc_tool_call
                for tool_call in response.tool_calls
                if (
                    lc_tool_call := _convert_cohere_v2_tool_call_to_langchain(tool_call)
                )
            ]
        else:
            content = (
                response.message.content[0].text if response.message.content else ""
            )
            tool_calls = []
        usage_metadata = _get_usage_metadata_v2(response)
        message = AIMessage(
            content=content,
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
        )
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

    @property
    def model_name(self) -> str:
        if self.model is not None:
            return self.model
        if self._default_model_name is None:
            self._default_model_name = self._get_default_model()
        return self._default_model_name

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        model = self.model_name
        return len(self.client.tokenize(text=text, model=model).tokens)


def _format_cohere_tool_calls(
    tool_calls: Optional[List[ToolCall]] = None,
) -> List[Dict]:
    """
    Formats a Cohere API response into the tool call format used elsewhere in Langchain.
    """
    if not tool_calls:
        return []

    formatted_tool_calls = []
    for tool_call in tool_calls:
        formatted_tool_calls.append(
            {
                "id": uuid.uuid4().hex[:],
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.parameters),
                },
            }
        )
    return formatted_tool_calls


def _format_cohere_tool_calls_v2(
    tool_calls: Optional[List[ToolCallV2]] = None,
) -> List[Dict[str, Any]]:
    """
    Formats a V2 Cohere API response into the tool
    call format used elsewhere in Langchain.
    """
    if not tool_calls:
        return []

    formatted_tool_calls = []
    for tool_call in tool_calls:
        if not tool_call.function:
            continue

        formatted_tool_calls.append(
            {
                "id": tool_call.id or uuid.uuid4().hex[:],
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    return formatted_tool_calls


def _convert_cohere_tool_call_to_langchain(tool_call: ToolCall) -> LC_ToolCall:
    """Convert a Cohere tool call into langchain_core.messages.ToolCall"""
    _id = uuid.uuid4().hex[:]
    return LC_ToolCall(name=tool_call.name, args=tool_call.parameters, id=_id)


def _convert_cohere_v2_tool_call_to_langchain(
    tool_call: ToolCallV2,
) -> Optional[LC_ToolCall]:
    """Convert a Cohere V2 tool call into langchain_core.messages.ToolCall"""
    _id = tool_call.id or uuid.uuid4().hex[:]
    if not tool_call.function or not tool_call.function.name:
        return None
    return LC_ToolCall(
        name=str(tool_call.function.name),
        args=json.loads(tool_call.function.arguments)
        if tool_call.function.arguments
        else {},
        id=_id,
    )


def _get_usage_metadata(response: NonStreamedChatResponse) -> Optional[UsageMetadata]:
    """Get standard usage metadata from chat response."""
    metadata = response.meta
    if metadata:
        if tokens := metadata.tokens:
            input_tokens = int(tokens.input_tokens or 0)
            output_tokens = int(tokens.output_tokens or 0)
            total_tokens = input_tokens + output_tokens
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    return None


def _get_usage_metadata_v2(response: ChatResponse) -> Optional[UsageMetadata]:
    """Get standard usage metadata from chat response."""
    metadata = response.usage
    if metadata:
        if tokens := metadata.tokens:
            input_tokens = int(tokens.input_tokens or 0)
            output_tokens = int(tokens.output_tokens or 0)
            total_tokens = input_tokens + output_tokens
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    return None
