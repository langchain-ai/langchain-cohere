import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from cohere.types import NonStreamedChatResponse, ToolCallV2
from cohere.types.chat_response import ChatResponse
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
    _format_to_cohere_tools,
)
from langchain_cohere.llms import BaseCohere
from langchain_cohere.react_multi_hop.prompt import convert_to_documents

if TYPE_CHECKING:
    from cohere.types import ListModelsResponse  # noqa: F401


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
        return "user"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, ToolMessage):
        return "tool"
    else:
        raise ValueError(f"Got unknown type {type(message).__name__}")


def _get_message_cohere_format(
    message: BaseMessage,
) -> Dict[
    str,
    Union[
        str,
        List[LC_ToolCall],
        List[ToolCallV2],
        List[Union[str, Dict[Any, Any]]],
        List[Dict[Any, Any]],
        None,
    ],
]:
    """Get the formatted message as required in cohere's api.

    Args:
        message: The BaseMessage.

    Returns:
        The formatted message as required in cohere's api.
    """

    if isinstance(message, AIMessage) and message.tool_calls:
        return {
            "role": get_role(message),
            "content": message.content,
            "tool_calls": message.tool_calls,
            "tool_plan": message.additional_kwargs["tool_plan"],
        }
    elif isinstance(message, HumanMessage) or isinstance(message, SystemMessage):
        return {"role": get_role(message), "content": message.content}
    elif isinstance(message, ToolMessage):
        return {
            "role": get_role(message),
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    else:
        raise ValueError(f"Got unknown type {message}")


def get_cohere_chat_request(
    messages: List[BaseMessage],
    *,
    documents: Optional[List[Document]] = None,
    stop_sequences: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the request for the Cohere chat API.

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
                        "data": {"text": parsed_doc.page_content},
                        "id": parsed_doc.metadata.get("id") or f"doc-{str(i)}",
                    }
                )
            elif isinstance(parsed_doc, dict):
                formatted_docs.append(parsed_doc)

    messages_formated = []
    for i, message in enumerate(messages):
        messages_formated.append(_get_message_cohere_format(message))

    req = {
        "messages": messages_formated,
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
        formatted_tools = _format_to_cohere_tools(tools)
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict.

        Returns:
            A Runnable that takes any ChatModel input and returns either a dict or
            Pydantic class as output.
        """
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
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

        return llm | output_parser

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "cohere-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        base_params = {"model": self.model, "temperature": self.temperature}
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

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        stream = self.client.chat_stream(**request)
        for data in stream:
            if data.type == "content-delta":
                content = data.delta.message.content.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk
            if data.type == "tool-call-start" or data.type == "tool-call-delta":
                delta = data.delta.tool_call
                if delta:
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
                                index=data.index,
                            )
                        ],
                    )
                    chunk = ChatGenerationChunk(message=message)
                    if run_manager:
                        run_manager.on_llm_new_token(delta, chunk=chunk)
                    yield chunk
            elif data.type == "message-end":
                usage_metadata = _get_usage_metadata(data.delta)
                message = AIMessageChunk(content="", usage_metadata=usage_metadata)
                yield ChatGenerationChunk(
                    message=message, usage_metadata=usage_metadata
                )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        stream = self.async_client.chat_stream(**request)
        async for data in stream:
            if data.type == "content-delta":
                content = data.delta.message.content.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk
            if data.type == "tool-call-start" or data.type == "tool-call-delta":
                delta = data.delta.tool_call
                if delta:
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
                                index=data.index,
                            )
                        ],
                    )
                    chunk = ChatGenerationChunk(message=message)
                    if run_manager:
                        run_manager.on_llm_new_token(delta, chunk=chunk)
                    yield chunk
            elif data.type == "message-end":
                usage_metadata = _get_usage_metadata(data.delta)
                message = AIMessageChunk(content="", usage_metadata=usage_metadata)
                yield ChatGenerationChunk(
                    message=message, usage_metadata=usage_metadata
                )

    def _get_generation_info(self, response: ChatResponse) -> Dict[str, Any]:
        """Get the generation info from cohere API response."""
        generation_info: Dict[str, Any] = {
            "id": response.id,
            "citations": response.message.citations,
            "finish_reason": response.finish_reason,
        }
        if response.message.tool_calls:
            generation_info["tool_calls"] = _format_cohere_tool_calls(
                response.message.tool_calls
            )
            generation_info["tool_plan"] = response.message.tool_plan
        return generation_info

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

        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        request["model"] = self.model
        request["temperature"] = self.temperature
        response = self.client.chat(**request)
        generation_info = self._get_generation_info(response)
        if "tool_calls" in generation_info:
            tool_calls = [
                _convert_cohere_tool_call_to_langchain(tool_call)
                for tool_call in response.message.tool_calls
            ]
        else:
            tool_calls = []
        usage_metadata = _get_usage_metadata(response)

        message = AIMessage(
            content=response.message.content[0].text
            if response.message.content
            else "",
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
            return generate_from_stream(stream_iter)

        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        request["model"] = self.model
        request["temperature"] = self.temperature
        response = await self.async_client.chat(**request)
        generation_info = self._get_generation_info(response)
        if "tool_calls" in generation_info:
            tool_calls = [
                _convert_cohere_tool_call_to_langchain(tool_call)
                for tool_call in response.message.tool_calls
            ]
        else:
            tool_calls = []
        usage_metadata = _get_usage_metadata(response)

        message = AIMessage(
            content=response.message.content[0].text
            if response.message.content
            else "",
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
        )
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

    def _get_default_model(self) -> str:
        """Fetches the current default model name."""
        response = self.client.models.list(default_only=True, endpoint="chat")  # type: "ListModelsResponse"
        if not response.models:
            raise Exception("invalid cohere list models response")
        if not response.models[0].name:
            raise Exception("invalid cohere list models response")
        return response.models[0].name

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
    tool_calls: Optional[List[ToolCallV2]] = None,
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
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": json.dumps(tool_call.function.arguments),
                },
                "type": "function",
            }
        )
    return formatted_tool_calls


def _convert_cohere_tool_call_to_langchain(tool_call: ToolCallV2) -> LC_ToolCall:
    """Convert a Cohere tool call into langchain_core.messages.ToolCall"""

    return LC_ToolCall(
        name=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        id=tool_call.id,
    )


def _get_usage_metadata(response: ChatResponse) -> Optional[UsageMetadata]:
    """Get standard usage metadata from chat response."""
    if usage := response.usage:
        input_tokens = int(usage.tokens.input_tokens or 0)
        output_tokens = int(usage.tokens.output_tokens or 0)
        total_tokens = input_tokens + output_tokens
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
