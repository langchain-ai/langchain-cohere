import json
import uuid
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

from cohere.types import NonStreamedChatResponse, ToolCall
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
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
    ToolMessage,
)
from langchain_core.messages import (
    ToolCall as LC_ToolCall,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_cohere.cohere_agent import (
    _convert_to_cohere_tool,
    _format_to_cohere_tools,
)
from langchain_cohere.llms import BaseCohere
from langchain_cohere.react_multi_hop.prompt import convert_to_documents


def _messages_to_cohere_tool_results(
    messages: List[BaseMessage],
) -> List[Dict[str, Any]]:
    """Get tool_results from messages."""
    tool_results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            tool_message = message
            previous_ai_msgs = [
                message
                for message in messages[:-1]
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
        return "User"
    elif isinstance(message, AIMessage):
        return "Chatbot"
    elif isinstance(message, SystemMessage):
        return "System"
    else:
        raise ValueError(f"Got unknown type {message}")


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
            if len(additional_kwargs["documents"]) > 0
            else None
        )
    elif documents is not None and len(documents) > 0:
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

    tool_results: Optional[List[Dict[str, Any]]] = _messages_to_cohere_tool_results(
        messages
    )
    if not tool_results:
        tool_results = None

    chat_history = [
        {"role": get_role(x), "message": x.content}
        for x in messages[:-1]
        if not isinstance(x, ToolMessage)
        and not (isinstance(x, AIMessage) and x.tool_calls)
    ]
    req = {
        "message": messages[-1].content,
        "chat_history": chat_history,
        "tool_results": tool_results,
        "documents": formatted_docs,
        "connectors": connectors,
        "prompt_truncation": prompt_truncation,
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

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = _format_to_cohere_tools(tools)
        return super().bind(tools=formatted_tools, **kwargs)

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
            elif data.event_type == "stream-end":
                generation_info = self._get_generation_info(data.response)
                tool_call_chunks = []
                if tool_calls := generation_info.get("tool_calls"):
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
                message = AIMessageChunk(
                    content="",
                    additional_kwargs=generation_info,
                    tool_call_chunks=tool_call_chunks,
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
        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )

        if hasattr(self.async_client, "chat_stream"):  # detect and support sdk v5
            stream = self.async_client.chat_stream(**request)
        else:
            stream = self.async_client.chat(**request, stream=True)

        async for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            elif data.event_type == "stream-end":
                generation_info = self._get_generation_info(data.response)
                tool_call_chunks = []
                if tool_calls := generation_info.get("tool_calls"):
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
                message = AIMessageChunk(
                    content="",
                    additional_kwargs=generation_info,
                    tool_call_chunks=tool_call_chunks,
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
        response = self.client.chat(**request)

        generation_info = self._get_generation_info(response)
        if "tool_calls" in generation_info:
            tool_calls = [
                _convert_cohere_tool_call_to_langchain(tool_call)
                for tool_call in response.tool_calls
            ]
        else:
            tool_calls = []
        message = AIMessage(
            content=response.text,
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
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

        request = get_cohere_chat_request(
            messages, stop_sequences=stop, **self._default_params, **kwargs
        )
        response = self.client.chat(**request)

        generation_info = self._get_generation_info(response)
        if "tool_calls" in generation_info:
            tool_calls = [
                _convert_cohere_tool_call_to_langchain(tool_call)
                for tool_call in response.tool_calls
            ]
        else:
            tool_calls = []
        message = AIMessage(
            content=response.text,
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
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

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        model: str
        if self.model is not None:
            model = self.model
        elif self._default_model_name is not None:
            model = self._default_model_name
        else:
            model = self._get_default_model()
            self._default_model_name = model

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
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.parameters),
                },
                "type": "function",
            }
        )
    return formatted_tool_calls


def _convert_cohere_tool_call_to_langchain(tool_call: ToolCall) -> LC_ToolCall:
    """Convert a Cohere tool call into langchain_core.messages.ToolCall"""
    _id = uuid.uuid4().hex[:]
    return LC_ToolCall(name=tool_call.name, args=tool_call.parameters, id=_id)
