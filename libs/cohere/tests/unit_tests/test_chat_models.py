"""Test chat model integration."""

from typing import Any, Dict, Generator, List, Optional
from unittest.mock import patch

import pytest
from cohere import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    ChatMessageEndEventDelta,
    ChatResponse,
    NonStreamedChatResponse,
    SystemChatMessageV2,
    ToolCall,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    ToolV2,
    ToolV2Function,
    Usage,
    UsageBilledUnits,
    UsageTokens,
    UserChatMessageV2,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pytest import WarningsRecorder

from langchain_cohere.chat_models import (
    BaseCohere,
    ChatCohere,
    _messages_to_cohere_tool_results_curr_chat_turn,
    get_cohere_chat_request,
    get_cohere_chat_request_v2,
)
from langchain_cohere.cohere_agent import (
    _format_to_cohere_tools_v2,
)


def test_initialization(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
) -> None:
    """Test chat model initialization."""
    ChatCohere(cohere_api_key="test")


@pytest.mark.parametrize(
    "chat_cohere_kwargs,expected",
    [
        pytest.param(
            {"cohere_api_key": "test"}, {"model": "command-r-plus"}, id="defaults"
        ),
        pytest.param(
            {
                "cohere_api_key": "test",
                "model": "foo",
                "temperature": 1.0,
                "preamble": "bar",
            },
            {
                "model": "foo",
                "temperature": 1.0,
                "preamble": "bar",
            },
            id="values are set",
        ),
    ],
)
def test_default_params(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    chat_cohere_kwargs: Dict[str, Any],
    expected: Dict,
) -> None:
    chat_cohere = ChatCohere(**chat_cohere_kwargs)
    actual = chat_cohere._default_params
    assert expected == actual


@pytest.mark.parametrize(
    "response, expected",
    [
        pytest.param(
            NonStreamedChatResponse(
                generation_id="foo",
                text="",
                tool_calls=[
                    ToolCall(name="tool1", parameters={"arg1": 1, "arg2": "2"}),
                    ToolCall(name="tool2", parameters={"arg3": 3, "arg4": "4"}),
                ],
            ),
            {
                "documents": None,
                "citations": None,
                "search_results": None,
                "search_queries": None,
                "is_search_required": None,
                "generation_id": "foo",
                "tool_calls": [
                    {
                        "id": "foo",
                        "function": {
                            "name": "tool1",
                            "arguments": '{"arg1": 1, "arg2": "2"}',
                        },
                        "type": "function",
                    },
                    {
                        "id": "foo",
                        "function": {
                            "name": "tool2",
                            "arguments": '{"arg3": 3, "arg4": "4"}',
                        },
                        "type": "function",
                    },
                ],
            },
            id="tools should be called",
        ),
        pytest.param(
            NonStreamedChatResponse(
                generation_id="foo",
                text="",
                tool_calls=[],
            ),
            {
                "documents": None,
                "citations": None,
                "search_results": None,
                "search_queries": None,
                "is_search_required": None,
                "generation_id": "foo",
            },
            id="no tools should be called",
        ),
        pytest.param(
            NonStreamedChatResponse(
                generation_id="foo",
                text="bar",
                tool_calls=[],
            ),
            {
                "documents": None,
                "citations": None,
                "search_results": None,
                "search_queries": None,
                "is_search_required": None,
                "generation_id": "foo",
            },
            id="chat response without tools/documents/citations/tools etc",
        ),
    ],
)
def test_get_generation_info(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    response: Any,
    expected: Dict[str, Any],
) -> None:
    chat_cohere = ChatCohere(cohere_api_key="test")

    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "foo"
        actual = chat_cohere._get_generation_info(response)

    assert expected == actual


@pytest.mark.parametrize(
    "response, documents, expected",
    [
        pytest.param(
            ChatResponse(
                id="foo",
                finish_reason="complete",
                message=AssistantMessageResponse(
                    tool_plan="I will use the magic_function tool to answer the question.",  # noqa: E501
                    tool_calls=[
                        ToolCallV2(
                            function=ToolCallV2Function(
                                name="tool1", arguments='{"arg1": 1, "arg2": "2"}'
                            )
                        ),
                        ToolCallV2(
                            function=ToolCallV2Function(
                                name="tool2", arguments='{"arg3": 3, "arg4": "4"}'
                            )
                        ),
                    ],
                    content=None,
                    citations=None,
                ),
                usage=Usage(tokens={"input_tokens": 215, "output_tokens": 38}),
            ),
            None,
            {
                "id": "foo",
                "finish_reason": "complete",
                "tool_plan": "I will use the magic_function tool to answer the question.",  # noqa: E501
                "tool_calls": [
                    {
                        "id": "foo",
                        "function": {
                            "name": "tool1",
                            "arguments": '{"arg1": 1, "arg2": "2"}',
                        },
                        "type": "function",
                    },
                    {
                        "id": "foo",
                        "function": {
                            "name": "tool2",
                            "arguments": '{"arg3": 3, "arg4": "4"}',
                        },
                        "type": "function",
                    },
                ],
                "token_count": {
                    "input_tokens": 215,
                    "output_tokens": 38,
                },
            },
            id="tools should be called",
        ),
        pytest.param(
            ChatResponse(
                id="foo",
                finish_reason="complete",
                message=AssistantMessageResponse(
                    tool_plan=None,
                    tool_calls=[],
                    content=[],
                    citations=None,
                ),
            ),
            None,
            {
                "id": "foo",
                "finish_reason": "complete",
            },
            id="no tools should be called",
        ),
        pytest.param(
            ChatResponse(
                id="foo",
                finish_reason="complete",
                message=AssistantMessageResponse(
                    tool_plan=None,
                    tool_calls=[],
                    content=[{"type": "text", "text": "How may I help you today?"}],
                    citations=None,
                ),
                usage=Usage(tokens={"input_tokens": 215, "output_tokens": 38}),
            ),
            None,
            {
                "id": "foo",
                "finish_reason": "complete",
                "content": "How may I help you today?",
                "token_count": {
                    "input_tokens": 215,
                    "output_tokens": 38,
                },
            },
            id="chat response without tools/documents/citations/tools etc",
        ),
        pytest.param(
            ChatResponse(
                id="foo",
                finish_reason="complete",
                message=AssistantMessageResponse(
                    tool_plan=None,
                    tool_calls=[],
                    content=[{"type": "text", "text": "How may I help you today?"}],
                    citations=None,
                ),
                usage=Usage(tokens={"input_tokens": 215, "output_tokens": 38}),
            ),
            [
                {
                    "id": "doc-1",
                    "data": {
                        "text": "doc-1 content",
                    },
                },
                {
                    "id": "doc-2",
                    "data": {
                        "text": "doc-2 content",
                    },
                },
            ],
            {
                "id": "foo",
                "finish_reason": "complete",
                "documents": [
                    {
                        "id": "doc-1",
                        "data": {
                            "text": "doc-1 content",
                        },
                    },
                    {
                        "id": "doc-2",
                        "data": {
                            "text": "doc-2 content",
                        },
                    },
                ],
                "content": "How may I help you today?",
                "token_count": {
                    "input_tokens": 215,
                    "output_tokens": 38,
                },
            },
            id="chat response with documents",
        ),
    ],
)
def test_get_generation_info_v2(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    response: ChatResponse,
    documents: Optional[List[Dict[str, Any]]],
    expected: Dict[str, Any],
) -> None:
    chat_cohere = ChatCohere(cohere_api_key="test")
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "foo"
        actual = chat_cohere._get_generation_info_v2(response, documents)
    assert expected == actual


@pytest.mark.parametrize(
    "final_delta, documents, tool_calls, expected",
    [
        pytest.param(
            ChatMessageEndEventDelta(
                finish_reason="complete",
                usage=Usage(
                    tokens=UsageTokens(input_tokens=215, output_tokens=38),
                    billed_units=UsageBilledUnits(input_tokens=215, output_tokens=38),
                ),
            ),
            None,
            None,
            {
                "finish_reason": "complete",
                "token_count": {
                    "input_tokens": 215.0,
                    "output_tokens": 38.0,
                    "total_tokens": 253.0,
                },
            },
            id="message-end no documents no tools",
        ),
        pytest.param(
            ChatMessageEndEventDelta(
                finish_reason="complete",
                usage=Usage(
                    tokens=UsageTokens(input_tokens=215, output_tokens=38),
                    billed_units=UsageBilledUnits(input_tokens=215, output_tokens=38),
                ),
            ),
            [{"id": "foo", "snippet": "some text"}],
            None,
            {
                "finish_reason": "complete",
                "token_count": {
                    "input_tokens": 215.0,
                    "output_tokens": 38.0,
                    "total_tokens": 253.0,
                },
                "documents": [
                    {
                        "id": "foo",
                        "snippet": "some text",
                    }
                ],
            },
            id="message-end with documents",
        ),
        pytest.param(
            ChatMessageEndEventDelta(
                finish_reason="complete",
                usage=Usage(
                    tokens=UsageTokens(input_tokens=215, output_tokens=38),
                    billed_units=UsageBilledUnits(input_tokens=215, output_tokens=38),
                ),
            ),
            None,
            [
                {
                    "id": "foo",
                    "type": "function",
                    "function": {
                        "name": "bar",
                        "arguments": "{'a': 1}",
                    },
                }
            ],
            {
                "finish_reason": "complete",
                "token_count": {
                    "input_tokens": 215.0,
                    "output_tokens": 38.0,
                    "total_tokens": 253.0,
                },
                "tool_calls": [
                    {
                        "id": "foo",
                        "type": "function",
                        "function": {
                            "name": "bar",
                            "arguments": "{'a': 1}",
                        },
                    }
                ],
            },
            id="message-end with tool_calls",
        ),
        pytest.param(
            ChatMessageEndEventDelta(
                finish_reason="complete",
                usage=Usage(
                    tokens=UsageTokens(input_tokens=215, output_tokens=38),
                    billed_units=UsageBilledUnits(input_tokens=215, output_tokens=38),
                ),
            ),
            [{"id": "foo", "snippet": "some text"}],
            [
                {
                    "id": "foo",
                    "type": "function",
                    "function": {
                        "name": "bar",
                        "arguments": "{'a': 1}",
                    },
                }
            ],
            {
                "finish_reason": "complete",
                "token_count": {
                    "input_tokens": 215.0,
                    "output_tokens": 38.0,
                    "total_tokens": 253.0,
                },
                "documents": [{"id": "foo", "snippet": "some text"}],
                "tool_calls": [
                    {
                        "id": "foo",
                        "type": "function",
                        "function": {
                            "name": "bar",
                            "arguments": "{'a': 1}",
                        },
                    }
                ],
            },
            id="message-end with documents and tool_calls",
        ),
    ],
)
def test_get_stream_info_v2(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    final_delta: ChatMessageEndEventDelta,
    documents: Optional[List[Dict[str, Any]]],
    tool_calls: Optional[List[Dict[str, Any]]],
    expected: Dict[str, Any],
) -> None:
    chat_cohere = ChatCohere(cohere_api_key="test")
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "foo"
        actual = chat_cohere._get_stream_info_v2(
            final_delta=final_delta, documents=documents, tool_calls=tool_calls
        )
    assert expected == actual


def test_messages_to_cohere_tool_results() -> None:
    human_message = HumanMessage(content="what is the value of magic_function(3)?")
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "magic_function",
                "args": {"input": 3},
                "id": "d86e6098-21e1-44c7-8431-40cfc6d35590",
            }
        ],
    )
    tool_message = ToolMessage(
        name="magic_function",
        content="5",
        tool_call_id="d86e6098-21e1-44c7-8431-40cfc6d35590",
    )
    messages = [human_message, ai_message, tool_message]
    results = _messages_to_cohere_tool_results_curr_chat_turn(messages)
    expected = [
        {
            "call": ToolCall(name="magic_function", parameters={"input": 3}),
            "outputs": [{"output": "5"}],
        }
    ]
    assert results == expected

    another_tool_message = ToolMessage(
        content="5",
        additional_kwargs={"name": "magic_function"},
        tool_call_id="d86e6098-21e1-44c7-8431-40cfc6d35590",
    )
    messages = [human_message, ai_message, another_tool_message]
    results = _messages_to_cohere_tool_results_curr_chat_turn(messages)
    assert results == expected


@pytest.mark.parametrize(
    "cohere_client_kwargs,messages,force_single_step,expected",
    [
        pytest.param(
            {"cohere_api_key": "test"},
            [HumanMessage(content="what is magic_function(12) ?")],
            True,
            {
                "message": "what is magic_function(12) ?",
                "chat_history": [],
                "force_single_step": True,
                "tools": [
                    {
                        "name": "magic_function",
                        "description": "Does a magical operation to a number.",
                        "parameter_definitions": {
                            "a": {"description": "", "type": "int", "required": True}
                        },
                    }
                ],
            },
            id="Single Message and force_single_step is True",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            [
                HumanMessage(content="what is magic_function(12) ?"),
                AIMessage(
                    content="I will use the magic_function tool to answer the question.",  # noqa: E501
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    id="run-8039f73d-2e50-4eec-809e-e3690a6d3a9a-0",
                    tool_calls=[
                        {
                            "name": "magic_function",
                            "args": {"a": 12},
                            "id": "e81dbae6937e47e694505f81e310e205",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="112", tool_call_id="e81dbae6937e47e694505f81e310e205"
                ),
            ],
            True,
            {
                "message": "what is magic_function(12) ?",
                "chat_history": [
                    {"role": "User", "message": "what is magic_function(12) ?"}
                ],
                "tool_results": [
                    {
                        "call": ToolCall(name="magic_function", parameters={"a": 12}),
                        "outputs": [{"output": "112"}],
                    }
                ],
                "force_single_step": True,
                "tools": [
                    {
                        "name": "magic_function",
                        "description": "Does a magical operation to a number.",
                        "parameter_definitions": {
                            "a": {"description": "", "type": "int", "required": True}
                        },
                    }
                ],
            },
            id="Multiple Messages with tool results and force_single_step is True",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            [HumanMessage(content="what is magic_function(12) ?")],
            False,
            {
                "message": "what is magic_function(12) ?",
                "chat_history": [],
                "tools": [
                    {
                        "name": "magic_function",
                        "description": "Does a magical operation to a number.",
                        "parameter_definitions": {
                            "a": {"description": "", "type": "int", "required": True}
                        },
                    }
                ],
                "force_single_step": False,
            },
            id="Single Message and force_single_step is False",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            [
                HumanMessage(content="what is magic_function(12) ?"),
                AIMessage(
                    content="I will use the magic_function tool to answer the question.",  # noqa: E501
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "91588a40-684d-40f9-ae87-e27c3b4cda87",
                        "tool_calls": [
                            {
                                "id": "0ba15e974d2b4bf6bf74ba8e4b268a7a",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"input_tokens": 912, "output_tokens": 22},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "91588a40-684d-40f9-ae87-e27c3b4cda87",
                        "tool_calls": [
                            {
                                "id": "0ba15e974d2b4bf6bf74ba8e4b268a7a",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"input_tokens": 912, "output_tokens": 22},
                    },
                    id="run-148af4fb-adf0-4f0c-b209-bffcde9a5f58-0",
                    tool_calls=[
                        {
                            "name": "magic_function",
                            "args": {"a": 12},
                            "id": "bbec5f815a0f4c609ccb36e98c4f0455",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="112", tool_call_id="bbec5f815a0f4c609ccb36e98c4f0455"
                ),
            ],
            False,
            {
                "message": "",
                "chat_history": [
                    {"role": "User", "message": "what is magic_function(12) ?"},
                    {
                        "role": "Chatbot",
                        "message": "I will use the magic_function tool to answer the question.",  # noqa: E501
                        "tool_calls": [
                            ToolCall(
                                name="magic_function",
                                parameters={"a": 12},
                                id="bbec5f815a0f4c609ccb36e98c4f0455",
                            )
                        ],
                    },
                ],
                "tool_results": [
                    {
                        "call": ToolCall(name="magic_function", parameters={"a": 12}),
                        "outputs": [{"output": "112"}],
                    }
                ],
                "tools": [
                    {
                        "name": "magic_function",
                        "description": "Does a magical operation to a number.",
                        "parameter_definitions": {
                            "a": {"description": "", "type": "int", "required": True}
                        },
                    }
                ],
                "force_single_step": False,
            },
            id="Multiple Messages with tool results and force_single_step is False",
        ),
    ],
)
def test_get_cohere_chat_request(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    cohere_client_kwargs: Dict[str, Any],
    messages: List[BaseMessage],
    force_single_step: bool,
    expected: Dict[str, Any],
) -> None:
    cohere_client = ChatCohere(**cohere_client_kwargs)

    tools = [
        {
            "name": "magic_function",
            "description": "Does a magical operation to a number.",
            "parameter_definitions": {
                "a": {"description": "", "type": "int", "required": True}
            },
        }
    ]

    result = get_cohere_chat_request(
        messages,
        stop_sequences=cohere_client.stop,
        force_single_step=force_single_step,
        tools=tools,
    )

    # Check that the result is a dictionary
    assert isinstance(result, dict)
    assert result == expected


@pytest.mark.parametrize(
    "cohere_client_v2_kwargs,preamble,messages,get_chat_req_kwargs,expected",
    [
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [HumanMessage(content="what is magic_function(12) ?")],
            {},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="what is magic_function(12) ?",
                    )
                ],
            },
            id="Single message",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            "You are a wizard, with the ability to perform magic using the magic_function tool.",  # noqa: E501
            [HumanMessage(content="what is magic_function(12) ?")],
            {},
            {
                "messages": [
                    SystemChatMessageV2(
                        role="system",
                        content="You are a wizard, with the ability to perform magic using the magic_function tool.",  # noqa: E501
                    ),
                    UserChatMessageV2(
                        role="user",
                        content="what is magic_function(12) ?",
                    ),
                ],
            },
            id="Single message with preamble",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [
                HumanMessage(content="Hello!"),
                AIMessage(
                    content="Hello, how may I assist you?",  # noqa: E501
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "91588a40-684d-40f9-ae87-e27c3b4cda87",
                        "tool_calls": None,
                        "token_count": {"input_tokens": 912, "output_tokens": 22},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "91588a40-684d-40f9-ae87-e27c3b4cda87",
                        "tool_calls": None,
                        "token_count": {"input_tokens": 912, "output_tokens": 22},
                    },
                    id="run-148af4fb-adf0-4f0c-b209-bffcde9a5f58-0",
                ),
                HumanMessage(content="Remember my name, its Bob."),
            ],
            {},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="Hello!",
                    ),
                    AssistantChatMessageV2(
                        role="assistant",
                        content="Hello, how may I assist you?",
                    ),
                    UserChatMessageV2(
                        role="user",
                        content="Remember my name, its Bob.",
                    ),
                ],
            },
            id="Multiple messages no tool usage",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            "You are a wizard, with the ability to perform magic using the magic_function tool.",  # noqa: E501
            [
                HumanMessage(content="Hello!"),
                AIMessage(
                    content="Hello, how may I assist you?",  # noqa: E501
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "91588a40-684d-40f9-ae87-e27c3b4cda87",
                        "tool_calls": None,
                        "token_count": {"input_tokens": 912, "output_tokens": 22},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "91588a40-684d-40f9-ae87-e27c3b4cda87",
                        "tool_calls": None,
                        "token_count": {"input_tokens": 912, "output_tokens": 22},
                    },
                    id="run-148af4fb-adf0-4f0c-b209-bffcde9a5f58-0",
                ),
                HumanMessage(content="Remember my name, its Bob."),
            ],
            {},
            {
                "messages": [
                    SystemChatMessageV2(
                        role="system",
                        content="You are a wizard, with the ability to perform magic using the magic_function tool.",  # noqa: E501
                    ),
                    UserChatMessageV2(
                        role="user",
                        content="Hello!",
                    ),
                    AssistantChatMessageV2(
                        role="assistant",
                        content="Hello, how may I assist you?",
                    ),
                    UserChatMessageV2(
                        role="user",
                        content="Remember my name, its Bob.",
                    ),
                ],
            },
            id="Multiple messages no tool usage, with preamble",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [
                HumanMessage(content="what is magic_function(12) ?"),
                AIMessage(
                    content="I will use the magic_function tool to answer the question.",  # noqa: E501
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    id="run-8039f73d-2e50-4eec-809e-e3690a6d3a9a-0",
                    tool_calls=[
                        {
                            "name": "magic_function",
                            "args": {"a": 12},
                            "id": "e81dbae6937e47e694505f81e310e205",
                        }
                    ],
                ),
                ToolMessage(
                    content="112", tool_call_id="e81dbae6937e47e694505f81e310e205"
                ),
            ],
            {},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="what is magic_function(12) ?",
                    ),
                    AssistantChatMessageV2(
                        role="assistant",
                        tool_plan="I will use the magic_function tool to answer the question.",  # noqa: E501
                        tool_calls=[
                            ToolCallV2(
                                id="e81dbae6937e47e694505f81e310e205",
                                type="function",
                                function=ToolCallV2Function(
                                    name="magic_function",
                                    arguments='{"a": 12}',
                                ),
                            )
                        ],
                    ),
                    ToolChatMessageV2(
                        role="tool",
                        tool_call_id="e81dbae6937e47e694505f81e310e205",
                        content=[
                            {
                                "type": "document",
                                "document": {"data": {"output": "112"}},
                            }
                        ],
                    ),
                ],
            },
            id="Multiple messages with tool usage",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            "You are a wizard, with the ability to perform magic using the magic_function tool.",  # noqa: E501
            [
                HumanMessage(content="what is magic_function(12) ?"),
                AIMessage(
                    content="I will use the magic_function tool to answer the question.",  # noqa: E501
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    id="run-8039f73d-2e50-4eec-809e-e3690a6d3a9a-0",
                    tool_calls=[
                        {
                            "name": "magic_function",
                            "args": {"a": 12},
                            "id": "e81dbae6937e47e694505f81e310e205",
                        }
                    ],
                ),
                ToolMessage(
                    content="112", tool_call_id="e81dbae6937e47e694505f81e310e205"
                ),
            ],
            {},
            {
                "messages": [
                    SystemChatMessageV2(
                        role="system",
                        content="You are a wizard, with the ability to perform magic using the magic_function tool.",  # noqa: E501
                    ),
                    UserChatMessageV2(
                        role="user",
                        content="what is magic_function(12) ?",
                    ),
                    AssistantChatMessageV2(
                        role="assistant",
                        tool_plan="I will use the magic_function tool to answer the question.",  # noqa: E501
                        tool_calls=[
                            ToolCallV2(
                                id="e81dbae6937e47e694505f81e310e205",
                                type="function",
                                function=ToolCallV2Function(
                                    name="magic_function",
                                    arguments='{"a": 12}',
                                ),
                            )
                        ],
                    ),
                    ToolChatMessageV2(
                        role="tool",
                        tool_call_id="e81dbae6937e47e694505f81e310e205",
                        content=[
                            {
                                "type": "document",
                                "document": {"data": {"output": "112"}},
                            }
                        ],
                    ),
                ],
            },
            id="Multiple messages with tool usage, and preamble",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [
                HumanMessage(content="what is magic_function(12) ?"),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    response_metadata={
                        "documents": None,
                        "citations": None,
                        "search_results": None,
                        "search_queries": None,
                        "is_search_required": None,
                        "generation_id": "b8e48c51-4340-4081-b505-5d51e78493ab",
                        "tool_calls": [
                            {
                                "id": "976f79f68d8342139d8397d6c89688c4",
                                "function": {
                                    "name": "magic_function",
                                    "arguments": '{"a": 12}',
                                },
                                "type": "function",
                            }
                        ],
                        "token_count": {"output_tokens": 9},
                    },
                    id="run-8039f73d-2e50-4eec-809e-e3690a6d3a9a-0",
                    tool_calls=[
                        {
                            "name": "magic_function",
                            "args": {"a": 12},
                            "id": "e81dbae6937e47e694505f81e310e205",
                        }
                    ],
                ),
                ToolMessage(
                    content="112", tool_call_id="e81dbae6937e47e694505f81e310e205"
                ),
            ],
            {},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="what is magic_function(12) ?",
                    ),
                    AssistantChatMessageV2(
                        role="assistant",
                        tool_plan="I will assist you using the tools provided.",
                        tool_calls=[
                            ToolCallV2(
                                id="e81dbae6937e47e694505f81e310e205",
                                type="function",
                                function=ToolCallV2Function(
                                    name="magic_function",
                                    arguments='{"a": 12}',
                                ),
                            )
                        ],
                    ),
                    ToolChatMessageV2(
                        role="tool",
                        tool_call_id="e81dbae6937e47e694505f81e310e205",
                        content=[
                            {
                                "type": "document",
                                "document": {"data": {"output": "112"}},
                            }
                        ],
                    ),
                ],
            },
            id="assistent message with tool calls empty content",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [
                HumanMessage(content="foo"),
            ],
            {"documents": ["bar"]},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="foo",
                    ),
                ],
                "documents": ["bar"],
            },
            id="document in kwargs (string)",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [
                HumanMessage(content="foo"),
            ],
            {"documents": [{"data": "bar"}]},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="foo",
                    ),
                ],
                "documents": [{"data": "bar"}],
            },
            id="document in kwargs (dict)",
        ),
        pytest.param(
            {"cohere_api_key": "test"},
            None,
            [
                HumanMessage(content="foo"),
            ],
            {"documents": [Document(page_content="bar")]},
            {
                "messages": [
                    UserChatMessageV2(
                        role="user",
                        content="foo",
                    ),
                ],
                "documents": [{"data": {"text": "bar"}, "id": "doc-0"}],
            },
            id="document in kwargs (Document)",
        ),
    ],
)
def test_get_cohere_chat_request_v2_tools(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    cohere_client_v2_kwargs: Dict[str, Any],
    preamble: str,
    messages: List[BaseMessage],
    get_chat_req_kwargs: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    cohere_client_v2 = ChatCohere(**cohere_client_v2_kwargs)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "magic_function",
                "description": "Does a magical operation to a number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "int",
                            "description": "",
                        }
                    },
                    "required": ["a"],
                },
            },
        }
    ]
    expected["tools"] = tools

    result = get_cohere_chat_request_v2(
        messages,
        stop_sequences=cohere_client_v2.stop,
        tools=tools,
        preamble=preamble,
        **get_chat_req_kwargs,
    )

    # Check that the result is a dictionary
    assert isinstance(result, dict)
    assert result == expected


def test_format_to_cohere_tools_v2() -> None:
    @tool
    def add_two_numbers(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b

    @tool
    def capital_cities(country: str) -> str:
        """Returns the capital city of a country"""
        return "France"

    tools = [add_two_numbers, capital_cities]
    result = _format_to_cohere_tools_v2(tools)

    expected = [
        ToolV2(
            type="function",
            function=ToolV2Function(
                name="add_two_numbers",
                description="Add two numbers together",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {
                            "description": "",
                            "type": "integer",
                        },
                        "b": {
                            "description": "",
                            "type": "integer",
                        },
                    },
                    "required": [
                        "a",
                        "b",
                    ],
                },
            ),
        ),
        ToolV2(
            type="function",
            function=ToolV2Function(
                name="capital_cities",
                description="Returns the capital city of a country",
                parameters={
                    "type": "object",
                    "properties": {
                        "country": {
                            "description": "",
                            "type": "string",
                        },
                    },
                    "required": [
                        "country",
                    ],
                },
            ),
        ),
    ]

    assert result == expected


def test_get_cohere_chat_request_v2_warn_connectors_deprecated(
    recwarn: WarningsRecorder,
) -> None:
    messages: List[BaseMessage] = [HumanMessage(content="Hello")]
    kwargs: Dict[str, Any] = {"connectors": ["some_connector"]}

    with pytest.raises(
        ValueError,
        match="The 'connectors' parameter is deprecated as of version 0.4.0.",
    ):
        get_cohere_chat_request_v2(messages, **kwargs)

    assert len(recwarn) == 1
    warning = recwarn.pop(DeprecationWarning)
    assert issubclass(warning.category, DeprecationWarning)
    assert "The 'connectors' parameter is deprecated as of version 0.4.0." in str(
        warning.message
    )
    assert "Please use the 'tools' parameter instead." in str(warning.message)


@pytest.mark.parametrize(
    "tool_choice_input,expected_tool_choice",
    [
        pytest.param("any", "REQUIRED", id="any -> REQUIRED"),
        pytest.param("ANY", "REQUIRED", id="ANY -> REQUIRED"),
        pytest.param("none", "NONE", id="none -> NONE"),
        pytest.param("NONE", "NONE", id="NONE -> NONE"),
        pytest.param("auto", None, id="auto -> removed"),
        pytest.param("AUTO", None, id="AUTO -> removed"),
        pytest.param("REQUIRED", "REQUIRED", id="REQUIRED -> REQUIRED (unchanged)"),
        pytest.param(None, None, id="None -> None (no tool_choice)"),
    ],
)
def test_bind_tools_tool_choice_translation(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
    tool_choice_input: Optional[str],
    expected_tool_choice: Optional[str],
) -> None:
    """Test that bind_tools() translates OpenAI-style tool_choice to Cohere format."""

    @tool
    def test_tool(x: int) -> int:
        """Test tool."""
        return x + 1

    chat = ChatCohere(cohere_api_key="test")

    # Bind tools with tool_choice
    if tool_choice_input is not None:
        bound = chat.bind_tools([test_tool], tool_choice=tool_choice_input)
    else:
        bound = chat.bind_tools([test_tool])

    # Check the bound kwargs
    if expected_tool_choice is not None:
        assert bound.kwargs.get("tool_choice") == expected_tool_choice
    else:
        assert "tool_choice" not in bound.kwargs
