"""Test chat model integration."""

import typing
from unittest.mock import patch

import pytest
from cohere.types import NonStreamedChatResponse, ToolCall
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langchain_cohere.chat_models import (
    ChatCohere,
    _messages_to_cohere_tool_results_curr_chat_turn,
    get_cohere_chat_request,
)


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatCohere(cohere_api_key="test")


@pytest.mark.parametrize(
    "chat_cohere,expected",
    [
        pytest.param(ChatCohere(cohere_api_key="test"), {}, id="defaults"),
        pytest.param(
            ChatCohere(
                cohere_api_key="test", model="foo", temperature=1.0, preamble="bar"
            ),
            {
                "model": "foo",
                "temperature": 1.0,
                "preamble": "bar",
            },
            id="values are set",
        ),
    ],
)
def test_default_params(chat_cohere: ChatCohere, expected: typing.Dict) -> None:
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
    response: typing.Any, expected: typing.Dict[str, typing.Any]
) -> None:
    chat_cohere = ChatCohere(cohere_api_key="test")

    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "foo"
        actual = chat_cohere._get_generation_info(response)

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
    "cohere_client,messages,force_single_step,expected",
    [
        pytest.param(
            ChatCohere(cohere_api_key="test"),
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
            ChatCohere(cohere_api_key="test"),
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
            ChatCohere(cohere_api_key="test"),
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
            ChatCohere(cohere_api_key="test"),
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
                            {
                                "name": "magic_function",
                                "args": {"a": 12},
                                "id": "bbec5f815a0f4c609ccb36e98c4f0455",
                            }
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
    cohere_client: ChatCohere,
    messages: typing.List[BaseMessage],
    force_single_step: bool,
    expected: typing.Dict[str, typing.Any],
) -> None:
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
