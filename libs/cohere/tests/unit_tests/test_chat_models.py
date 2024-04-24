"""Test chat model integration."""
import typing
from unittest.mock import patch

import pytest
from cohere.types import NonStreamedChatResponse, ToolCall
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_cohere.chat_models import ChatCohere, _messages_to_cohere_tool_results


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
    results = _messages_to_cohere_tool_results(messages)
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
    results = _messages_to_cohere_tool_results(messages)
    assert results == expected
