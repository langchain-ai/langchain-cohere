"""Unit tests for reasoning trace functionality in ChatCohere."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from cohere import (
    NonStreamedChatResponse,
    Usage,
    UsageBilledUnits,
    UsageTokens,
)
from langchain_core.messages import AIMessage, HumanMessage

from langchain_cohere.chat_models import ChatCohere


def test_reasoning_trace_extraction(patch_base_cohere_get_default_model: Any) -> None:
    """Test that reasoning traces are properly extracted as content blocks."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Let me think step by step"),
        MagicMock(type="text", text="Final answer here", thinking=None),
    ]
    mock_response.message.tool_plan = None
    mock_response.message.tool_calls = []
    mock_response.message.citations = None

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 10
    mock_response.usage.tokens.output_tokens = 20
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 10
    mock_response.usage.billed_units.output_tokens = 20
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result = llm._generate(messages=[HumanMessage(content="Test prompt")])

    # Content should be a list of content blocks matching OpenAI format
    content = result.generations[0].message.content
    assert isinstance(content, list)
    assert len(content) == 2

    # First block is reasoning
    reasoning_block = content[0]
    assert isinstance(reasoning_block, dict)
    assert reasoning_block["type"] == "reasoning"
    assert reasoning_block["id"] == "test-id"
    assert len(reasoning_block["summary"]) == 1
    assert reasoning_block["summary"][0]["text"] == "Let me think step by step"

    # Second block is text
    text_block = content[1]
    assert isinstance(text_block, dict)
    assert text_block["type"] == "text"
    assert text_block["text"] == "Final answer here"

    # Thinking should NOT be in additional_kwargs
    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert "thinking" not in generation_info
    assert "content" not in generation_info


def test_reasoning_trace_only(patch_base_cohere_get_default_model: Any) -> None:
    """Test when response contains only thinking content."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Thinking only")
    ]
    mock_response.message.tool_plan = None
    mock_response.message.tool_calls = []
    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 5
    mock_response.usage.tokens.output_tokens = 15
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 5
    mock_response.usage.billed_units.output_tokens = 15
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result: Any = llm._generate(messages=[HumanMessage(content="Test prompt")])

    # Content should be a list with just the reasoning block (no text block)
    content = result.generations[0].message.content
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "reasoning"
    assert content[0]["summary"][0]["text"] == "Thinking only"


def test_reasoning_trace_with_tool_calls(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test reasoning trace extraction when tool calls are present."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Thinking with tools"),
        MagicMock(type="text", text="Final answer"),
    ]
    mock_response.message.tool_plan = "Plan to use tools"
    mock_response.message.citations = None

    # Create a proper mock for tool calls that returns strings
    tool_call_mock = MagicMock()
    tool_call_mock.id = "call-1"
    tool_call_mock.function = MagicMock()
    tool_call_mock.function.name = "tool1"
    tool_call_mock.function.arguments = '{"arg": "value"}'
    mock_response.message.tool_calls = [tool_call_mock]

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 15
    mock_response.usage.tokens.output_tokens = 25
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 15
    mock_response.usage.billed_units.output_tokens = 25
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result: Any = llm._generate(messages=[HumanMessage(content="Test prompt")])

    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert "tool_calls" in generation_info
    # When tool calls are present, content is the tool_plan (string)
    assert result.generations[0].message.content == "Plan to use tools"


def test_tool_calls_preserved_in_generate(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test that tool calls are NOT reset to empty list after extraction."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = []
    mock_response.message.tool_plan = "I will call the search tool"
    mock_response.message.citations = None

    # Create mock tool call
    tool_call_mock = MagicMock()
    tool_call_mock.id = "call-123"
    tool_call_mock.function = MagicMock()
    tool_call_mock.function.name = "search"
    tool_call_mock.function.arguments = '{"query": "test"}'
    mock_response.message.tool_calls = [tool_call_mock]

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 10
    mock_response.usage.tokens.output_tokens = 20
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 10
    mock_response.usage.billed_units.output_tokens = 20
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result = llm._generate(messages=[HumanMessage(content="Search for test")])

    # CRITICAL: Verify tool_calls is NOT empty
    message = cast(AIMessage, result.generations[0].message)
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "search"


@pytest.mark.asyncio
async def test_agenerate_extracts_thinking(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test that _agenerate extracts thinking content into content blocks."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Step by step reasoning"),
        MagicMock(type="text", text="Final answer", thinking=None),
    ]
    mock_response.message.tool_plan = None
    mock_response.message.tool_calls = []
    mock_response.message.citations = None

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 10
    mock_response.usage.tokens.output_tokens = 20
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 10
    mock_response.usage.billed_units.output_tokens = 20
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.async_client = MagicMock()
    llm.async_client.v2.chat = AsyncMock(return_value=mock_response)

    result = await llm._agenerate(messages=[HumanMessage(content="Test")])

    # Content should be a list of content blocks
    content = result.generations[0].message.content
    assert isinstance(content, list)
    assert len(content) == 2
    reasoning = content[0]
    assert isinstance(reasoning, dict)
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == "Step by step reasoning"
    text = content[1]
    assert isinstance(text, dict)
    assert text["type"] == "text"
    assert text["text"] == "Final answer"


def test_multiple_content_items_both_extracted(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test response with both thinking and text content extracts both."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Let me think..."),
        MagicMock(type="text", text="Here is the answer", thinking=None),
    ]
    mock_response.message.tool_plan = None
    mock_response.message.tool_calls = []
    mock_response.message.citations = None

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 10
    mock_response.usage.tokens.output_tokens = 20
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 10
    mock_response.usage.billed_units.output_tokens = 20
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result = llm._generate(messages=[HumanMessage(content="Test")])

    # Content should be a list of content blocks
    content = result.generations[0].message.content
    assert isinstance(content, list)
    assert len(content) == 2
    reasoning = content[0]
    assert isinstance(reasoning, dict)
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == "Let me think..."
    text = content[1]
    assert isinstance(text, dict)
    assert text["type"] == "text"
    assert text["text"] == "Here is the answer"

    # Thinking should NOT be in generation_info
    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert "thinking" not in generation_info


def test_tool_calls_with_thinking_content(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test that tool calls work correctly alongside thinking content."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="I need to search for this"),
    ]
    mock_response.message.tool_plan = "Search for information"
    mock_response.message.citations = None

    tool_call_mock = MagicMock()
    tool_call_mock.id = "call-456"
    tool_call_mock.function = MagicMock()
    tool_call_mock.function.name = "web_search"
    tool_call_mock.function.arguments = '{"q": "weather"}'
    mock_response.message.tool_calls = [tool_call_mock]

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 15
    mock_response.usage.tokens.output_tokens = 25
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 15
    mock_response.usage.billed_units.output_tokens = 25
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result = llm._generate(messages=[HumanMessage(content="What's the weather?")])

    # Both should be present
    message = cast(AIMessage, result.generations[0].message)
    assert len(message.tool_calls) == 1
    # When tool calls are present, content is the tool_plan (string)
    assert result.generations[0].message.content == "Search for information"


@pytest.mark.asyncio
async def test_agenerate_with_tool_calls_and_thinking(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test that _agenerate extracts both thinking and tool calls correctly."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="I need to call the tool"),
        MagicMock(type="text", text="Using the tool now", thinking=None),
    ]
    mock_response.message.tool_plan = "I will search for the answer"
    mock_response.message.citations = None

    tool_call_mock = MagicMock()
    tool_call_mock.id = "call-async-1"
    tool_call_mock.function = MagicMock()
    tool_call_mock.function.name = "web_search"
    tool_call_mock.function.arguments = '{"query": "async test"}'
    mock_response.message.tool_calls = [tool_call_mock]

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 12
    mock_response.usage.tokens.output_tokens = 30
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 12
    mock_response.usage.billed_units.output_tokens = 30
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-async-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.async_client = MagicMock()
    llm.async_client.v2.chat = AsyncMock(return_value=mock_response)

    result = await llm._agenerate(messages=[HumanMessage(content="Search async")])

    # Verify tool calls are present
    message = cast(AIMessage, result.generations[0].message)
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "web_search"
    assert message.tool_calls[0]["args"] == {"query": "async test"}

    # When tool calls are present, content should be the tool_plan
    assert message.content == "I will search for the answer"

    # Verify generation_info has tool_calls but NOT thinking
    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert "tool_calls" in generation_info
    assert "thinking" not in generation_info

    # Verify usage_metadata is populated correctly
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] == 12
    assert message.usage_metadata["output_tokens"] == 30
    assert message.usage_metadata["total_tokens"] == 42


def test_usage_metadata_correctly_mapped_from_response(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test that _get_usage_metadata_v2 correctly maps token counts
    from the Cohere response onto LangChain UsageMetadata."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Reasoning about the problem"),
        MagicMock(type="text", text="The answer is 42", thinking=None),
    ]
    mock_response.message.tool_plan = None
    mock_response.message.tool_calls = []
    mock_response.message.citations = None

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 25
    mock_response.usage.tokens.output_tokens = 150  # Higher due to reasoning tokens
    mock_response.usage.tokens.dict.return_value = {
        "input_tokens": 25,
        "output_tokens": 150,
    }
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 25
    mock_response.usage.billed_units.output_tokens = 150
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-usage-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat.return_value = mock_response

    result = llm._generate(messages=[HumanMessage(content="Think hard about this")])

    # Verify content is a list of content blocks
    message = cast(AIMessage, result.generations[0].message)
    content = message.content
    assert isinstance(content, list)
    assert len(content) == 2
    reasoning = content[0]
    assert isinstance(reasoning, dict)
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == "Reasoning about the problem"
    text = content[1]
    assert isinstance(text, dict)
    assert text["type"] == "text"
    assert text["text"] == "The answer is 42"

    # Verify thinking is NOT in generation_info
    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert "thinking" not in generation_info

    # Verify usage_metadata is correctly populated
    usage = message.usage_metadata
    assert usage is not None, "usage_metadata should not be None"
    assert (
        usage["input_tokens"] == 25
    ), f"Expected input_tokens=25, got {usage['input_tokens']}"
    assert (
        usage["output_tokens"] == 150
    ), f"Expected output_tokens=150, got {usage['output_tokens']}"
    assert (
        usage["total_tokens"] == 175
    ), f"Expected total_tokens=175, got {usage['total_tokens']}"
    # Verify total = input + output
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    # Verify token_count is also in generation_info (raw API data)
    assert "token_count" in generation_info
    token_count = generation_info["token_count"]
    assert token_count["input_tokens"] == 25
    assert token_count["output_tokens"] == 150


@pytest.mark.asyncio
async def test_agenerate_usage_metadata_correctly_mapped(
    patch_base_cohere_get_default_model: Any,
) -> None:
    """Test that _agenerate correctly maps usage metadata including
    reasoning token counts."""
    mock_response = MagicMock(spec=NonStreamedChatResponse)
    mock_response.message = MagicMock()
    mock_response.message.content = [
        MagicMock(type="thinking", text=None, thinking="Deep reasoning here"),
        MagicMock(type="text", text="Final result", thinking=None),
    ]
    mock_response.message.tool_plan = None
    mock_response.message.tool_calls = []
    mock_response.message.citations = None

    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.tokens = MagicMock(spec=UsageTokens)
    mock_response.usage.tokens.input_tokens = 18
    mock_response.usage.tokens.output_tokens = 200  # High: includes reasoning tokens
    mock_response.usage.billed_units = MagicMock(spec=UsageBilledUnits)
    mock_response.usage.billed_units.input_tokens = 18
    mock_response.usage.billed_units.output_tokens = 200
    mock_response.finish_reason = "COMPLETE"
    mock_response.id = "test-async-usage-id"

    llm = ChatCohere(cohere_api_key="test-key")
    llm.async_client = MagicMock()
    llm.async_client.v2.chat = AsyncMock(return_value=mock_response)

    result = await llm._agenerate(messages=[HumanMessage(content="Reason about this")])

    message = cast(AIMessage, result.generations[0].message)

    # Verify content is a list of content blocks
    content = message.content
    assert isinstance(content, list)
    assert len(content) == 2
    reasoning = content[0]
    assert isinstance(reasoning, dict)
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == "Deep reasoning here"
    text = content[1]
    assert isinstance(text, dict)
    assert text["type"] == "text"
    assert text["text"] == "Final result"

    # Verify thinking is NOT in generation_info
    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert "thinking" not in generation_info

    # Verify usage_metadata
    usage = message.usage_metadata
    assert usage is not None, "usage_metadata should not be None"
    assert usage["input_tokens"] == 18
    assert usage["output_tokens"] == 200
    assert usage["total_tokens"] == 218
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
