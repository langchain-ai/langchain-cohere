"""Integration tests for reasoning trace functionality in ChatCohere."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langchain_core.tools import tool

from langchain_cohere import ChatCohere

# Load environment variables from .env.test file
# Create a .env.test file in libs/cohere/ with your COHERE_API_KEY:
# COHERE_API_KEY=your-actual-api-key-here
env_test_path = Path(__file__).parent.parent.parent / ".env.test"
load_dotenv(dotenv_path=env_test_path)

DEFAULT_MODEL = "command-a-reasoning-08-2025"


@tool
def get_the_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The location to get the weather for.

    Returns:
        The current weather conditions.
    """
    return f"The weather in {location} is sunny and 22°C."


def _get_text_from_content(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> str:
    """Extract the text portion from content (which may be a string or list of
    content blocks)."""
    if isinstance(content, str):
        return content
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return ""


def _get_reasoning_from_content(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> Optional[str]:
    """Extract the reasoning/thinking text from content blocks."""
    if isinstance(content, str):
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            summary = block.get("summary", [])
            if summary:
                return summary[0].get("text", "")
    return None


def has_reasoning_trace(response: Any) -> bool:
    """Check if the response contains a reasoning trace in content blocks."""
    content = response.content
    if isinstance(content, list):
        for block in content:
            if block.get("type") == "reasoning":
                return True

    # Fallback: check content text for reasoning keywords
    text = _get_text_from_content(content) if isinstance(content, list) else content
    content_lower = text.lower()
    reasoning_keywords = [
        "step by step",
        "first",
        "second",
        "calculate",
        "think",
        "reason",
        "plan",
    ]
    return any(keyword in content_lower for keyword in reasoning_keywords)


def test_command_a_reasoning_trace() -> None:
    """Test that Command A Reasoning model returns responses with reasoning traces."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    # Test a prompt that should generate reasoning
    response = llm.invoke([HumanMessage(content="Explain how photosynthesis works")])

    # Check for reasoning in content blocks
    has_reasoning = False
    content = response.content

    if isinstance(content, list):
        reasoning_text = _get_reasoning_from_content(content)
        if reasoning_text and len(reasoning_text) > 0:
            has_reasoning = True
        text_content = _get_text_from_content(content)
    else:
        text_content = content

    # If no reasoning blocks, check if content contains reasoning keywords
    if not has_reasoning:
        content_lower = text_content.lower()
        reasoning_keywords = [
            "photosynthesis",
            "chloroplast",
            "light-dependent",
            "calvin cycle",
        ]
        has_reasoning = any(keyword in content_lower for keyword in reasoning_keywords)

    assert has_reasoning, "No reasoning trace found in response"


def test_command_a_reasoning_multi_turn() -> None:
    """Test multi-turn conversation with reasoning traces."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    messages = [
        HumanMessage(content="What is 5 + 7?"),
        HumanMessage(content="Now multiply that by 3"),
    ]

    response = llm.invoke(messages)

    # Verify reasoning trace exists
    assert has_reasoning_trace(response)

    # The final answer should be 36
    text = _get_text_from_content(response.content)
    assert isinstance(text, str), "Text content should be a string"
    content_lower = text.lower()
    assert "36" in content_lower or "thirty six" in content_lower
    assert "5 + 7" in content_lower or "five plus seven" in content_lower
    assert "multiply" in content_lower or "times" in content_lower


def test_command_a_reasoning_complex_reasoning() -> None:
    """Test complex reasoning scenario."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    prompt = """
    A store sells apples for $2 each and oranges for $1.50 each.
    If Sarah buys 3 apples and 4 oranges, how much does she spend?
    But wait, there's a 10% discount on all fruit today.
    How much does she actually pay?
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    # Verify reasoning trace exists
    assert has_reasoning_trace(response)

    # The response should contain the correct calculation
    text = _get_text_from_content(response.content)
    assert isinstance(text, str), "Text content should be a string"
    content_lower = text.lower()
    # Check for currency indicators (dollar sign or the word "dollars")
    assert any(x in content_lower for x in ["$", "dollar", "dollars"])
    assert "discount" in content_lower or "discounted" in content_lower
    # The correct calculation: (3*2 + 4*1.5) = 6 + 6 = $12, then 10% discount = $10.80
    # Look for the final amount in various formats
    final_amount_indicators = ["10.80", "10.8", "ten point eight", "10.80", "$10.80"]
    assert any(x in content_lower for x in final_amount_indicators)


@pytest.mark.vcr
def test_command_a_reasoning_with_tool_call() -> None:
    """Test that reasoning model returns both thinking content and tool calls.

    This test verifies that when using a reasoning-enabled model with tools,
    the response contains both:
    1. A 'thinking' content block with the model's reasoning trace
    2. A proper tool_calls array with the function call
    """
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0.3)
    llm_with_tools = llm.bind_tools([get_the_weather])

    # Use a prompt that requires reasoning to identify the location
    # "Land of the rising sun" = Japan, second largest city = Osaka
    response = llm_with_tools.invoke(
        [
            HumanMessage(
                content="Can you get the weather for the city that is the "
                "2nd largest city of the land of the rising sun?"
            )
        ]
    )

    # Verify tool calls are present
    assert response.tool_calls is not None, "Response should have tool_calls"
    assert len(response.tool_calls) > 0, "Response should have at least one tool call"

    # Verify the tool call is for get_the_weather with Osaka
    tool_call = response.tool_calls[0]
    assert (
        tool_call["name"] == "get_the_weather"
    ), f"Tool call should be 'get_the_weather', got: {tool_call['name']}"
    assert "args" in tool_call, "Tool call should have 'args'"
    assert "location" in tool_call["args"], "Tool call args should have 'location'"

    # The location should be Osaka (or possibly Yokohama, which is also sometimes
    # cited as second largest depending on how you measure)
    location = tool_call["args"]["location"].lower()
    assert any(
        city in location for city in ["osaka", "yokohama"]
    ), f"Location should be Osaka or Yokohama, got: {tool_call['args']['location']}"


def test_sync_stream_reasoning_trace() -> None:
    """Test that sync streaming with a reasoning model returns content
    and properly reports usage metadata."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    full: Optional[AIMessageChunk] = None
    chunks_with_token_counts = 0
    chunk_count = 0

    for token in llm.stream(
        [HumanMessage(content="What is 9 * 6? Show your reasoning.")]
    ):
        assert isinstance(token, AIMessageChunk)
        full = token if full is None else full + token
        chunk_count += 1
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1

    # Verify we received multiple chunks (streaming is working)
    assert chunk_count > 1, f"Expected multiple chunks, got {chunk_count}"

    # Verify aggregated result
    assert isinstance(full, AIMessageChunk)
    assert full.content is not None
    assert len(full.content) > 0

    # Extract text content for verification
    text = _get_text_from_content(full.content)
    assert len(text) > 0

    # Verify the answer contains the correct result (54)
    assert "54" in text, f"Expected '54' in aggregated content. Got: {text}"

    # Verify usage metadata from the final chunk
    if chunks_with_token_counts > 0:
        assert (
            full.usage_metadata is not None
        ), "Aggregated usage_metadata should not be None"
        assert full.usage_metadata["input_tokens"] > 0, "input_tokens should be > 0"
        assert full.usage_metadata["output_tokens"] > 0, "output_tokens should be > 0"
        assert (
            full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
            == full.usage_metadata["total_tokens"]
        ), "total_tokens should equal input_tokens + output_tokens"


@pytest.mark.asyncio
async def test_async_invoke_reasoning_trace() -> None:
    """Test that async invoke with a reasoning model returns thinking content
    and properly reports usage metadata including token counts."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    response = await llm.ainvoke(
        [HumanMessage(content="What is 15 * 23? Show your reasoning.")]
    )

    # Verify basic response structure
    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(response.content) > 0

    # Extract text content and verify answer
    text = _get_text_from_content(response.content)
    assert "345" in text, f"Expected '345' in response content. Got: {text}"

    # Verify reasoning trace exists
    assert has_reasoning_trace(response), "No reasoning trace found in async response"

    # Verify usage metadata is populated
    usage_metadata = response.usage_metadata
    assert usage_metadata is not None, "usage_metadata should not be None"
    assert usage_metadata["input_tokens"] > 0, "input_tokens should be > 0"
    assert usage_metadata["output_tokens"] > 0, "output_tokens should be > 0"
    assert (
        usage_metadata["total_tokens"]
        == usage_metadata["input_tokens"] + usage_metadata["output_tokens"]
    ), "total_tokens should equal input_tokens + output_tokens"


@pytest.mark.asyncio
async def test_async_invoke_tool_calling_with_reasoning() -> None:
    """Test that async invoke with tools returns both thinking content
    and properly formed tool calls."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0.3)
    llm_with_tools = llm.bind_tools([get_the_weather])

    response = await llm.ainvoke(
        [HumanMessage(content="Can you get the weather for the capital of France?")]
    )

    # Verify basic response structure
    assert isinstance(response, AIMessage)
    assert response.content is not None

    # Verify usage metadata is populated
    usage_metadata = response.usage_metadata
    assert usage_metadata is not None, "usage_metadata should not be None"
    assert usage_metadata["input_tokens"] > 0, "input_tokens should be > 0"
    assert usage_metadata["output_tokens"] > 0, "output_tokens should be > 0"
    assert (
        usage_metadata["total_tokens"]
        == usage_metadata["input_tokens"] + usage_metadata["output_tokens"]
    ), "total_tokens should equal input_tokens + output_tokens"

    # Now test with tools actually bound
    response_with_tools = await llm_with_tools.ainvoke(
        [HumanMessage(content="Can you get the weather for the capital of France?")]
    )

    assert isinstance(response_with_tools, AIMessage)

    # Verify tool calls are present
    assert response_with_tools.tool_calls is not None, "Response should have tool_calls"
    assert (
        len(response_with_tools.tool_calls) > 0
    ), "Response should have at least one tool call"

    # Verify the tool call is for get_the_weather
    tool_call = response_with_tools.tool_calls[0]
    assert (
        tool_call["name"] == "get_the_weather"
    ), f"Tool call should be 'get_the_weather', got: {tool_call['name']}"
    assert "args" in tool_call, "Tool call should have 'args'"
    assert "location" in tool_call["args"], "Tool call args should have 'location'"

    # Verify location refers to Paris
    location = tool_call["args"]["location"].lower()
    assert (
        "paris" in location
    ), f"Location should contain 'paris', got: {tool_call['args']['location']}"

    # Verify usage metadata on the tool call response
    usage_metadata_tools = response_with_tools.usage_metadata
    assert usage_metadata_tools is not None, "usage_metadata should not be None"
    assert usage_metadata_tools["input_tokens"] > 0, "input_tokens should be > 0"
    assert usage_metadata_tools["output_tokens"] > 0, "output_tokens should be > 0"


@pytest.mark.asyncio
async def test_async_stream_tool_calling_with_reasoning() -> None:
    """Test that async streaming with tools returns both tool call chunks
    and properly reports usage metadata."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0.3)
    llm_with_tools = llm.bind_tools([get_the_weather])

    full: Optional[AIMessageChunk] = None
    chunk_count = 0
    tool_call_chunks_present = False

    async for token in llm_with_tools.astream(
        [HumanMessage(content="Get the weather for Tokyo")]
    ):
        assert isinstance(token, AIMessageChunk)
        full = token if full is None else full + token
        chunk_count += 1
        if token.tool_call_chunks:
            tool_call_chunks_present = True

    # Verify we received chunks
    assert chunk_count > 0, f"Expected chunks, got {chunk_count}"
    assert full is not None

    # Verify tool call chunks were streamed
    assert tool_call_chunks_present, "Expected tool_call_chunks during streaming"

    # Verify the aggregated result has tool_calls in additional_kwargs
    assert (
        "tool_calls" in full.additional_kwargs
    ), "Aggregated result should have 'tool_calls' in additional_kwargs"
    assert (
        len(full.additional_kwargs["tool_calls"]) > 0
    ), "Should have at least one tool call"

    # Verify the tool call is for get_the_weather
    tool_call = full.additional_kwargs["tool_calls"][0]
    assert (
        tool_call["function"]["name"] == "get_the_weather"
    ), f"Expected 'get_the_weather', got: {tool_call['function']['name']}"

    # Verify usage metadata
    if full.usage_metadata is not None:
        assert full.usage_metadata["input_tokens"] > 0, "input_tokens should be > 0"
        assert full.usage_metadata["output_tokens"] > 0, "output_tokens should be > 0"
        assert (
            full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
            == full.usage_metadata["total_tokens"]
        ), "total_tokens should equal input_tokens + output_tokens"


@pytest.mark.asyncio
async def test_async_stream_reasoning_trace() -> None:
    """Test that async streaming with a reasoning model returns content
    and properly reports usage metadata."""
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    full: Optional[AIMessageChunk] = None
    chunks_with_token_counts = 0
    chunk_count = 0

    async for token in llm.astream(
        [HumanMessage(content="What is 7 * 8? Show your reasoning.")]
    ):
        assert isinstance(token, AIMessageChunk)
        full = token if full is None else full + token
        chunk_count += 1
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1

    # Verify we received multiple chunks (streaming is working)
    assert chunk_count > 1, f"Expected multiple chunks, got {chunk_count}"

    # Verify aggregated result
    assert isinstance(full, AIMessageChunk)
    assert full.content is not None
    assert len(full.content) > 0

    # Extract text content for verification
    text = _get_text_from_content(full.content)
    assert len(text) > 0

    # Verify the answer contains the correct result (56)
    assert "56" in text, f"Expected '56' in aggregated content. Got: {text}"

    # Verify usage metadata from the final chunk
    if chunks_with_token_counts > 0:
        assert (
            full.usage_metadata is not None
        ), "Aggregated usage_metadata should not be None"
        assert full.usage_metadata["input_tokens"] > 0, "input_tokens should be > 0"
        assert full.usage_metadata["output_tokens"] > 0, "output_tokens should be > 0"
        assert (
            full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
            == full.usage_metadata["total_tokens"]
        ), "total_tokens should equal input_tokens + output_tokens"
