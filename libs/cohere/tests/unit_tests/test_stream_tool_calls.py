"""Unit tests for tool call streaming in ChatCohere."""

from functools import reduce
from types import SimpleNamespace
from typing import Any, AsyncIterator, List
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from langchain_cohere.chat_models import ChatCohere

EXPECTED_TOOL_CALLS = [
    {
        "name": "get_weather",
        "args": {"city": "Paris"},
        "id": "call_1",
        "type": "tool_call",
    }
]


def _tool_call_event(event_type: str, **function: Any) -> SimpleNamespace:
    tool_calls = SimpleNamespace(id="call_1", function=SimpleNamespace(**function))
    return SimpleNamespace(
        type=event_type,
        index=0,
        delta=SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls)),
    )


def _stream_events() -> List[SimpleNamespace]:
    billed_units = SimpleNamespace(input_tokens=3, output_tokens=5)
    return [
        SimpleNamespace(
            type="tool-plan-delta",
            delta=SimpleNamespace(
                message=SimpleNamespace(tool_plan="Check the weather")
            ),
        ),
        _tool_call_event("tool-call-start", name="get_weather", arguments=""),
        _tool_call_event("tool-call-delta", arguments='{"city": '),
        _tool_call_event("tool-call-delta", arguments='"Paris"}'),
        SimpleNamespace(type="tool-call-end", index=0),
        SimpleNamespace(
            type="message-end",
            delta=SimpleNamespace(
                finish_reason="TOOL_CALL",
                usage=SimpleNamespace(billed_units=billed_units),
            ),
        ),
    ]


async def _async_events() -> AsyncIterator[SimpleNamespace]:
    for event in _stream_events():
        yield event


def _merge(chunks: List[Any]) -> AIMessageChunk:
    merged = reduce(lambda left, right: left + right, [c.message for c in chunks])
    assert isinstance(merged, AIMessageChunk)
    return merged


def test_stream_tool_calls(patch_base_cohere_get_default_model: Any) -> None:
    llm = ChatCohere(cohere_api_key="test-key")
    llm.client = MagicMock()
    llm.client.v2.chat_stream.return_value = iter(_stream_events())
    run_manager = MagicMock()

    chunks = list(
        llm._stream(
            [HumanMessage(content="weather in Paris?")], run_manager=run_manager
        )
    )

    merged = _merge(chunks)
    assert merged.tool_calls == EXPECTED_TOOL_CALLS
    assert merged.content == "Check the weather"
    assert merged.usage_metadata is not None
    assert merged.usage_metadata["total_tokens"] == 8
    assert run_manager.on_llm_new_token.call_count == 5


@pytest.mark.asyncio
async def test_astream_tool_calls_match_stream(
    patch_base_cohere_get_default_model: Any,
) -> None:
    llm = ChatCohere(cohere_api_key="test-key")
    llm.async_client = MagicMock()
    llm.async_client.v2.chat_stream.return_value = _async_events()

    chunks = [
        chunk
        async for chunk in llm._astream([HumanMessage(content="weather in Paris?")])
    ]

    merged = _merge(chunks)
    assert merged.tool_calls == EXPECTED_TOOL_CALLS
    assert merged.content == "Check the weather"
