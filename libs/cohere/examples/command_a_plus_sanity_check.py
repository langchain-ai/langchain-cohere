"""Sanity checks for Command A+ (``command-a-plus-05-2026``) via langchain-cohere.

This script exercises the *existing* ``langchain_cohere`` code against the
Command A+ model to demonstrate that the current library already supports it.
It covers the model's headline capabilities:

  * Plain text generation (sync invoke, streaming, async)
  * Multi-turn conversations
  * Reasoning / thinking traces
  * Vision (image input) via URL and base64 data URI
  * Vision + reasoning combined in a single request
  * Tool calling (and tool calling + reasoning)
  * Structured output (JSON schema) and JSON mode
  * Grounded generation with documents + citations
  * Token counting

Usage:
    export COHERE_API_KEY=your-key
    python libs/cohere/examples/command_a_plus_sanity_check.py

    # Optional overrides:
    python libs/cohere/examples/command_a_plus_sanity_check.py \
        --model command-a-plus-05-2026 \
        --image https://your-image-url.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import time
import traceback
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_cohere import ChatCohere

DEFAULT_MODEL = "command-a-plus-05-2026"
DEFAULT_IMAGE_URL = "https://cohere.com/favicon-32x32.png"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _content_to_text(content: Union[str, List[Union[str, Dict[str, Any]]]]) -> str:
    """Extract the plain-text portion of a message content."""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(p for p in parts if p)


def _reasoning_from_content(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> Optional[str]:
    """Extract the reasoning/thinking text from content blocks, if present."""
    if not isinstance(content, list):
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            summary = block.get("summary", [])
            if summary:
                return summary[0].get("text", "")
    return None


def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _image_as_data_uri(url: str) -> str:
    """Download an image and return it as a base64 data URI."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        raw = resp.read()
        content_type = resp.headers.get("Content-Type", "image/png")
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city to get the weather for.
    """
    return f"The weather in {location} is sunny and 22 degrees Celsius."


class PersonInfo(BaseModel):
    """Information about a person."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job")


# --------------------------------------------------------------------------- #
# Individual checks. Each returns a short human-readable result string and
# raises on failure (the runner catches and records the failure).
# --------------------------------------------------------------------------- #
def check_token_count(llm: ChatCohere) -> str:
    n = llm.get_num_tokens("Hello, Command A+!")
    assert n > 0, "expected a positive token count"
    return f"tokenized 'Hello, Command A+!' -> {n} tokens"


def check_basic_invoke(llm: ChatCohere) -> str:
    resp = llm.invoke([HumanMessage(content="Reply with exactly: pong")])
    text = _content_to_text(resp.content)
    assert text.strip(), "expected non-empty text response"
    usage = resp.usage_metadata or {}
    return f"text={_truncate(text, 60)!r} | usage={usage}"


def check_streaming(llm: ChatCohere) -> str:
    chunks = 0
    pieces: List[str] = []
    for chunk in llm.stream([HumanMessage(content="Count from 1 to 5.")]):
        chunks += 1
        pieces.append(_content_to_text(chunk.content))
    assert chunks > 1, f"expected multiple chunks, got {chunks}"
    return f"received {chunks} chunks | text={_truncate(''.join(pieces), 60)!r}"


def check_async_invoke(llm: ChatCohere) -> str:
    async def _run() -> AIMessage:
        return await llm.ainvoke([HumanMessage(content="Reply with exactly: async-ok")])

    resp = asyncio.run(_run())
    text = _content_to_text(resp.content)
    assert text.strip(), "expected non-empty async response"
    return f"text={_truncate(text, 60)!r}"


def check_multi_turn(llm: ChatCohere) -> str:
    messages = [
        HumanMessage(content="My favorite number is 7. Remember it."),
        AIMessage(content="Got it, your favorite number is 7."),
        HumanMessage(content="What is my favorite number times 6?"),
    ]
    resp = llm.invoke(messages)
    text = _content_to_text(resp.content)
    assert "42" in text, f"expected '42' in response, got: {_truncate(text)}"
    return f"text={_truncate(text, 80)!r}"


def check_reasoning(llm: ChatCohere) -> str:
    resp = llm.invoke(
        [
            HumanMessage(
                content="A store sells apples for $2 and oranges for $1.50. "
                "Sarah buys 3 apples and 4 oranges, then gets a 10% discount. "
                "How much does she pay? Show your reasoning."
            )
        ]
    )
    reasoning = _reasoning_from_content(resp.content)
    text = _content_to_text(resp.content)
    assert any(x in text for x in ["10.80", "10.8", "$10.80"]), (
        f"expected correct total in answer, got: {_truncate(text)}"
    )
    has_trace = bool(reasoning)
    return (
        f"reasoning_block={'yes' if has_trace else 'no'} | "
        f"reasoning={_truncate(reasoning or '', 80)!r} | "
        f"answer={_truncate(text, 80)!r}"
    )


def check_vision_url(llm: ChatCohere, image_url: str) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in one short sentence."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    resp = llm.invoke([message])
    text = _content_to_text(resp.content)
    assert text.strip(), "expected non-empty vision response"
    return f"image_url | text={_truncate(text, 100)!r}"


def check_vision_base64(llm: ChatCohere, image_url: str) -> str:
    data_uri = _image_as_data_uri(image_url)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What colors appear in this image?"},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
    )
    resp = llm.invoke([message])
    text = _content_to_text(resp.content)
    assert text.strip(), "expected non-empty base64 vision response"
    return f"base64 data URI ({len(data_uri)} chars) | text={_truncate(text, 90)!r}"


def check_vision_plus_reasoning(llm: ChatCohere, image_url: str) -> str:
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Look at this image and reason step by step about "
                "what kind of website it might belong to.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    resp = llm.invoke([message])
    text = _content_to_text(resp.content)
    reasoning = _reasoning_from_content(resp.content)
    assert text.strip(), "expected non-empty response for vision+reasoning"
    return (
        f"reasoning_block={'yes' if reasoning else 'no'} | "
        f"answer={_truncate(text, 90)!r}"
    )


def check_tool_calling(llm: ChatCohere) -> str:
    llm_with_tools = llm.bind_tools([get_weather])
    resp = llm_with_tools.invoke(
        [HumanMessage(content="What's the weather in Toronto?")]
    )
    assert resp.tool_calls, "expected at least one tool call"
    call = resp.tool_calls[0]
    assert call["name"] == "get_weather", f"unexpected tool: {call['name']}"
    assert "location" in call["args"], "expected 'location' arg"
    return f"tool={call['name']} args={call['args']}"


def check_tool_calling_plus_reasoning(llm: ChatCohere) -> str:
    llm_with_tools = llm.bind_tools([get_weather])
    resp = llm_with_tools.invoke(
        [
            HumanMessage(
                content="Get the weather for the capital of France. "
                "Think about which city that is first."
            )
        ]
    )
    assert resp.tool_calls, "expected at least one tool call"
    call = resp.tool_calls[0]
    reasoning = _reasoning_from_content(resp.content)
    loc = str(call["args"].get("location", "")).lower()
    assert "paris" in loc, f"expected Paris, got: {call['args']}"
    return (
        f"reasoning_block={'yes' if reasoning else 'no'} | "
        f"tool={call['name']} args={call['args']}"
    )


def check_structured_output(llm: ChatCohere) -> str:
    structured = llm.with_structured_output(PersonInfo)
    result = structured.invoke(
        [
            HumanMessage(
                content="Extract: Ada Lovelace was a 36-year-old mathematician."
            )
        ]
    )
    assert isinstance(result, PersonInfo), f"unexpected type: {type(result)}"
    assert result.name, "expected a name"
    return f"parsed -> {result.model_dump()}"


def check_json_mode(llm: ChatCohere) -> str:
    structured = llm.with_structured_output(PersonInfo, method="json_mode")
    result = structured.invoke(
        [
            HumanMessage(
                content="Return JSON with keys name, age, occupation for: "
                "Grace Hopper, 45, computer scientist."
            )
        ]
    )
    assert isinstance(result, PersonInfo), f"unexpected type: {type(result)}"
    return f"parsed -> {result.model_dump()}"


def check_documents_citations(llm: ChatCohere) -> str:
    docs = [
        {"id": "doc-1", "text": "Cohere was founded in 2019."},
        {"id": "doc-2", "text": "Cohere's headquarters is in Toronto, Canada."},
    ]
    resp = llm.invoke(
        [HumanMessage(content="Where is Cohere headquartered and when was it founded?")],
        documents=docs,
    )
    text = _content_to_text(resp.content)
    citations = resp.additional_kwargs.get("citations")
    assert "toronto" in text.lower(), f"expected grounded answer, got: {_truncate(text)}"
    n_citations = len(citations) if citations else 0
    return f"citations={n_citations} | answer={_truncate(text, 90)!r}"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _run_check(name: str, fn: Callable[[], str]) -> Tuple[str, bool, str, float]:
    start = time.time()
    try:
        detail = fn()
        return name, True, detail, time.time() - start
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=2).strip().splitlines()
        detail = f"{type(exc).__name__}: {exc}"
        if tb:
            detail += f"\n        {tb[-1]}"
        return name, False, detail, time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check Command A+ through langchain-cohere."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to test")
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE_URL, help="Image URL for vision checks"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip image/vision checks"
    )
    args = parser.parse_args()

    if not os.environ.get("COHERE_API_KEY"):
        print("ERROR: COHERE_API_KEY environment variable is not set.")
        return 2

    print("=" * 72)
    print("Command A+ sanity check via langchain-cohere")
    print("=" * 72)
    import cohere  # local import so the banner prints even if import fails

    try:
        from importlib.metadata import version as _pkg_version

        lc_cohere_version = _pkg_version("langchain-cohere")
    except Exception:  # noqa: BLE001
        lc_cohere_version = "n/a"

    print(f"model               : {args.model}")
    print(f"langchain_cohere    : {lc_cohere_version}")
    print(f"cohere SDK          : {cohere.__version__}")
    print(f"vision image        : {args.image}")
    print("-" * 72)

    llm = ChatCohere(model=args.model, temperature=0.3)

    checks: List[Tuple[str, Callable[[], str]]] = [
        ("token count", lambda: check_token_count(llm)),
        ("basic invoke (text)", lambda: check_basic_invoke(llm)),
        ("streaming", lambda: check_streaming(llm)),
        ("async invoke", lambda: check_async_invoke(llm)),
        ("multi-turn", lambda: check_multi_turn(llm)),
        ("reasoning trace", lambda: check_reasoning(llm)),
        ("tool calling", lambda: check_tool_calling(llm)),
        ("tool calling + reasoning", lambda: check_tool_calling_plus_reasoning(llm)),
        ("structured output (json_schema)", lambda: check_structured_output(llm)),
        ("json mode", lambda: check_json_mode(llm)),
        ("documents + citations", lambda: check_documents_citations(llm)),
    ]
    if not args.skip_vision:
        checks.extend(
            [
                ("vision (image URL)", lambda: check_vision_url(llm, args.image)),
                ("vision (base64)", lambda: check_vision_base64(llm, args.image)),
                (
                    "vision + reasoning",
                    lambda: check_vision_plus_reasoning(llm, args.image),
                ),
            ]
        )

    results = []
    for name, fn in checks:
        name, passed, detail, elapsed = _run_check(name, fn)
        results.append((name, passed))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}  ({elapsed:.2f}s)")
        print(f"        {detail}")

    print("-" * 72)
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    print("=" * 72)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
