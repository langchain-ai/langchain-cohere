"""
Test ChatCohere chat model implementation.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_cohere import ChatCohere

DEFAULT_MODEL = "command-r"


@pytest.mark.vcr()
def test_stream() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere(model=DEFAULT_MODEL)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.vcr()
async def test_astream() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere(model=DEFAULT_MODEL)

    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
    if chunks_with_token_counts != 1:
        raise AssertionError(
            "Expected exactly one chunk with token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )


async def test_abatch() -> None:
    """Test streaming tokens from ChatCohere"""
    llm = ChatCohere(model=DEFAULT_MODEL)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatCohere."""
    llm = ChatCohere(model=DEFAULT_MODEL)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatCohere."""
    llm = ChatCohere()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatCohere."""
    llm = ChatCohere(model=DEFAULT_MODEL)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    usage_metadata = result.usage_metadata
    assert usage_metadata is not None
    assert usage_metadata["input_tokens"] > 0
    assert usage_metadata["output_tokens"] > 0
    assert (
        usage_metadata["total_tokens"]
        == usage_metadata["input_tokens"] + usage_metadata["output_tokens"]
    )


@pytest.mark.vcr()
def test_invoke() -> None:
    """Test invoke tokens from ChatCohere."""
    llm = ChatCohere(model=DEFAULT_MODEL)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


@pytest.mark.vcr()
def test_invoke_tool_calls() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it calls the tool
    result = tool_llm.invoke("Erick, 27 years old")

    assert isinstance(result, AIMessage)
    additional_kwargs = result.additional_kwargs
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Person"
    assert tool_call["args"] == {"name": "Erick", "age": 27}


@pytest.mark.vcr()
def test_streaming_tool_call() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it calls the tool
    strm = tool_llm.stream("Erick, 27 years old")

    additional_kwargs = None
    tool_call_chunks_present = False
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        additional_kwargs = chunk.additional_kwargs
        if chunk.tool_call_chunks:
            tool_call_chunks_present = True

    assert additional_kwargs is not None
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }
    assert isinstance(chunk, AIMessageChunk)
    assert isinstance(chunk.tool_call_chunks, list)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "Person"
    assert tool_call_chunk["args"] is not None
    assert json.loads(tool_call_chunk["args"]) == {"name": "Erick", "age": 27}
    assert tool_call_chunks_present


@pytest.mark.vcr()
async def test_async_streaming_tool_call() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it calls the tool
    strm = tool_llm.astream("Erick, 27 years old")

    additional_kwargs = None
    tool_call_chunks_present = False
    tool_plan = None
    async for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        additional_kwargs = chunk.additional_kwargs
        if chunk.tool_call_chunks:
            tool_call_chunks_present = True
            tool_plan = chunk.content

    assert additional_kwargs is not None
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }
    assert isinstance(chunk, AIMessageChunk)
    assert isinstance(chunk.tool_call_chunks, list)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "Person"
    assert tool_call_chunk["args"] is not None
    assert json.loads(tool_call_chunk["args"]) == {"name": "Erick", "age": 27}
    assert tool_call_chunks_present
    assert (
        tool_plan
        == "I will use the Person tool to create a person with the name Erick and age 27, and then relay this information to the user."  # noqa: E501
    )


@pytest.mark.vcr()
def test_invoke_multiple_tools() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    @tool
    def add_two_numbers(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b

    @tool
    def capital_cities(country: str) -> str:
        """Returns the capital city of a country"""
        return "France"

    tool_llm = llm.bind_tools([add_two_numbers, capital_cities])

    result = tool_llm.invoke("What is the capital of France")
    print(result)

    assert isinstance(result, AIMessage)
    additional_kwargs = result.additional_kwargs
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "capital_cities"
    parameters = json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"])
    assert {"country": "France"} == parameters


@pytest.mark.xfail(
    reason="Cohere models return empty output when a tool is passed in but not called."
)
def test_streaming_tool_call_no_tool_calls() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)

    class Person(BaseModel):
        name: str = Field(type=str, description="The name of the person")
        age: int = Field(type=int, description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it doesn't call the tool
    strm = tool_llm.stream("What is 2+2?")
    acc: Any = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        acc = chunk if acc is None else acc + chunk
    assert acc.content != ""
    assert "tool_calls" not in acc.additional_kwargs


@pytest.mark.vcr()
def test_get_num_tokens_with_specified_model() -> None:
    llm = ChatCohere(temperature=0, model="command-r")
    expected = 3  # This may change if the replay also changes.

    actual = llm.get_num_tokens("hello, world")

    assert expected == actual


@pytest.mark.vcr()
def test_get_num_tokens_with_default_model() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL, temperature=0)
    expected = 3  # This may change if the replay also changes.

    actual = llm.get_num_tokens("hello, world")

    assert expected == actual


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [
                HumanMessage(content="what is the value of magic_function(3)?"),
                AIMessage(
                    content="I will call magic_function tool with input 3",
                    tool_calls=[
                        {
                            "name": "magic_function",
                            "args": {"input": 3},
                            "id": "d86e6098-21e1-44c7-8431-40cfc6d35590",
                        }
                    ],
                ),
                ToolMessage(
                    name="magic_function",
                    content="5",
                    tool_call_id="d86e6098-21e1-44c7-8431-40cfc6d35590",
                ),
            ],
            id="Single tool call ",
        ),
    ],
)
def test_tool_call_with_tool_results(messages: List[BaseMessage]) -> None:
    llm = ChatCohere(temperature=0)
    response = llm.invoke(messages)
    assert isinstance(response, AIMessage)


class Person(BaseModel):
    """Name and age of the person"""

    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "schema,method,expected",
    [
        pytest.param(Person, "json_mode", Person(name="John", age=30)),
        pytest.param(Person, "json_schema", Person(name="John", age=30)),
        pytest.param(Person, "function_calling", Person(name="John", age=30)),
        pytest.param(
            {
                "description": "Name and age of the person",
                "properties": {
                    "name": {
                        "description": "Name of the person",
                        "title": "Name",
                        "type": "string",
                    },
                    "age": {
                        "description": "Age of the person",
                        "title": "Age",
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
                "title": "Person",
                "type": "object",
            },
            "json_mode",
            {"name": "John", "age": 30},
        ),
        pytest.param(
            {
                "description": "Name and age of the person",
                "properties": {
                    "name": {
                        "description": "Name of the person",
                        "title": "Name",
                        "type": "string",
                    },
                    "age": {
                        "description": "Age of the person",
                        "title": "Age",
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
                "title": "Person",
                "type": "object",
            },
            "json_schema",
            {"name": "John", "age": 30},
        ),
    ],
)
def test_cohere_with_structured_output(
    schema: Union[Dict, Type[BaseModel]],
    method: Literal["function_calling", "json_mode", "json_schema"],
    expected: Union[Dict, Type[BaseModel]],
) -> None:
    cohere_client = ChatCohere(model=DEFAULT_MODEL)
    llm_with_structured_output = cohere_client.with_structured_output(schema, method)
    result = llm_with_structured_output.invoke(
        "The person's name is John, and he is 30 years old"
    )
    assert result == expected
