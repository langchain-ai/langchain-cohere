from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langchain_cohere import ChatCohere


@pytest.mark.parametrize(
    "messages,expected_content",
    [
        pytest.param(
            [
                HumanMessage(content="what is the value of magic_function(3)?"),
                AIMessage(
                    content="",
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
            "The value of magic_function(3) is **5**.",
            marks=pytest.mark.vcr,
            id="magic function",
        )
    ],
)
def test_tool_calling_agent(messages: List[BaseMessage], expected_content: str) -> None:
    llm = ChatCohere()

    actual = llm.invoke(messages)
    actual_citations = actual.additional_kwargs.get("citations")

    assert expected_content == actual.content

    assert actual_citations is not None
    assert len(actual_citations) > 0
