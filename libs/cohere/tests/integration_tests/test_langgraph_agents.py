"""
Test ChatCohere chat model implementation.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

import sys
from typing import Sequence, Union

import pytest
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field

from langchain_cohere import ChatCohere

DEFAULT_MODEL = "command-r-plus"


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires >= python3.9")
@pytest.mark.vcr()
def test_langgraph_react_agent() -> None:
    from langgraph.prebuilt import create_react_agent  # type: ignore

    @tool
    def web_search(query: str) -> Union[int, str]:
        """Search the web to the answer to the question with a query search string.

        Args:
            query: The search query to surf the web with
        """
        if "obama" and "age" in query.lower():
            return 60
        if "president" in query:
            return "Barack Obama is the president of the USA"
        if "premier" in query:
            return "Chelsea won the premier league"
        return "The team called Fighter's Foxes won the champions league"

    @tool("python_interpeter_temp")
    def python_tool(code: str) -> str:
        """Executes python code and returns the result.
        The code runs in a static sandbox without interactive mode,
        so print output or save output to a file.

        Args:
            code: Python code to execute.
        """
        if "math.sqrt" in code:
            return "7.75"
        return "The code ran successfully"

    system_message = "You are a helpful assistant. Respond only in English."

    tools = [web_search, python_tool]
    model = ChatCohere(model=DEFAULT_MODEL)

    app = create_react_agent(model, tools, prompt=system_message)

    query = (
        "Find Barack Obama's age and use python tool to find the square root of his age"
    )

    messages = app.invoke({"messages": [("human", query)]})

    model_output = {
        "input": query,
        "output": messages["messages"][-1].content,
    }
    assert "7.7" in model_output.get("output", "").lower()

    message_history = messages["messages"]

    new_query = "who won the premier league"

    messages = app.invoke({"messages": message_history + [("human", new_query)]})
    final_answer = {
        "input": new_query,
        "output": messages["messages"][-1].content,
    }
    assert "chelsea" in final_answer.get("output", "").lower()


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires >= python3.9")
@pytest.mark.vcr()
def test_langchain_tool_calling_agent() -> None:
    def magic_function(input: int) -> int:
        """Applies a magic function to an input."""
        return input + 2

    magic_function_tool = Tool(
        name="magic_function",
        description="Applies a magic function to an input.",
        func=magic_function,
    )
    magic_function_tool.name = "magic_function"

    class MagicFunctionInput(BaseModel):
        input: int = Field(description="Number to apply the magic function to.")

    magic_function_tool.args_schema = MagicFunctionInput

    tools: Sequence[BaseTool] = [magic_function_tool]
    model = ChatCohere(model=DEFAULT_MODEL)

    query = "what is the value of magic_function(3)?"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            # Placeholders fill up a **list** of messages
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    ans = agent_executor.invoke({"input": query})
    assert "5" in ans.get("output", "").lower()
