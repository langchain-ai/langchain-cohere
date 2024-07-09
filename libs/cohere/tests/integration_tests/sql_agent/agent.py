"""
Test test_sql_agent implementation.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

import pytest
from langchain_community.utilities import SQLDatabase

from langchain_cohere import ChatCohere
from langchain_cohere.sql_agent.agent import create_sql_agent


@pytest.mark.vcr()
def test_sql_agent() -> None:
    db = SQLDatabase.from_uri(
        "sqlite:///tests/integration_tests/sql_agent/db/employees.db"
    )
    llm = ChatCohere(model="command-r-plus", temperature=0)
    agent_executor = create_sql_agent(
        llm, db=db, agent_type="tool-calling", verbose=True
    )
    resp = agent_executor.invoke({"input": "which employee has the highest salary?"})
    assert "output" in resp.keys()
    assert "jane doe" in resp.get("output", "").lower()
