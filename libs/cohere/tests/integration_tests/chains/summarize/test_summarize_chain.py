"""
Test test_sql_agent implementation.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

import pytest
from langchain_core.documents.base import Document

from langchain_cohere import ChatCohere, load_summarize_chain


@pytest.mark.vcr()
def test_load_summarize_chain() -> None:
    llm = ChatCohere(model="command-a-03-2025", temperature=0)
    agent_executor = load_summarize_chain(llm, chain_type="stuff")
    with open("tests/integration_tests/chains/summarize/docs/ginger_benefits.txt") as f:
        doc_lines = f.readlines()
    docs = [Document("".join(doc_lines))]
    resp = agent_executor.invoke({"documents": docs})
    content = resp.content.lower()
    assert "ginger" in content
    assert any(word in content for word in ["digestion", "nausea", "health"])
