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
    llm = ChatCohere(model="command-r-plus", temperature=0)
    agent_executor = load_summarize_chain(llm, chain_type="stuff")
    with open("tests/integration_tests/chains/summarize/docs/ginger_benefits.txt") as f:
        doc_lines = f.readlines()
    docs = [Document("".join(doc_lines))]
    resp = agent_executor.invoke({"documents": docs})
    assert (
        resp.content
        == "Ginger has a range of health benefits, including improving digestion, relieving nausea, reducing bloating and gas, and providing antioxidants to manage free radicals. It may also have anti-inflammatory properties. Ginger can be consumed in various forms, including tea, ale, candies, and as an ingredient in many dishes. Fresh ginger root is flavourful, while ginger powder is a convenient and economical alternative. Ginger supplements are not recommended due to potential unknown ingredients and the unregulated nature of the supplement industry."  # noqa: E501
    )
