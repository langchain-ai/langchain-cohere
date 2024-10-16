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
@pytest.mark.xfail(reason="This test is flaky as the model outputs can vary. Outputs should be verified manually!")
def test_load_summarize_chain() -> None:
    llm = ChatCohere(model="command-r-plus", temperature=0)
    agent_executor = load_summarize_chain(llm, chain_type="stuff")
    with open("tests/integration_tests/chains/summarize/docs/ginger_benefits.txt") as f:
        doc_lines = f.readlines()
    docs = [Document("".join(doc_lines))]
    resp = agent_executor.invoke({"documents": docs})
    assert (
        resp.content
        == "Ginger is a fragrant spice that adds a spicy kick to sweet and savoury foods. It has a range of health benefits, including aiding digestion, relieving nausea, and easing bloating and gas. It contains antioxidants and anti-inflammatory compounds, and can be consumed in various forms, including tea, ale, candies, and Asian dishes. Ginger supplements are not recommended, as they may contain unlisted ingredients and are not well-regulated."  # noqa: E501
    )
