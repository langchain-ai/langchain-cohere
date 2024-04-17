"""
Test CohereRerank.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

import pytest
from langchain_core.documents import Document

from langchain_cohere import CohereRerank


@pytest.mark.vcr()
def test_langchain_cohere_rerank_documents() -> None:
    rerank = CohereRerank()
    test_documents = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document."),
    ]
    test_query = "Test query"
    results = rerank.rerank(test_documents, test_query)
    assert len(results) == 2


@pytest.mark.vcr()
def test_langchain_cohere_rerank_with_rank_fields() -> None:
    rerank = CohereRerank()
    test_documents = [
        {"content": "This document is about Penguins.", "subject": "Physics"},
        {"content": "This document is about Physics.", "subject": "Penguins"},
    ]
    test_query = "penguins"

    response = rerank.rerank(test_documents, test_query, rank_fields=["content"])

    assert len(response) == 2
    assert response[0]["index"] == 0
    results = {r["index"]: r["relevance_score"] for r in response}
    assert results[0] > results[1]
