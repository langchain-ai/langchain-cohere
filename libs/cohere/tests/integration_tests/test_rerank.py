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
