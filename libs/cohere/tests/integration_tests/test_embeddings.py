"""
Test CohereEmbeddings

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""
import pytest

from langchain_cohere import CohereEmbeddings


@pytest.mark.vcr()
def test_langchain_cohere_embedding_documents() -> None:
    documents = ["foo bar"]
    embedding = CohereEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


@pytest.mark.vcr()
def test_langchain_cohere_embedding_query() -> None:
    document = "foo bar"
    embedding = CohereEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
