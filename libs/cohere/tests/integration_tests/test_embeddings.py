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
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


@pytest.mark.vcr()
def test_langchain_cohere_embedding_multiple_documents() -> None:
    documents = ["foo bar", "bar foo"]
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0
    assert len(output[1]) > 0


@pytest.mark.vcr()
def test_langchain_cohere_embedding_query() -> None:
    document = "foo bar"
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    output = embedding.embed_query(document)
    assert len(output) > 0


@pytest.mark.vcr()
def test_langchain_cohere_embedding_query_int8_embedding_type() -> None:
    document = "foo bar"
    embedding = CohereEmbeddings(
        model="embed-english-light-v3.0", embedding_types=["int8"]
    )
    output = embedding.embed_query(document)
    assert len(output) > 0


@pytest.mark.vcr()
def test_langchain_cohere_embedding_documents_int8_embedding_type() -> None:
    documents = ["foo bar"]
    embedding = CohereEmbeddings(
        model="embed-english-light-v3.0", embedding_types=["int8"]
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0
