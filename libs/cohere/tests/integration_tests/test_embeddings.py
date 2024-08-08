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
def test_langchain_cohere_embedding_query() -> None:
    document = "foo bar"
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    output = embedding.embed_query(document)
    assert len(output) > 0


@pytest.mark.vcr()
def test_langchain_cohere_embedding_float() -> None:
    document = "foo bar"
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    # Call the embed_with_retry directly, rather than through `embed_query`
    # or `embed_documents` so that we can verify the correct fields are
    # present in the response.
    output = embedding.embed_with_retry(
        model=embedding.model,
        texts=[document],
        input_type="search_query",
        truncate=embedding.truncate,
        embedding_types=embedding.embedding_types,
    )
    assert output
    assert output.embeddings
    assert output.embeddings.float
    assert len(output.embeddings.float) > 0
