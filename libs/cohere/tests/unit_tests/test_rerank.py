"""Test chat model integration."""

import cohere
import pytest
from langchain_core.documents import Document

from langchain_cohere import CohereRerank


def test_initialization() -> None:
    """Test chat model initialization."""
    CohereRerank(cohere_api_key="test", model="rerank-v3.5")


def test_doc_to_string_with_string() -> None:
    """Test document serialization"""
    reranker = CohereRerank(cohere_api_key="test", model="rerank-v3.5")
    out = reranker._document_to_str("test str")
    assert out == "test str"


def test_doc_to_string_with_document() -> None:
    """Test document serialization"""
    reranker = CohereRerank(cohere_api_key="test", model="rerank-v3.5")
    out = reranker._document_to_str(Document(page_content="test str"))
    assert out == "test str"


def test_doc_to_string_with_dict() -> None:
    """Test document serialization"""
    reranker = CohereRerank(cohere_api_key="test", model="rerank-v3.5")
    out = reranker._document_to_str({"title": "hello", "text": "test str"})
    assert out == "title: hello\ntext: test str\n"


def test_doc_to_string_with_dicts_with_rank_fields() -> None:
    """Test document serialization"""
    reranker = CohereRerank(cohere_api_key="test", model="rerank-v3.5")
    out = reranker._document_to_str(
        document={"title": "hello", "text": "test str"}, rank_fields=["text"]
    )
    assert out == "text: test str\n"


@pytest.mark.parametrize(
    "client, valid",
    [
        pytest.param(
            cohere.Client(api_key="test"),
            False,
            id="ClientV1 should not be valid",
        ),
        pytest.param(
            cohere.ClientV2(api_key="test"),
            True,
            id="ClientV2 should be valid",
        ),
        pytest.param(
            None,
            True,
            id="client=None should be valid",
        ),
    ],
)
def test_error_with_client_v1(client: cohere.Client, valid: bool) -> None:
    """Test error with client v1"""
    if valid:
        CohereRerank(cohere_api_key="test", model="rerank-v3.5", client=client)
    else:
        with pytest.raises(
            ValueError,
            match=(
                "The 'client' parameter must be an instance of cohere.ClientV2.\n"
                "You may create the ClientV2 object like:\n\n"
                "import cohere\nclient = cohere.ClientV2(...)"
            ),
        ):
            CohereRerank(cohere_api_key="test", model="rerank-v3.5", client=client)
