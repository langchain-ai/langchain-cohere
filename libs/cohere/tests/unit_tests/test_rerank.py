"""Test chat model integration."""

from langchain_core.documents import Document

from langchain_cohere import CohereRerank

import pytest
import cohere

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

def test_error_with_client_v1() -> None:
    """Test error with client v1"""
    with pytest.raises(ValueError, match="The 'client' parameter must be an instance of cohere.ClientV2."):
        client_v1 = cohere.Client()
        reranker = CohereRerank(cohere_api_key="test", model="rerank-v3.5", client=client_v1)