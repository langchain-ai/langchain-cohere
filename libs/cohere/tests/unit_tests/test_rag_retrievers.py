"""Test rag retriever integration."""

from typing import Generator, Optional

from langchain_cohere.chat_models import BaseCohere, ChatCohere
from langchain_cohere.rag_retrievers import CohereRagRetriever


def test_initialization(
    patch_base_cohere_get_default_model: Generator[Optional[BaseCohere], None, None],
) -> None:
    """Test chat model initialization."""
    CohereRagRetriever(llm=ChatCohere(cohere_api_key="test"))
