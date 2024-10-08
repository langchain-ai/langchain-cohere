"""Test rag retriever integration."""


from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.rag_retrievers import CohereRagRetriever
from typing import Generator, Optional
from unittest.mock import MagicMock

def test_initialization(patch_base_cohere_get_default_model: 
                        Generator[Optional[MagicMock], None, None]) -> None:
    """Test chat model initialization."""
    CohereRagRetriever(llm=ChatCohere(cohere_api_key="test"))
