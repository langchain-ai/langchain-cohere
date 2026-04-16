from langchain_cohere.chains.summarize.summarize_chain import load_summarize_chain
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.common import CohereCitation
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_cohere.rag_retrievers import CohereRagRetriever
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.rerank import CohereRerank

_LAZY_IMPORTS = {
    "create_sql_agent": "langchain_cohere.sql_agent.agent",
}

__all__ = [
    "CohereCitation",
    "ChatCohere",
    "CohereEmbeddings",
    "CohereRagRetriever",
    "CohereRerank",
    "create_cohere_react_agent",
    "create_sql_agent",
    "load_summarize_chain",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'langchain_cohere' has no attribute {name!r}")
