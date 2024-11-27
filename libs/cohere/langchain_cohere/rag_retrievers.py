from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _get_docs(response: Any) -> List[Document]:
    docs = []
    if (
        "documents" in response.generation_info
        and response.generation_info["documents"]
    ):
        for doc in response.generation_info["documents"]:
            content = doc.get("snippet", None) or doc.get("text", None)
            if content is not None:
                docs.append(Document(page_content=content, metadata=doc))

    docs.append(
        Document(
            page_content=response.message.content,
            metadata={
                "type": "model_response",
                "citations": response.generation_info["citations"],
                "token_count": response.generation_info["token_count"],
            },
        )
    )
    return docs


class CohereRagRetriever(BaseRetriever):
    """Cohere Chat API with RAG."""

    llm: BaseChatModel
    """Cohere ChatModel to use."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        documents: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        messages: List[List[BaseMessage]] = [[HumanMessage(content=query)]]
        res = self.llm.generate(
            messages,
            documents=documents,
            callbacks=run_manager.get_child(),
            **kwargs,
        ).generations[0][0]
        return _get_docs(res)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        documents: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        messages: List[List[BaseMessage]] = [[HumanMessage(content=query)]]
        res = (
            await self.llm.agenerate(
                messages,
                documents=documents,
                callbacks=run_manager.get_child(),
                **kwargs,
            )
        ).generations[0][0]
        return _get_docs(res)
