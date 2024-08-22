"""
Test ChatCohere chat model.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
)

from langchain_cohere import ChatCohere, CohereRagRetriever

DEFAULT_MODEL = "command-r"


@pytest.mark.vcr()
def test_connectors() -> None:
    """Test connectors parameter support from ChatCohere."""
    llm = ChatCohere(model=DEFAULT_MODEL).bind(connectors=[{"id": "web-search"}])

    result = llm.invoke("Who directed dune two? reply with just the name.")
    assert isinstance(result.content, str)


@pytest.mark.vcr()
def test_documents() -> None:
    docs = [{"text": "The sky is green."}]
    llm = ChatCohere(model=DEFAULT_MODEL).bind(documents=docs)
    prompt = "What color is the sky?"

    result = llm.invoke(prompt)
    assert isinstance(result.content, str)
    assert len(result.response_metadata["documents"]) == 1


@pytest.mark.vcr()
def test_documents_chain() -> None:
    llm = ChatCohere(model=DEFAULT_MODEL)

    def get_documents(_: Any) -> List[Document]:
        return [Document(page_content="The sky is green.")]

    def format_input_msgs(input: Dict[str, Any]) -> List[HumanMessage]:
        return [
            HumanMessage(
                content=input["message"],
                additional_kwargs={
                    "documents": input.get("documents", None),
                },
            )
        ]

    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("input_msgs")])
    chain: RunnableSerializable[Any, Any] = (
        {"message": RunnablePassthrough(), "documents": get_documents}
        | RunnablePassthrough()
        | {"input_msgs": format_input_msgs}
        | prompt
        | llm
    )

    result = chain.invoke("What color is the sky?")
    assert isinstance(result.content, str)
    assert len(result.response_metadata["documents"]) == 1


@pytest.mark.vcr()
def test_who_are_cohere() -> None:
    user_query = "Who founded Cohere?"
    llm = ChatCohere()
    retriever = CohereRagRetriever(llm=llm, connectors=[{"id": "web-search"}])

    actual = retriever.get_relevant_documents(user_query)

    answer = actual.pop()
    citations = answer.metadata.get("citations")

    relevant_documents = actual
    assert len(relevant_documents) > 0
    expected_answer = "cohere has 3 founders; aidan gomez, ivan zhang, and nick frosst. aidan gomez is also the current ceo. all three founders attended the university of toronto."  # noqa: E501
    assert expected_answer == answer.page_content.lower()
    assert citations is not None
    assert len(citations) > 0


@pytest.mark.vcr()
def test_who_founded_cohere_with_custom_documents() -> None:
    rag = CohereRagRetriever(llm=ChatCohere(model=DEFAULT_MODEL))

    docs = rag.invoke(
        "who is the founder of cohere?",
        documents=[
            Document(page_content="Langchain supports cohere RAG!"),
            Document(page_content="The sky is a mixture of brown and purple!"),
            Document(page_content="Barack Obama is the founder of Cohere!"),
        ],
    )
    answer = docs.pop()
    citations = answer.metadata.get("citations")
    relevant_documents = docs
    assert len(relevant_documents) > 0
    expected_answer = "barack obama is the founder of cohere."
    assert expected_answer == answer.page_content.lower()
    assert citations is not None
    assert len(citations) > 0
