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


def get_num_documents_from_v2_response(response: Dict[str, Any]) -> int:
    document_ids = set()
    for c in response["citations"]:
        document_ids.update({doc.id for doc in c.sources if doc.type == "document"})
    return len(document_ids)


@pytest.mark.vcr()
@pytest.mark.xfail(
    reason="Chat V2 no longer relies on connectors, \
                   so this test is no longer valid."
)
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
    assert get_num_documents_from_v2_response(result.response_metadata) == 1


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
    assert get_num_documents_from_v2_response(result.response_metadata) == 1


@pytest.mark.vcr()
@pytest.mark.xfail(
    reason="Chat V2 no longer relies on connectors, \
                   so this test is no longer valid."
)
def test_who_are_cohere() -> None:
    user_query = "Who founded Cohere?"
    llm = ChatCohere(model=DEFAULT_MODEL)
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
    assert len(relevant_documents) == 0
    expected_answer = "according to my sources, cohere was founded by barack obama."
    assert expected_answer == answer.page_content.lower()
    assert citations is not None
    assert len(citations) > 0
