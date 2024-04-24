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


@pytest.mark.vcr()
def test_connectors() -> None:
    """Test connectors parameter support from ChatCohere."""
    llm = ChatCohere().bind(connectors=[{"id": "web-search"}])

    result = llm.invoke("Who directed dune two? reply with just the name.")
    assert isinstance(result.content, str)


@pytest.mark.vcr()
def test_documents() -> None:
    docs = [{"text": "The sky is green."}]
    llm = ChatCohere().bind(documents=docs)
    prompt = "What color is the sky?"

    result = llm.invoke(prompt)
    assert isinstance(result.content, str)
    assert len(result.response_metadata["documents"]) == 1


@pytest.mark.vcr()
def test_documents_chain() -> None:
    llm = ChatCohere()

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
    user_query = "Who are Cohere?"
    llm = ChatCohere()
    retriever = CohereRagRetriever(llm=llm, connectors=[{"id": "web-search"}])

    actual = retriever.get_relevant_documents(user_query)

    answer = actual.pop()
    citations = answer.metadata.get("citations")

    relevant_documents = actual
    assert len(relevant_documents) > 0

    expected_text = """Cohere is a Canadian multinational technology company that provides access to advanced Large Language Models and NLP tools through an easy-to-use API. It was founded in 2019 by Aidan Gomez, Ivan Zhang, and Nick Frosst, and is headquartered in Toronto and San Francisco, with offices in Palo Alto and London. 

Cohere's mission is to build machines that understand the world and make them safely accessible to all. They aim to transform enterprises and their products with AI that unlocks a more intuitive way to generate, search, and summarize information. 

The company is focused on generative AI for businesses, helping them deploy chatbots, search engines, copywriting, summarization, and other AI-driven products. Cohere's platform is cloud-agnostic and can be deployed on virtual private cloud (VPC) or on-site, offering flexibility and control to its users. 

They have raised significant funding from prominent investors and have partnerships with major companies such as Oracle, Salesforce, and McKinsey."""  # noqa: E501
    assert expected_text == answer.page_content
    assert citations is not None
    assert len(citations) > 0
