"""
Test create_csv_agernt implementation.

Uses the replay testing functionality, so you don't need an API key to run these tests.
https://python.langchain.com/docs/contributing/testing#recording-http-interactions-with-pytest-vcr

When re-recording these tests you will need to set COHERE_API_KEY.
"""

import pytest

from langchain_cohere import ChatCohere
from langchain_cohere.csv_agent.agent import create_csv_agent


@pytest.mark.vcr()
def test_single_csv() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere()
    csv_agent = create_csv_agent(
        llm, path="tests/integration_tests/csv_agent/csv/movie_ratings.csv"
    )
    resp = csv_agent.invoke({"input": "Which movie has the highest average rating?"})
    assert "output" in resp
    assert (
        "The Shawshank Redemption has the highest average rating of 7.25."  # noqa: E501
        == resp["output"]
    )


@pytest.mark.vcr()
def test_multiple_csv() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere()
    csv_agent = create_csv_agent(
        llm,
        path=[
            "tests/integration_tests/csv_agent/csv/movie_ratings.csv",
            "tests/integration_tests/csv_agent/csv/movie_bookings.csv",
        ],
    )
    resp = csv_agent.invoke({"input": "Which movie has the highest average rating?"})
    assert "output" in resp
    assert (
        "The movie with the highest average rating is *The Shawshank Redemption* with an average rating of 7.25."  # noqa: E501
        == resp["output"]
    )
    resp = csv_agent.invoke(
        {"input": "who bought the most number of tickets to finding nemo?"}
    )
    assert "output" in resp
    assert (
        "Penelope bought the most tickets to Finding Nemo."  # noqa: E501
        == resp["output"]
    )
