"""Test Cohere API wrapper."""
from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock

import pytest
from cohere import AsyncClientV2, ClientV2
from pydantic import SecretStr

from langchain_cohere.llms import BaseCohere, Cohere


def test_cohere_api_key(
    patch_base_cohere_get_default_model: Generator[Optional[MagicMock], None, None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that cohere api key is a secret key."""
    # test initialization from init
    assert isinstance(BaseCohere(cohere_api_key="1").cohere_api_key, SecretStr)

    # test initialization from env variable
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    assert isinstance(BaseCohere().cohere_api_key, SecretStr)


def test_client_v2_is_initialised() -> None:
    llm = Cohere(cohere_api_key="test")
    client = llm.client
    async_client = llm.async_client

    assert isinstance(client, ClientV2)
    assert isinstance(async_client, AsyncClientV2)


@pytest.mark.parametrize(
    "cohere_kwargs,expected",
    [
        pytest.param(
            {"cohere_api_key": "test"}, {"model": "command-r-plus"}, id="defaults"
        ),
        pytest.param(
            {
                # the following are arbitrary testing values which shouldn't be used:
                "cohere_api_key": "test",
                "model": "foo",
                "temperature": 0.1,
                "max_tokens": 2,
                "k": 3,
                "p": 4,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.6,
                "truncate": "START",
            },
            {
                "model": "foo",
                "temperature": 0.1,
                "max_tokens": 2,
                "k": 3,
                "p": 4,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.6,
                "truncate": "START",
            },
            id="with values set",
        ),
    ],
)
def test_default_params(cohere_kwargs: Dict[str, Any], expected: Dict) -> None:
    cohere = Cohere(**cohere_kwargs)
    actual = cohere._default_params
    assert expected == actual


def test_tracing_params() -> None:
    # Test standard tracing params
    llm = Cohere(model="foo", cohere_api_key="api-key")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "cohere",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
    }

    llm = Cohere(model="foo", temperature=0.1, max_tokens=10, cohere_api_key="api-key")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "cohere",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
        "ls_max_tokens": 10,
    }


# def test_saving_loading_llm(tmp_path: Path) -> None:
#     """Test saving/loading an Cohere LLM."""
#     llm = BaseCohere(max_tokens=10)
#     llm.save(file_path=tmp_path / "cohere.yaml")
#     loaded_llm = load_llm(tmp_path / "cohere.yaml")
#     assert_llm_equality(llm, loaded_llm)
