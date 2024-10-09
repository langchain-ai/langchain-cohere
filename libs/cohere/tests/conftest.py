from typing import Dict, Generator, Optional
from unittest.mock import MagicMock, patch

import pytest

from langchain_cohere.llms import BaseCohere


@pytest.fixture(scope="module")
def vcr_config() -> Dict:
    return {
        # IMPORTANT: Filter out the Authorization header from stored replay test data.
        "filter_headers": [("Authorization", None)],
        "ignore_hosts": ["storage.googleapis.com"],
        "record_mode": "all",
    }


@pytest.fixture(scope="module")
def patch_base_cohere_get_default_model() -> Generator[Optional[MagicMock], None, None]:
    # IMPORTANT: Since this fixture is module scoped, it only needs to be called once,
    # in the top-level test function. It will ensure that the get_default_model method
    # is mocked for all tests in that module.
    with patch.object(
        BaseCohere, "_get_default_model", return_value="command-r-plus"
    ) as mock_get_default_model:
        yield mock_get_default_model
