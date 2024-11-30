from typing import Dict, Generator
from unittest.mock import patch

import pytest

from langchain_cohere.llms import BaseCohere


@pytest.fixture(scope="module")
def vcr_config() -> Dict:
    return {
        # IMPORTANT: Filter out the Authorization header from stored replay test data.
        "filter_headers": [("Authorization", None)],
        "ignore_hosts": ["storage.googleapis.com"],
    }


@pytest.fixture
def patch_base_cohere_get_default_model() -> Generator[BaseCohere, None, None]:
    with patch.object(
        BaseCohere, "_get_default_model", return_value="command-r-plus", autospec=True
    ) as mock_get_default_model:
        yield mock_get_default_model
