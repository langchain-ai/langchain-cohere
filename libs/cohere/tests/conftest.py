import os
from pathlib import Path
from typing import Dict, Generator
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from langchain_cohere.llms import BaseCohere

# Load environment variables from .env.test if it exists
# This allows developers to store API keys locally without setting env vars
env_test_path = Path(__file__).parent.parent / ".env.test"
if env_test_path.exists():
    load_dotenv(env_test_path)


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


@pytest.fixture(scope="session")
def vcr_cassette_dir() -> str:
    return os.path.join("tests", "integration_tests", "cassettes")
