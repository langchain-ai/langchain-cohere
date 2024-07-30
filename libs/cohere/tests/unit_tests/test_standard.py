"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_cohere import ChatCohere


class TestCohereStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCohere

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "command-r-plus",
            "temperature": 0,
            "cohere_api_key": "test_key",
        }
