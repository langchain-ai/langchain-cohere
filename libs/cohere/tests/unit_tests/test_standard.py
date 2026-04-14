"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, Field

from langchain_cohere import ChatCohere


class Person(BaseModel):
    """Record attributes of a person."""

    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


class TestCohereStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCohere

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "command-a-03-2025",
            "temperature": 0,
            "cohere_api_key": "test_key",
        }
