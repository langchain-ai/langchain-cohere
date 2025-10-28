"""Standard LangChain interface tests"""

from typing import Any, Type

import pytest
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
            "model": "command-r-plus",
            "temperature": 0,
            "cohere_api_key": "test_key",
        }

    @pytest.mark.xfail(reason="Standard test not moved to pydantic V2...")
    @pytest.mark.parametrize("schema", [Person, Person.model_json_schema()])
    def test_with_structured_output(
        self,
        model: BaseChatModel,
        schema: Any,
    ) -> None:
        if not self.has_structured_output:
            return

        assert model.with_structured_output(schema) is not None
