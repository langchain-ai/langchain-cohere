"""Standard LangChain interface tests"""

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_cohere import ChatCohere


class Person(BaseModel):
    """Record attributes of a person."""

    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


def test_chat_model_can_instantiate():
    model = ChatCohere(
        model="command-r-plus",
        temperature=0,
        cohere_api_key="test_key",
    )
    assert isinstance(model, BaseChatModel)

def test_with_structured_output():
    class Person(BaseModel):
        name: str
        age: int

    model = ChatCohere(
        model="command-r-plus",
        temperature=0,
        cohere_api_key="test_key",
    )

    out = model.with_structured_output(Person)
    assert out is not None
