"""Load summarizing chains."""
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core._api import beta
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.prompts.chat import (
    BaseMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda, RunnableSerializable

from langchain_cohere.chains.summarize.prompt import RAG_SUMMARIZATION_PREAMBLE
from langchain_cohere.chat_models import ChatCohere


def create_summarize_prompt(
    prompt_message: BaseMessage = HumanMessage(
        content="Please summarize the documents in a concise manner."
    ),
    extra_prompt_messages: List[BaseMessagePromptTemplate] = [],
) -> ChatPromptTemplate:
    """Create prompt for this agent.
    Args:
        system_message: Message to use as the system message that will be the
            first in the prompt.
        extra_prompt_messages: Prompt messages that will be placed between the
            system message and the new human input.
    Returns:
        A prompt template to pass into this agent.
    """
    extra_prompt_messages = extra_prompt_messages or []
    messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]
    if prompt_message:
        messages = [prompt_message]
    else:
        messages = [prompt_message] + extra_prompt_messages
    return ChatPromptTemplate(messages=messages)


def _load_stuff_chain(
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
) -> RunnableSerializable:
    if "preamble" in llm.__dict__ and not llm.__dict__.get("preamble"):
        llm = ChatCohere(**llm.__dict__)
        llm.preamble = RAG_SUMMARIZATION_PREAMBLE

    if not prompt:
        prompt = create_summarize_prompt()

    def llm_with_docs(input_: dict) -> RunnableSerializable[Any, Any]:
        docs = input_["documents"]
        return RunnableLambda(lambda x: x["input"]) | llm.bind(documents=docs)

    runnable = (
        RunnablePassthrough.assign(
            documents=lambda x: x["documents"],
            input=lambda x: prompt.format_prompt(**x),  # type: ignore[union-attr]
        )
        | llm_with_docs
    )
    return runnable


@beta(
    message="""Makes use of Cohere's grounded RAG summarization, 
        which may change in a later langchain-cohere version"""
)
def load_summarize_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    **kwargs: Any,
) -> RunnableSerializable:
    """Load summarizing chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Currently, only "stuff"
            is supported in this implementation.
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.

    Returns:
        A chain to use for summarizing.
    """

    loader_mapping: Dict[
        str,
        Callable[
            [BaseLanguageModel[Any], BasePromptTemplate[Any]],
            RunnableSerializable[Any, Any],
        ],
    ] = {
        "stuff": _load_stuff_chain,
    }
    if chain_type not in loader_mapping:
        raise ValueError(
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )
    return loader_mapping[chain_type](llm, **kwargs)  # type: ignore[call-arg]
