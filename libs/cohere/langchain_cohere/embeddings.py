import typing
from typing import Any, Dict, List, Optional, Sequence, Union

import cohere
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Extra, root_validator, model_validator
from langchain_core.utils import get_from_dict_or_env

from .utils import _create_retry_decorator
from pydantic import ConfigDict
from typing_extensions import Self




class CohereEmbeddings(BaseModel, Embeddings):
    """
    Implements the Embeddings interface with Cohere's text representation language
    models.

    Find out more about us at https://cohere.com and https://huggingface.co/CohereForAI

    This implementation uses the Embed API - see https://docs.cohere.com/reference/embed

    To use this you'll need to a Cohere API key - either pass it to cohere_api_key
    parameter or set the COHERE_API_KEY environment variable.

    API keys are available on https://cohere.com - it's free to sign up and trial API
    keys work with this implementation.

    Basic Example:
        .. code-block:: python

            cohere_embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
            text = "This is a test document."

            query_result = cohere_embeddings.embed_query(text)
            print(query_result)

            doc_result = cohere_embeddings.embed_documents([text])
            print(doc_result)
    """

    client: Any  #: :meta private:
    """Cohere client."""
    async_client: Any  #: :meta private:
    """Cohere async client."""
    model: Optional[str] = None
    """Model name to use. It is mandatory to specify the model name."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    cohere_api_key: Optional[str] = None

    embedding_types: Sequence[str] = ["float"]
    "Specifies the types of embeddings you want to get back"

    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[float] = None
    """Timeout in seconds for the Cohere API request."""
    user_agent: str = "langchain:partner"
    """Identifier for the application making the request."""

    base_url: Optional[str] = None
    """Override the default Cohere API URL."""

    model_config = ConfigDict(arbitrary_types_allowed=True,extra="forbid",)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        request_timeout = (self.request_timeout or None)

        client_name = self.user_agent
        self.client = cohere.Client(
            cohere_api_key,
            timeout=request_timeout,
            client_name=client_name,
            base_url=self.base_url,
        )
        self.async_client = cohere.AsyncClient(
            cohere_api_key,
            timeout=request_timeout,
            client_name=client_name,
            base_url=self.base_url,
        )

        return self

    @model_validator(mode="after")
    def validate_model_specified(self) -> Self:
        """Validate that model is specified."""
        model = (self.model or None)
        if not model:
            raise ValueError(
                "Did not find `model`! Please "
                " pass `model` as a named parameter."
                " Please check out"
                " https://docs.cohere.com/reference/embed"
                " for available models."
            )

        return self

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the embed call."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        def _embed_with_retry(**kwargs: Any) -> Any:
            return self.client.embed(**kwargs)

        return _embed_with_retry(**kwargs)

    def aembed_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the embed call."""
        retry_decorator = _create_retry_decorator(self.max_retries)

        @retry_decorator
        async def _embed_with_retry(**kwargs: Any) -> Any:
            return await self.async_client.embed(**kwargs)

        return _embed_with_retry(**kwargs)

    def embed(
        self,
        texts: List[str],
        *,
        input_type: typing.Optional[cohere.EmbedInputType] = None,
    ) -> List[List[float]]:
        response = self.embed_with_retry(
            model=self.model,
            texts=texts,
            input_type=input_type,
            truncate=self.truncate,
            embedding_types=self.embedding_types,
        )
        embeddings = response.dict().get("embeddings", [])
        embeddings_as_float: List[List[float]] = []
        for embedding_type in self.embedding_types:
            e: List[List[Union[int, float]]] = embeddings.get(embedding_type)
            if not e:
                continue
            for i in range(len(e)):
                embeddings_as_float.append(list(map(float, e[i])))
        return embeddings_as_float

    async def aembed(
        self,
        texts: List[str],
        *,
        input_type: typing.Optional[cohere.EmbedInputType] = None,
    ) -> List[List[float]]:
        response = await self.aembed_with_retry(
            model=self.model,
            texts=texts,
            input_type=input_type,
            truncate=self.truncate,
            embedding_types=self.embedding_types,
        )
        embeddings = response.dict().get("embeddings", [])
        embeddings_as_float: List[List[float]] = []
        for embedding_type in self.embedding_types:
            e: List[List[Union[int, float]]] = embeddings.get(embedding_type)
            if not e:
                continue
            for i in range(len(e)):
                embeddings_as_float.append(list(map(float, e[i])))
        return embeddings_as_float

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self.embed(texts, input_type="search_document")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return await self.aembed(texts, input_type="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed([text], input_type="search_query")[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return (await self.aembed([text], input_type="search_query"))[0]
