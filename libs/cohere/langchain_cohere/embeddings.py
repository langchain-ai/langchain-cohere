import typing
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import cohere
from cohere.types.embed_by_type_response import EmbedByTypeResponse
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from .utils import _create_retry_decorator

_DEFAULT_INPUT_TYPE: "cohere.EmbedInputType" = "search_document"


class CohereEmbeddings(BaseModel, Embeddings):
    """
    Implements the `Embeddings` interface with Cohere's text representation language
    models.

    Find out more about us at https://cohere.com and https://huggingface.co/CohereForAI

    This implementation uses the Embed API - see https://docs.cohere.com/reference/embed

    To use this you'll need to a Cohere API key - either pass it to cohere_api_key
    parameter or set the `COHERE_API_KEY` environment variable.

    API keys are available on https://cohere.com - it's free to sign up and trial API
    keys work with this implementation.

    Basic Example:
        ```python
        cohere_embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
        text = "This is a test document."

        query_result = cohere_embeddings.embed_query(text)
        print(query_result)

        doc_result = cohere_embeddings.embed_documents([text])
        print(doc_result)
        ```
    """

    client: Any  #: :meta private:
    """Cohere client."""

    async_client: Any  #: :meta private:
    """Cohere async client."""

    model: Optional[str] = None
    """Model name to use. It is mandatory to specify the model name."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end `("NONE"|"START"|"END")`"""  # noqa: E501

    cohere_api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("COHERE_API_KEY", default=None)
    )

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        cohere_api_key = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY"
        )
        if isinstance(cohere_api_key, SecretStr):
            cohere_api_key = cohere_api_key.get_secret_value()

        request_timeout = values.get("request_timeout")

        client_name = values.get("user_agent", "langchain:partner")
        values["client"] = cohere.ClientV2(
            cohere_api_key,
            timeout=request_timeout,
            client_name=client_name,
            base_url=values.get("base_url"),
        )
        values["async_client"] = cohere.AsyncClientV2(
            cohere_api_key,
            timeout=request_timeout,
            client_name=client_name,
            base_url=values.get("base_url"),
        )

        return values

    @model_validator(mode="after")
    def validate_model_specified(self) -> Self:  # type: ignore[valid-type]
        """Validate that model is specified."""
        if not self.model:
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

    def _resolve_input_type(
        self,
        input_type: typing.Optional[cohere.EmbedInputType],
    ) -> cohere.EmbedInputType:
        if input_type is not None:
            return input_type
        warnings.warn(
            (
                "Calling CohereEmbeddings.embed/aembed without `input_type` is "
                "deprecated. The Cohere v2 Embed API requires `input_type` to be "
                f"set explicitly; defaulting to {_DEFAULT_INPUT_TYPE!r} for "
                "backward compatibility. This fallback will be removed in a "
                "future release."
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        return _DEFAULT_INPUT_TYPE

    def _build_embed_kwargs(
        self,
        texts: List[str],
        input_type: cohere.EmbedInputType,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": self.embedding_types,
        }
        if self.truncate is not None:
            kwargs["truncate"] = self.truncate
        return kwargs

    def _embed_response_to_float(self, response: EmbedByTypeResponse) -> List[List[float]]:
        embeddings = response.dict().get("embeddings", {}) or {}
        embeddings_as_float: List[List[float]] = []
        for embedding_type in self.embedding_types:
            e: List[List[Union[int, float]]] = embeddings.get(embedding_type)
            if not e:
                continue
            for i in range(len(e)):
                embeddings_as_float.append(list(map(float, e[i])))
        return embeddings_as_float

    def embed(
        self,
        texts: List[str],
        *,
        input_type: typing.Optional[cohere.EmbedInputType] = None,
    ) -> List[List[float]]:
        resolved_input_type = self._resolve_input_type(input_type)
        response = self.embed_with_retry(
            **self._build_embed_kwargs(texts, resolved_input_type)
        )
        return self._embed_response_to_float(response)

    async def aembed(
        self,
        texts: List[str],
        *,
        input_type: typing.Optional[cohere.EmbedInputType] = None,
    ) -> List[List[float]]:
        resolved_input_type = self._resolve_input_type(input_type)
        response = await self.aembed_with_retry(
            **self._build_embed_kwargs(texts, resolved_input_type)
        )
        return self._embed_response_to_float(response)

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
