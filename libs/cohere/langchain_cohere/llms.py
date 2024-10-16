from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cohere
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.load.serializable import Serializable
from langchain_core.utils import secret_from_env
from pydantic import (
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

from .utils import _create_retry_decorator


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cohere.types import ListModelsResponse  # noqa: F401


def completion_with_retry(llm: Cohere, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm.max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return llm.client.generate(**kwargs)

    return _completion_with_retry(**kwargs)


def acompletion_with_retry(llm: Cohere, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm.max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await llm.async_client.generate(**kwargs)

    return _completion_with_retry(**kwargs)


class BaseCohere(Serializable):
    """Base class for Cohere models."""

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    
    chat_v2: Optional[Any] = None
    "Cohere chat v2."

    async_chat_v2: Optional[Any] = None
    "Cohere async chat v2."

    model: Optional[str] = Field(default=None)
    """Model name to use."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    cohere_api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("COHERE_API_KEY", default=None)
    )
    """Cohere API key. If not provided, will be read from the environment variable."""

    stop: Optional[List[str]] = None

    streaming: bool = Field(default=False)
    """Whether to stream the results."""

    user_agent: str = "langchain:partner"
    """Identifier for the application making the request."""

    timeout_seconds: Optional[float] = 300
    """Timeout in seconds for the Cohere API request."""

    base_url: Optional[str] = None
    """Override the default Cohere API URL."""

    def _get_default_model(self) -> str:
        """Fetches the current default model name."""
        response = self.client.models.list(default_only=True, endpoint="chat")  # type: "ListModelsResponse"
        if not response.models:
            raise Exception("invalid cohere list models response")
        if not response.models[0].name:
            raise Exception("invalid cohere list models response")
        return response.models[0].name

    @model_validator(mode="after")
    def validate_environment(self) -> Self:  # type: ignore[valid-type]
        """Validate that api key and python package exists in environment."""
        if isinstance(self.cohere_api_key, SecretStr):
            cohere_api_key: Optional[str] = self.cohere_api_key.get_secret_value()
        else:
            cohere_api_key = self.cohere_api_key
        client_name = self.user_agent
        timeout_seconds = self.timeout_seconds
        self.client = cohere.Client(
            api_key=cohere_api_key,
            timeout=timeout_seconds,
            client_name=client_name,
            base_url=self.base_url,
        )
        self.chat_v2 = self.client.v2.chat

        self.async_client = cohere.AsyncClient(
            api_key=cohere_api_key,
            client_name=client_name,
            timeout=timeout_seconds,
            base_url=self.base_url,
        )
        self.async_chat_v2 = self.async_client.v2.chat

        if not self.model:
            self.model = self._get_default_model()

        return self


class Cohere(LLM, BaseCohere):
    """Cohere large language models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_cohere import Cohere

            cohere = Cohere(cohere_api_key="my-api-key")
    """

    max_tokens: Optional[int] = None
    """Denotes the number of tokens to predict per generation."""

    k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    p: Optional[int] = None
    """Total probability mass of tokens to consider at each step."""

    frequency_penalty: Optional[float] = None
    """Penalizes repeated tokens according to frequency. Between 0 and 1."""

    presence_penalty: Optional[float] = None
    """Penalizes repeated tokens. Between 0 and 1."""

    truncate: Optional[str] = None
    """Specify how the client handles inputs longer than the maximum token
    length: Truncate from START, END or NONE"""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Configurable parameters for calling Cohere's generate API."""
        base_params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "k": self.k,
            "p": self.p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "truncate": self.truncate,
        }
        return {k: v for k, v in base_params.items() if v is not None}

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"cohere_api_key": "COHERE_API_KEY"}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "cohere"

    def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> dict:
        params = self._default_params
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop_sequences"] = self.stop
        else:
            params["stop_sequences"] = stop
        return {**params, **kwargs}

    def _process_response(self, response: Any, stop: Optional[List[str]]) -> str:
        text = response.generations[0].text
        # If stop tokens are provided, Cohere's endpoint returns them.
        # In order to make this consistent with other endpoints, we strip them.
        if stop:
            text = enforce_stop_tokens(text, stop)
        return text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Cohere's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = cohere("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        response = completion_with_retry(
            self, model=self.model, prompt=prompt, **params
        )
        _stop = params.get("stop_sequences")
        return self._process_response(response, _stop)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call out to Cohere's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = await cohere("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        response = await acompletion_with_retry(
            self, model=self.model, prompt=prompt, **params
        )
        _stop = params.get("stop_sequences")
        return self._process_response(response, _stop)
