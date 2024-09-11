"""Test HuggingFace embeddings."""

from typing import Type

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_cohere import CohereEmbeddings


class TestHuggingFaceEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[CohereEmbeddings]:
        return CohereEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "embed-english-light-v3.0"}
