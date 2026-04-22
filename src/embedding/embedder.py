import sys
from langchain_huggingface import HuggingFaceEmbeddings
from src.observability.logger import logging
from src.exception import MyException


class HuggingFaceEmbedder:
    """
    Thin wrapper around LangChain's HuggingFaceEmbeddings to delay initialization
    until it is actually needed.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        # Model name should eventually come from configuration/constants.
        self.model_name = model_name
        self._embedder = None

    def get_embedder(self) -> HuggingFaceEmbeddings:
        """Create (once) and return the HuggingFace embedding model."""
        if self._embedder is None:
            try:
                logging.info("Initializing the HuggingFace embedder.")
                self._embedder = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                raise MyException(e, sys)
        return self._embedder