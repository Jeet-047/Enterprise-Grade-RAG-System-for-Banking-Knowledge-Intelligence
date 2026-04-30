import sys
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank
from src.exception import MyException
from src.observability.logger import logging

class CrossEncoderReranker:
    """
    Reranker wrapper with pluggable backend.

    Supports:
    - NVIDIA API reranker via langchain_nvidia_ai_endpoints.NVIDIARerank
    - Local sentence-transformers CrossEncoder

    Returns documents ordered by relevance score (highest first).
    """

    def __init__(
        self,
        model_name: str,
        config_path: str = "config/settings.yaml"
    ):
        try:
            logging.info("Loading NVIDIA reranker model: %s", model_name)
            self.model = NVIDIARerank(model=model_name)
        except Exception as e:
            raise MyException(e, sys)

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int | None = None
    ) -> List[Document]:
        """
        Score and reorder candidate documents.

        Args:
            query: User query string.
            documents: Candidate documents to score.
            top_k: Optional cap on number of documents to return.

        Returns:
            Documents ordered by cross-encoder score (highest first).
        """
        if not documents:
            return []

        try:
            reranked_docs = self.model.compress_documents(
                query=query,
                documents=list(documents),
            )
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]

            logging.debug(
                "Reranked %d documents, returning %d", len(documents), len(reranked_docs)
            )
            return reranked_docs
        except Exception as e:
            raise MyException(e, sys)

