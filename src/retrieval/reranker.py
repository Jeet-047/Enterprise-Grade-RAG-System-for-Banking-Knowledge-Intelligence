import sys
import inspect
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank
from src.exception import MyException
from src.observability.logger import logging

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        config_path: str = "config/settings.yaml"
    ):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                logging.info("Loading NVIDIA reranker model: %s", self.model_name)
                self._model = NVIDIARerank(model=self.model_name)
            except Exception as e:
                raise MyException(e, sys)
        return self._model

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int | None = None
    ) -> List[Document]:
        if not documents:
            return []

        try:
            doc_list = list(documents)
            top_k = top_k or len(doc_list)
            compress_sig = inspect.signature(self.model.compress_documents)
            kwargs = {
                "query": query,
                "documents": doc_list,
            }
            if "top_n" in compress_sig.parameters:
                kwargs["top_n"] = top_k

            reranked_docs = self.model.compress_documents(**kwargs)
            if top_k is not None:
                reranked_docs = list(reranked_docs)[:top_k]

            logging.debug(
                "Reranked %d documents, returning %d", len(documents), len(reranked_docs)
            )
            return reranked_docs
        except Exception as e:
            raise MyException(e, sys)
