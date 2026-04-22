import sys
from typing import List, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.exception import MyException
from src.observability.logger import logging
from src.retrieval.reranker import CrossEncoderReranker
from src.utils import compute_k, count_documents, cosine_similarity


class RerankMMRRetriever:
    """
    Retrieve -> rerank -> diversify (MMR) pipeline built around FAISS.

    Usage:
        retriever = RerankMMRRetriever(vector_store, reranker)
        docs = retriever.retrieve("query")
    """

    def __init__(
        self,
        vector_store: FAISS,
        reranker: CrossEncoderReranker,
        embedder: Embeddings | None = None,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.embedder = embedder or getattr(vector_store, "embedding_function", None)

        if self.embedder is None:
            raise MyException("No embedding function available for MMR.", sys)

    def retrieve(
        self,
        query: str,
        *,
        initial_pct: float | None = None,
        rerank_pct: float | None = None,
        mmr_pct: float | None = None,
        lambda_mult: float = 0.5,
        min_chunk: int | None = None,
    ) -> List[Document]:
        """
        Run vector search -> rerank -> MMR over the reranked set.
        Args:
            query: Search query.
            initial_pct: Percentage of total chunks to fetch from vector search.
            rerank_pct: Percentage of initial_k to keep after reranking.
            mmr_pct: Percentage of rerank_k to keep after MMR.
            lambda_mult: Trade-off for MMR (1.0 = purely relevance).
            min_chunk: If total chunks <= min_chunk, skip rerank/MMR and return all.
        """
        try:
            try:
                total_docs = count_documents(self.vector_store)
                logging.info("Total documents in vector store: %d", total_docs)
            except Exception as e:
                logging.error("Failed to count documents in vector store: %s", e)
                raise MyException(f"Vector store access failed: {e}", sys)
            
            if total_docs == 0:
                logging.warning("Vector store is empty. No documents to retrieve.")
                return []
            
            # Short-circuit for small corpora
            if min_chunk is not None and total_docs <= min_chunk:
                logging.info(
                    "Total documents (%d) <= min_chunk (%d). Skipping rerank/MMR.",
                    total_docs,
                    min_chunk,
                )
                return self.vector_store.similarity_search(query, k=total_docs)
            
            initial_k_final = compute_k(
                total=total_docs,
                pct=initial_pct,
                upper_bound=total_docs,
            )
            rerank_k_final = compute_k(
                total=initial_k_final,
                pct=rerank_pct,
                upper_bound=initial_k_final,
            )
            mmr_k_final = compute_k(
                total=rerank_k_final,
                pct=mmr_pct,
                upper_bound=rerank_k_final,
            )

            if initial_k_final <= 0:
                logging.warning("Computed initial_k is 0. No documents will be retrieved.")
                return []
            
            if rerank_k_final <= 0:
                logging.warning("Computed rerank_k is 0. Adjusting to use at least 1 document.")
                rerank_k_final = min(1, initial_k_final)
            
            if mmr_k_final <= 0:
                logging.warning("Computed mmr_k is 0. Adjusting to use at least 1 document.")
                mmr_k_final = min(1, rerank_k_final)

            try:
                initial_docs = self.vector_store.similarity_search(query, k=initial_k_final)
                logging.info(
                    "Initial vector search returned %d docs (k=%d)",
                    len(initial_docs),
                    initial_k_final,
                )
            except Exception as e:
                logging.error("Initial vector search failed: %s", e)
                raise MyException(f"Vector search failed: {e}", sys)

            try:
                reranked_docs = self.reranker.rerank(
                    query, initial_docs, top_k=rerank_k_final
                )
                logging.info(
                    "Reranked docs down to %d (rerank_k=%d)",
                    len(reranked_docs),
                    rerank_k_final,
                )
            except Exception as e:
                logging.error("Reranking failed: %s", e)
                raise MyException(f"Reranking failed: {e}", sys)

            try:
                diversified_docs = self._apply_mmr(
                    query, reranked_docs, k=mmr_k_final, lambda_mult=lambda_mult
                )
                logging.info(
                    "MMR selected %d docs (mmr_k=%d)", len(diversified_docs), mmr_k_final
                )
                return diversified_docs
            except Exception as e:
                logging.error("MMR diversification failed: %s", e)
                raise MyException(f"MMR diversification failed: {e}", sys)
        except MyException:
            raise
        except Exception as e:
            logging.error("Unexpected error in retrieve: %s", e)
            raise MyException(e, sys)

    def _apply_mmr(
        self, query: str, candidates: Sequence[Document], k: int, lambda_mult: float
    ) -> List[Document]:
        """Apply maximal marginal relevance over reranked candidates."""
        if not candidates or k <= 0:
            return []

        try:
            query_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
            logging.debug("Successfully embedded query for MMR.")
        except Exception as e:
            logging.error("Failed to embed query in MMR: %s", e)
            raise MyException(f"Query embedding failed in MMR: {e}", sys)

        try:
            doc_vecs = [
                np.array(vec, dtype=np.float32)
                for vec in self.embedder.embed_documents(
                    [doc.page_content for doc in candidates]
                )
            ]
            logging.debug("Successfully embedded %d documents for MMR.", len(doc_vecs))
        except Exception as e:
            logging.error("Failed to embed documents in MMR: %s", e)
            raise MyException(f"Document embedding failed in MMR: {e}", sys)

        # Pre-compute norms to avoid redundant calculations in the MMR selection loop
        try:
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.array([np.linalg.norm(vec) for vec in doc_vecs], dtype=np.float32)
            logging.debug("Pre-computed norms for query and documents.")
        except Exception as e:
            logging.error("Failed to compute norms for MMR: %s", e)
            raise MyException(f"Norm computation failed in MMR: {e}", sys)

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        try:
            while remaining and len(selected) < k:
                if not selected:
                    # Pick best relevance to query
                    chosen = max(
                        remaining,
                        key=lambda idx: cosine_similarity(query_vec, query_norm, doc_vecs[idx], doc_norms[idx])
                    )
                else:
                    # Pick document with best balance of relevance and diversity
                    def mmr_score(idx):
                        relevance = cosine_similarity(query_vec, query_norm, doc_vecs[idx], doc_norms[idx])
                        redundancy = max(
                            cosine_similarity(doc_vecs[idx], doc_norms[idx], doc_vecs[sel_idx], doc_norms[sel_idx])
                            for sel_idx in selected
                        )
                        return lambda_mult * relevance - (1 - lambda_mult) * redundancy
                    
                    chosen = max(remaining, key=mmr_score)
                
                selected.append(chosen)
                remaining.remove(chosen)
            
            logging.debug("MMR selection completed with %d documents.", len(selected))
            return [candidates[idx] for idx in selected]
        except Exception as e:
            logging.error("Failed during MMR selection loop: %s", e)
            raise MyException(f"MMR selection failed: {e}", sys)

