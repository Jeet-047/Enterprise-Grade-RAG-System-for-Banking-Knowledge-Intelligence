import os
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Sequence

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.exception import MyException
from src.ingestion.extractor import DocumentExtractor
from src.ingestion.loaders import DocumentLoader
from src.observability.logger import logging
from src.preprocessing.clean_normalize import DocumentNormalizationAndCleaning
from src.preprocessing.chunking import DocumentChunker
from src.rag.prompts import SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_FALLBACK_KB
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RerankMMRRetriever
from src.vectorstore.faiss_store import FaissVectorStore
from src.guardrails.hallucination import HallucinationDetector
from src.utils import read_yaml_file, build_context
from dotenv import load_dotenv

load_dotenv()

def _get_internal_api_key() -> str:
    """Read API key at call time to avoid stale import-time values."""
    return os.getenv("INTERNAL_API_KEY", "secret-key").strip()


def _ensure_valid_ssl_cert_env() -> None:
    """Remove broken SSL_CERT_FILE env var so httpx can use default cert store."""
    cert_file = os.getenv("SSL_CERT_FILE", "").strip()
    if cert_file and not os.path.exists(cert_file):
        logging.warning("Ignoring invalid SSL_CERT_FILE path: %s", cert_file)
        os.environ.pop("SSL_CERT_FILE", None)


class RAGPipeline:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self._config = None
        self._config_path = config_path
        self._llm = None
        self._reranker = None
        self._hallucination_detector = None
        self.vector_store = None
        self.retriever = None

    @property
    def config(self):
        if self._config is None:
            self._config = read_yaml_file(self._config_path)
        return self._config

    @property
    def llm(self):
        if self._llm is None:
            _ensure_valid_ssl_cert_env()
            gen_cfg = self.config.get("generation", {})
            self._llm = ChatGroq(
                model=gen_cfg["llm_model"],
                temperature=gen_cfg["temperature"],
                max_tokens=gen_cfg["max_output_tokens"],
                streaming=True
            )
        return self._llm

    @property
    def reranker(self):
        if self._reranker is None:
            retr_cfg = self.config.get("retrieval", {})
            self._reranker = CrossEncoderReranker(
                model_name=retr_cfg.get("reranker_model", "nv-rerank-qa-mistral-4b:1"))
        return self._reranker

    @property
    def kb_api_url(self):
        return self.config.get("KB", {}).get("kb_api_url")

    @property
    def hallucination_detector(self):
        if self._hallucination_detector is None:
            self._hallucination_detector = HallucinationDetector()
        return self._hallucination_detector

    # ----------------------------
    # Data preparation
    # ----------------------------
    def prepare_vector_store(self) -> None:
        docs_cfg = self.config.get("documents", {})
        if not docs_cfg:
            raise MyException("No documents configured for processing.", sys)

        chunk_cfg = self.config.get("chunking", {})
        similarity_threshold = chunk_cfg["similarity_threshold"]

        logging.info("Starting vector store preparation with %d document(s)", len(docs_cfg))

        loader = DocumentLoader()
        extractor = DocumentExtractor()
        cleaner = DocumentNormalizationAndCleaning()
        chunker = DocumentChunker()

        all_chunks = []
        for idx, doc_info in enumerate(docs_cfg, 1):
            if not doc_info.get("enabled", True):
                logging.info("Skipping disabled document: %s", doc_info.get("path", "unknown"))
                continue

            path = doc_info["path"]
            logging.info("[%d/%d] Processing document: %s", idx, len(docs_cfg), path)

            try:
                loaded = loader.load_document(path)
                extracted = extractor.extract_document_info(loaded, path)
                cleaned = cleaner.initialize_document_normalizer(extracted)
                cleaned = [d for d in cleaned if (d.get("text") or "").strip()]
                if not cleaned:
                    logging.warning("No extractable text found in document: %s", path)
                    continue
                chunks = chunker.chunk_document(cleaned, similarity_threshold)
                if not chunks:
                    logging.warning("No chunks generated from document (possibly low/empty text): %s", path)
                    continue

                logging.info("Generated %d chunks from document: %s", len(chunks), path)
                all_chunks.extend(chunks)
            except Exception as e:
                logging.error("Failed to process document %s: %s", path, e)
                raise MyException(f"Error processing document {path}: {e}", sys)

        if not all_chunks:
            raise MyException(
                "No chunks generated from configured documents. Source may have no extractable text (e.g., scanned PDF/image-only pages).",
                sys,
            )

        logging.info("Creating vector store with %d total chunks...", len(all_chunks))
        self.vector_store = FaissVectorStore().create_vector_store(all_chunks)
        self.retriever = RerankMMRRetriever(self.vector_store, self.reranker)
        logging.info("Vector store prepared successfully with %d chunks", len(all_chunks))

    # ----------------------------
    # Retrieval + Routing
    # ----------------------------
    def retrieve(self, query: str) -> List[Document]:
        if self.retriever is None:
            raise MyException("Retriever not initialized. Call prepare_vector_store().", sys)

        query_preview = query[:100]
        logging.info("Retrieving documents for query: %s", query_preview)

        retr_cfg = self.config.get("retrieval", {})
        optional_keys = ["lambda_mult", "initial_pct", "rerank_pct", "mmr_pct", "min_chunk"]
        retrieve_kwargs = {k: retr_cfg[k] for k in optional_keys if k in retr_cfg}

        documents = self.retriever.retrieve(query, **retrieve_kwargs)
        logging.info("Retrieved %d documents for query", len(documents))
        return documents

    def answer(self, query: str) -> dict:
        query_preview = query[:100]
        logging.info("Generating answer for query: %s", query_preview)

        documents = self.retrieve(query)
        if not documents:
            logging.warning("No documents retrieved for query: %s", query)
            return {
                "final_answer": "I don't have enough information to answer this question based on the provided documents.",
                "source": "rag",
                "confidence_score": 0.0,
            }

        answer = self._answer_with_stuff(query, documents)
        logging.info("Answer generated, checking hallucination")

        detection = self.hallucination_detector.detect_hallucination(answer, build_context(documents))
        confidence_score = detection.get("similarity_score", 0.0)

        if detection.get("is_hallucinated"):
            logging.warning("Hallucination detected for query, using secure KB fallback")
            token = self.request_kb_token()
            if token:
                kb_info = self.secure_kb_fetch(token, query)
                if kb_info and kb_info.get("data"):
                    final_answer = self._answer_with_kb(query, kb_info["data"])
                    return {
                        "final_answer": final_answer,
                        "source": "kb-secure",
                        "confidence_score": kb_info.get("score", 0.0),
                    }
                logging.warning("Secure KB fetch returned no match for query: %s", query)
            else:
                logging.warning("Secure KB token request failed for query: %s", query)
        else:
            logging.info("No hallucination detected, return final answer.")

        return {
            "final_answer": answer,
            "source": "rag",
            "confidence_score": confidence_score,
        }

    # ----------------------------
    # Prompting strategies
    # ----------------------------
    def _answer_with_stuff(self, query: str, docs: Sequence[Document]) -> str:
        context_str = build_context(docs, include_citations=False)
        user_prompt = USER_PROMPT.format(context=context_str, question=query)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        logging.info("Using Stuff strategy with %d docs", len(docs))
        logging.info("Context length: %d characters", len(context_str))

        response = self.llm.invoke(messages)
        answer = getattr(response, "content", str(response))
        logging.info("Generated answer length: %d characters", len(answer))
        return answer

    # ----------------------------
    # Secure KB Fallback Functions
    # ----------------------------

    def request_kb_token(self) -> str | None:
        url = urllib.parse.urljoin(self.kb_api_url, "/kb/token")
        request_headers = {
            "X-API-KEY": _get_internal_api_key(),
            "Content-Type": "application/json",
        }
        request_obj = urllib.request.Request(url, headers=request_headers, method="POST", data=b"")

        try:
            with urllib.request.urlopen(request_obj, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload.get("token")
        except urllib.error.HTTPError as exc:
            logging.error("KB token request failed with HTTP status %s", exc.code)
        except Exception as exc:
            logging.error("KB token request failed: %s", exc)
        return None

    def secure_kb_fetch(self, token: str, query: str) -> dict | None:
        query_string = urllib.parse.urlencode({"query": query})
        url = f"{self.kb_api_url}/kb/fetch?{query_string}"
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        request_obj = urllib.request.Request(url, headers=request_headers, method="POST", data=b"")

        try:
            with urllib.request.urlopen(request_obj, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                score = payload.get("score")
                if score is None:
                    score = 0.0
                return {"data": payload.get("data"), "score": float(score)}
        except urllib.error.HTTPError as exc:
            logging.error("Secure KB fetch failed with HTTP status %s", exc.code)
        except Exception as exc:
            logging.error("Secure KB fetch failed: %s", exc)
        return None

    def _answer_with_kb(self, query: str, kb_data: str) -> str:
        user_prompt = USER_PROMPT_FALLBACK_KB.format(kb_context=kb_data, question=query)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return getattr(response, "content", str(response))

    
