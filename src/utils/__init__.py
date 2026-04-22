import math
import os
import sys
from typing import Dict, Sequence

import yaml
import tiktoken
import numpy as np
from langchain_core.documents import Document

from src.exception import MyException
from src.observability.logger import logging


def num_tokens_from_string(text: str, model_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a common encoding if model_name is not directly supported
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def load_configs(config_dir: str) -> Dict:
    """Load all configuration files from the specified directory."""
    try:
        configs = {}
        for name in ["ingestion", "chunking", "retrieval", "generation", "pipeline"]:
            path = os.path.join(config_dir, f"{name}.yaml")
            if os.path.exists(path):
                configs.update(read_yaml_file(path) or {})
            else:
                logging.warning("Config file missing: %s", path)
        return configs
    except Exception as e:
        raise MyException(e, sys) from e


def compute_k(*, total: int, pct: float | None, upper_bound: int) -> int:
    """
    Convert a percentage to an integer k, clamped to available docs.

    - Uses ceil to avoid losing small fractions.
    - Ensures the value is >= 0 and <= upper_bound.
    """
    if total <= 0 or upper_bound <= 0:
        return 0

    if pct is None:
        return 0

    calculated = int(math.ceil(total * pct))
    return max(0, min(calculated, upper_bound))


def count_documents(vector_store) -> int:
    """
    Safely count documents in a FAISS store.
    """
    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        try:
            return len(ids)
        except Exception:
            pass

    docstore = getattr(vector_store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        try:
            return len(docstore._dict)
        except Exception:
            pass
    return 0

def cosine_similarity(vec_a: np.ndarray, norm_a: float, vec_b: np.ndarray, norm_b: float) -> float:
    """Compute cosine similarity using pre-computed norms."""
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


# ----------------------------
# RAG Pipeline Helper Functions
# ----------------------------

def build_context(docs: Sequence[Document], include_citations: bool = False) -> str:
    """Concatenate documents into a single context string."""
    
    parts = [doc.page_content for doc in docs]
    return "\n\n".join(parts)
