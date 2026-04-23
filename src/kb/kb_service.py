from sklearn.metrics.pairwise import cosine_similarity
from src.kb.kb_store import KB_STORE
from src.utils import read_yaml_file
from src.exception import MyException
import sys

_model = None
_kb_embeddings = None
_kb_keys = None
_SIMILARITY_THRESHOLD = None
_TOP_K_RELEVANT = None


def _initialize_kb_service() -> None:
    global _model, _kb_embeddings, _kb_keys, _SIMILARITY_THRESHOLD, _TOP_K_RELEVANT
    if _model is not None:
        return

    from sentence_transformers import SentenceTransformer

    kb_cfg = read_yaml_file("config/settings.yaml").get("KB", {})
    _SIMILARITY_THRESHOLD = kb_cfg["similarity_threshold"]
    _TOP_K_RELEVANT = kb_cfg["top_k_relevant"]

    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _kb_keys = list(KB_STORE.keys())
    _kb_embeddings = _model.encode(_kb_keys, show_progress_bar=False)


def fetch_from_kb(query) -> str | None:
    """
    Fetch the most relevant information from the knowledge base based on the query.
    """
    try:
        _initialize_kb_service()
        query_emb = _model.encode([query], show_progress_bar=False)

        scores = cosine_similarity(query_emb, _kb_embeddings)[0]
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        matched_values = []

        for idx in ranked_indices:
            if scores[idx] < _SIMILARITY_THRESHOLD:
                break
            matched_values.append(KB_STORE[_kb_keys[idx]])
            if len(matched_values) >= _TOP_K_RELEVANT:
                break

        if matched_values:
            return "\n\n".join(matched_values)

        return None

    except Exception as e:
        raise MyException(f"Error during KB fetching: {e}", sys)
