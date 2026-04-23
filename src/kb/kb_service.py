from sklearn.metrics.pairwise import cosine_similarity
from src.kb.kb_store import KB_STORE
from src.embedding.embedder import HuggingFaceEmbedder
from src.exception import MyException
from src.utils import read_yaml_file
import numpy as np
import sys

# Initialize embedder
model = HuggingFaceEmbedder().get_embedder()
# Prepare KB
kb_items = list(KB_STORE.items())
kb_texts = [value for _, value in kb_items]
# Embed KB documents
kb_embeddings = model.embed_documents(kb_texts)
# Load config
kb_cfg = read_yaml_file("config/settings.yaml").get("KB", {})
similarity_threshold = kb_cfg["similarity_threshold"]
top_k_relevant = kb_cfg["top_k_relevant"]


def fetch_from_kb(query: str) -> str | None:
    """
    Fetch top-N most relevant KB entries using semantic similarity
    and return them as a formatted string.
    """
    try:
        query = query.lower().strip()

        # Embed query
        query_emb = model.embed_query(query)

        # Convert to numpy
        query_emb = np.array(query_emb).reshape(1, -1)
        kb_embeddings_np = np.array(kb_embeddings)

        # Compute similarity scores
        scores = cosine_similarity(query_emb, kb_embeddings_np)[0]

        # Sort indices by score (descending)
        sorted_indices = scores.argsort()[::-1]

        selected_results = []

        for idx in sorted_indices:
            score = float(scores[idx])

            if score < similarity_threshold:
                continue

            _, value = kb_items[idx]

            # Avoid duplicates
            if value not in selected_results:
                selected_results.append(value)

            if len(selected_results) >= top_k_relevant:
                break

        if not selected_results:
            return None

        # Format output cleanly
        formatted_output = "\n\n".join(
            [f"{i+1}. {text}" for i, text in enumerate(selected_results)]
        )

        return formatted_output

    except Exception as e:
        raise MyException(f"Error during KB fetching: {e}", sys)