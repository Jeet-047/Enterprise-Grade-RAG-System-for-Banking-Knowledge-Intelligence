from functools import lru_cache
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.rag.pipeline import RAGPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    source: str
    confidence_score: float


@lru_cache(maxsize=1)
def _get_pipeline() -> "RAGPipeline":
    # Import lazily to avoid heavy model initialization at API startup.
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.prepare_vector_store()
    return pipeline


def run_rag(query: str) -> dict[str, Any]:
    return _get_pipeline().answer(query)


@router.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    result = run_rag(payload.query)
    return {
        "final_answer": result.get("final_answer", ""),
        "source": result.get("source", "rag"),
        "confidence_score": float(result.get("confidence_score", 0.0)),
    }
