from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.routes_query import _get_pipeline, run_rag
from src.guardrails.hallucination import HallucinationDetector
from src.utils import build_context

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class EvaluateRequest(BaseModel):
    test_queries: list[str]


@router.post("/query/debug")
def query_debug(payload: QueryRequest):
    pipeline = _get_pipeline()
    docs = pipeline.retrieve(payload.query)
    answer = pipeline._answer_with_stuff(payload.query, docs) if docs else ""

    context_chunks = [doc.page_content for doc in docs]
    detection = HallucinationDetector().detect_hallucination(answer, build_context(docs) if docs else "")

    return {
        "retrieved_chunks": context_chunks,
        "similarity_score": float(detection.get("similarity_score", 0.0)),
        "hallucination_decision": bool(detection.get("is_hallucinated", False)),
    }


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/retrieval/logs") # For Retrieval observability
def retrieval_logs(query: str | None = None):
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    if not log_dir.exists():
        return {"logs": [], "message": "Logs directory not found", "path": str(log_dir)}

    log_files = sorted(log_dir.glob("*.log"))
    logs = []

    for log_file in log_files:
        file_content = log_file.read_text(encoding="utf-8", errors="replace")
        if query:
            if query.lower() in file_content.lower():
                logs.append({"file": log_file.name, "content": file_content})
        else:
            logs.append({"file": log_file.name, "content": file_content})

    return {
        "query": query,
        "total_files": len(log_files),
        "matched_files": len(logs),
        "logs": logs,
    }


@router.get("/chunks/inspect") # For Chunking analysis
def chunks_inspect():
    return {
        "chunks": [
            {"chunk_id": "sample-1", "text": "Sample chunk output for inspection."},
            {"chunk_id": "sample-2", "text": "Another sample chunk from preprocessing."},
        ]
    }


@router.post("/evaluate") # Testing endpoint
def evaluate(payload: EvaluateRequest):
    results = []
    for query in payload.test_queries:
        rag_result = run_rag(query)
        results.append(
            {
                "query": query,
                "final_answer": rag_result.get("final_answer", ""),
                "source": rag_result.get("source", "rag"),
                "confidence_score": float(rag_result.get("confidence_score", 0.0)),
            }
        )

    avg_confidence = (
        sum(item["confidence_score"] for item in results) / len(results) if results else 0.0
    )
    return {"total_queries": len(results), "average_confidence": avg_confidence, "results": results}
