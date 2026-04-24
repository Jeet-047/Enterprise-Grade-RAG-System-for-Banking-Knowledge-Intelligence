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

    if not detection.get("is_hallucinated"):
        return {
            "retrieved_chunks": context_chunks,
            "similarity_score": float(detection.get("similarity_score", 0.0)),
            "hallucination_decision": False,
        }
    else:
        token = pipeline.request_kb_token()
        kb_data = pipeline.secure_kb_fetch(token, payload.query)
        kb_answer = pipeline._answer_with_kb(payload.query, kb_data)
        return {
            "retrieved_chunks": context_chunks,
            "similarity_score": float(detection.get("similarity_score", 0.0)),
            "hallucination_decision": True,
            "kb_used": True,
            "kb_data": kb_data,
            "kb_answer": kb_answer
        }



@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/retrieval/logs") # For Retrieval observability
def retrieval_logs():
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    if not log_dir.exists():
        return {"message": "Logs directory not found", "path": str(log_dir)}

    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return {"message": "No log files found", "path": str(log_dir)}

    latest_log_file = max(log_files, key=lambda p: p.stat().st_mtime)
    file_content = latest_log_file.read_text(encoding="utf-8", errors="replace")
    # Add a blank line between log lines for better readability in browser JSON viewers.
    formatted_content = file_content.replace("\n", "\n\n")

    return {
        "file": latest_log_file.name,
        "content": formatted_content,
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
