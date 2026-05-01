from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import APIRouter, File, Form, UploadFile
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


class IndexResponse(BaseModel):
    indexed_documents: int
    uploaded_files: list[str]
    message: str


@lru_cache(maxsize=1)
def _get_pipeline() -> "RAGPipeline":
    # Import lazily to avoid heavy model initialization at API startup.
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.prepare_vector_store()
    return pipeline


def run_rag(query: str) -> dict[str, Any]:
    return _get_pipeline().answer(query)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _save_uploads(files: list[UploadFile], target_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        filename = Path(file.filename or "").name
        if not filename:
            continue
        destination = target_dir / filename
        content = file.file.read()
        destination.write_bytes(content)
        saved_paths.append(destination)
    return saved_paths


def _write_documents_to_config(config_path: Path, document_paths: list[str]) -> None:
    cfg: dict[str, Any] = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg["documents"] = [{"path": p.replace("\\", "/"), "enabled": True} for p in document_paths]
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


@router.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    result = run_rag(payload.query)
    return {
        "final_answer": result.get("final_answer", ""),
        "source": result.get("source", "rag"),
        "confidence_score": float(result.get("confidence_score", 0.0)),
    }


def _parse_web_urls(raw: str) -> list[str]:
    urls: list[str] = []
    for line in (raw or "").splitlines():
        u = line.strip()
        if u.startswith(("http://", "https://")):
            urls.append(u)
    return urls


@router.post("/documents/index", response_model=IndexResponse)
def index_documents(
    files: list[UploadFile] | None = File(None),
    web_urls: str = Form(""),
):
    allowed_extensions = {".pdf", ".txt", ".doc", ".docx", ".csv"}
    upload_list = list(files or [])
    valid_files = [
        file for file in upload_list
        if Path(file.filename or "").suffix.lower() in allowed_extensions
    ]
    url_list = _parse_web_urls(web_urls)

    if not valid_files and not url_list:
        return {
            "indexed_documents": 0,
            "uploaded_files": [],
            "message": "No supported files or URLs. Upload pdf, txt, doc, docx, or csv—or add https:// URLs (one per line).",
        }

    root = _project_root()
    upload_dir = root / "data" / "raw" / "uploaded"
    config_path = root / "config" / "settings.yaml"

    saved_paths = _save_uploads(valid_files, upload_dir) if valid_files else []
    indexed_paths = [str(p).replace("\\", "/") for p in saved_paths]
    document_paths = indexed_paths + url_list
    _write_documents_to_config(config_path, document_paths)

    _get_pipeline.cache_clear()
    _get_pipeline()

    uploaded_labels = [Path(p).name for p in indexed_paths]
    uploaded_labels.extend(url_list)
    return {
        "indexed_documents": len(document_paths),
        "uploaded_files": uploaded_labels,
        "message": "Documents uploaded and indexed successfully.",
    }
