import os
from pathlib import Path

project_name = "src"

list_of_files = [

    # Core
    f"{project_name}/__init__.py",

    # Config
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/settings.py",

    # Ingestion
    f"{project_name}/ingestion/__init__.py",
    f"{project_name}/ingestion/loaders.py",
    f"{project_name}/ingestion/extractor.py",

    # Preprocessing
    f"{project_name}/preprocessing/__init__.py",
    f"{project_name}/preprocessing/clean_normalize.py",
    f"{project_name}/preprocessing/chunking.py",

    # Embeddings
    f"{project_name}/embedding/__init__.py",
    f"{project_name}/embedding/embedder.py",

    # Vector Store
    f"{project_name}/vectorstore/__init__.py",
    f"{project_name}/vectorstore/faiss_store.py",

    # Retrieval
    f"{project_name}/retrieval/__init__.py",
    f"{project_name}/retrieval/retriever.py",
    f"{project_name}/retrieval/reranker.py",

    # RAG Pipeline
    f"{project_name}/rag/__init__.py",
    f"{project_name}/rag/prompts.py",
    f"{project_name}/rag/pipeline.py",

    # Hallucination Detection
    f"{project_name}/guardrails/__init__.py",
    f"{project_name}/guardrails/hallucination.py",

    # Knowledge Base
    f"{project_name}/kb/__init__.py",
    f"{project_name}/kb/kb_store.py",
    f"{project_name}/kb/kb_service.py",

    # Token Security Layer
    f"{project_name}/security/__init__.py",
    f"{project_name}/security/token_manager.py",

    # API Layer
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/app.py",
    f"{project_name}/api/routes_query.py",
    f"{project_name}/api/routes_kb.py",
    f"{project_name}/api/routes_debug.py",

    # Observability / Logging
    f"{project_name}/observability/__init__.py",
    f"{project_name}/observability/logger.py",

    # Evaluation
    f"{project_name}/evaluation/__init__.py",
    f"{project_name}/evaluation/evaluator.py",

    # UI (Optional)
    f"{project_name}/ui/__init__.py",
    f"{project_name}/ui/streamlit_app.py",

    # Exceptions
    f"{project_name}/exception/__init__.py",

    # Root Files
    "requirements.txt",
    "Dockerfile",
    "pyproject.toml",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")
