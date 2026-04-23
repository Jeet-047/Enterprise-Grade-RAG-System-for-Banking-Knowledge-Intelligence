from fastapi import FastAPI

from src.api.routes_kb import router as kb_router
from src.api.routes_query import router as query_router
from src.api.routes_debug import router as debug_router

app = FastAPI(
    title="Enterprise-Grade RAG System for Banking Knowledge Intelligence",
    description="An robust, secure Retrieval-Augmented Generation (RAG) system for a banking domain.",
    version="1.0.0"
)

app.include_router(query_router)
app.include_router(kb_router)
app.include_router(debug_router)

# Root endpoint
@app.get("/")
def root():
    return {"message": "RAG API is running"}
