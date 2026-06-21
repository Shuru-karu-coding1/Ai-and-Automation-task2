"""
Stage 5: API Deployment — FastAPI Application
----------------------------------------------
Exposes the RAG pipeline as a production-ready REST API.

Endpoints:
  POST /query   — Submit a question, receive a grounded answer + sources
  GET  /health  — Pipeline readiness check
  GET  /docs    — Auto-generated Swagger UI (FastAPI default)

Start the server:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Example request:
    curl -X POST http://localhost:8000/query \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is the data retention policy?"}'
"""

import os
import sys
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Allow imports from src/ when running as `uvicorn api.app:app`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import run_query, CONFIDENCE_THRESHOLD

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

MODELS_DIR = Path("models")

# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    """Holds pre-loaded pipeline resources shared across requests."""

    index_loaded: bool = False
    index:        Optional[faiss.Index] = None
    metadata:     Optional[list[dict]]  = None
    startup_time: float = 0.0


state = AppState()


# ── Lifespan — load indices at startup ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the serialised FAISS index and chunk metadata at server startup.
    Frees resources (if any) on shutdown.
    """
    logger.info("Loading pipeline indices ...")
    t0 = time.perf_counter()

    try:
        index_path = MODELS_DIR / "local.faiss"
        meta_path  = MODELS_DIR / "local_meta.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(
                "Index files not found — run Stage 1 and Stage 2 first. "
                "Server will start but /query will return 503."
            )
        else:
            state.index = faiss.read_index(str(index_path))
            with open(meta_path, encoding="utf-8") as f:
                state.metadata = json.load(f)
            state.index_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Index loaded — {state.index.ntotal} vectors, "
                f"{len(state.metadata)} chunks in {elapsed:.2f}s"
            )
    except Exception as exc:
        logger.error(f"Index load failed: {exc}")

    state.startup_time = time.perf_counter()
    yield
    logger.info("Shutting down API.")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Knowledge Retrieval API",
    description=(
        "Semantic document retrieval system with anti-hallucination guardrails. "
        "Answers questions grounded in your document corpus and returns verifiable "
        "source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Input payload for the /query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question to answer from the document corpus.",
        examples=["What is the data retention policy for customer records?"],
    )
    variant: str = Field(
        default="b",
        description="Pipeline variant: 'a'=local LLM, 'b'=Gemini, 'c'=reranked.",
        examples=["b"],
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        """Reject queries that are empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Query must not be blank.")
        return v.strip()

    @field_validator("variant")
    @classmethod
    def valid_variant(cls, v: str) -> str:
        if v not in ("a", "b", "c"):
            raise ValueError("variant must be one of: a, b, c")
        return v


class SourceDocument(BaseModel):
    """Metadata for a single retrieved source chunk."""

    file:     str = Field(..., description="Source PDF filename.")
    page:     int = Field(..., description="Page number within the PDF.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")


class QueryResponse(BaseModel):
    """
    Response payload for the /query endpoint.

    Fields:
        answer:     Generated answer grounded in retrieved context.
        confidence: Mean cosine similarity of retrieved chunks (0–1).
                    Values ≥ 0.35 indicate the query was answered;
                    values < 0.35 trigger the anti-hallucination refusal.
        sources:    List of source documents used to generate the answer.
        latency_ms: End-to-end pipeline latency in milliseconds.
        variant:    Pipeline variant that produced this response.
        refused:    True if the guardrail blocked the answer.
    """

    answer:     str             = Field(..., description="Grounded answer text.")
    confidence: float           = Field(..., description="Retrieval confidence score (0–1).", ge=0.0, le=1.0)
    sources:    list[SourceDocument] = Field(..., description="Source documents cited.")
    latency_ms: float           = Field(..., description="End-to-end latency in ms.")
    variant:    str             = Field(..., description="Pipeline variant identifier.")
    refused:    bool            = Field(False, description="True if guardrail refused to answer.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Customer records must be retained for a minimum of 7 years per compliance regulations (Source: policy_manual.pdf, Page 14).",
                    "confidence": 0.82,
                    "sources": [
                        {"file": "policy_manual.pdf", "page": 14, "chunk_id": "a3f2b1c4d5e6"}
                    ],
                    "latency_ms": 1240.5,
                    "variant": "B-gemini",
                    "refused": False,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response payload for the /health endpoint."""

    status:        str  = Field(..., description="'ok' or 'degraded'.")
    index_loaded:  bool = Field(..., description="Whether the vector index is loaded.")
    index_vectors: int  = Field(..., description="Number of vectors in the FAISS index.")
    uptime_s:      float = Field(..., description="Server uptime in seconds.")


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every incoming request with method, path, and response time.
    """
    start  = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed:.1f} ms)"
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the document knowledge base",
    tags=["RAG"],
)
async def query_endpoint(payload: QueryRequest) -> QueryResponse:
    """
    Submit a natural language question and receive a grounded answer with sources.

    The pipeline:
    1. Embeds the query with sentence-transformers.
    2. Retrieves top-5 semantically similar chunks from FAISS.
    3. Applies anti-hallucination guardrail (refuses if max_sim < 0.35).
    4. Generates an answer via the selected LLM variant.
    5. Returns answer + confidence score + source metadata.

    **Variants:**
    - `a` — Local Ollama (llama3:8b-instruct) — zero external dependency
    - `b` — Google Gemini 1.5 Flash — best quality/speed balance (recommended)
    - `c` — Gemini + cross-encoder reranking — highest precision

    **Confidence score:** Mean cosine similarity of retrieved chunks.
    Values ≥ 0.35 indicate a relevant retrieval; lower values trigger refusal.
    """
    if not state.index_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector index is not loaded. "
                "Run Stage 1 (preprocess.py) and Stage 2 (features.py) first, "
                "then restart the server."
            ),
        )

    query = payload.query
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    logger.info(f"Query [{payload.variant.upper()}]: {query!r}")

    try:
        result = run_query(query, payload.variant)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    logger.info(
        f"Response — confidence={result['confidence']:.3f}, "
        f"latency={result['latency_ms']:.0f}ms, "
        f"refused={result.get('refused', False)}"
    )

    sources = [SourceDocument(**s) for s in result.get("sources", [])]

    return QueryResponse(
        answer     = result["answer"],
        confidence = round(float(result["confidence"]), 4),
        sources    = sources,
        latency_ms = round(result["latency_ms"], 1),
        variant    = result["variant"],
        refused    = result.get("refused", False),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Pipeline health check",
    tags=["System"],
)
async def health_endpoint() -> HealthResponse:
    """
    Return the current health status of the pipeline.

    Checks whether the FAISS index is loaded and how many vectors are available.
    Use this endpoint for liveness/readiness probes.
    """
    n_vectors = state.index.ntotal if state.index_loaded and state.index else 0
    uptime    = time.perf_counter() - state.startup_time

    return HealthResponse(
        status        = "ok" if state.index_loaded else "degraded",
        index_loaded  = state.index_loaded,
        index_vectors = n_vectors,
        uptime_s      = round(uptime, 1),
    )


@app.get("/", include_in_schema=False)
async def root():
    """Redirect users to the interactive API docs."""
    return JSONResponse(
        {"message": "RAG API is running. Visit /docs for the Swagger UI."}
    )
