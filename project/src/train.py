"""
Stage 3: LLM Inference & Context Orchestration
------------------------------------------------
Implements three RAG pipeline variants:

  Variant A — Local LLM (Ollama / llama3:8b-instruct)
  Variant B — API LLM  (Google Gemini 1.5 Flash)
  Variant C — Reranked (Gemini + cross-encoder reranking)

Anti-hallucination guardrail:
  If the max cosine similarity of retrieved chunks < CONFIDENCE_THRESHOLD,
  the pipeline returns a grounded refusal instead of an answer.
  Threshold = 0.35 — calibrated so that clearly off-topic queries are
  blocked while relevant queries (typically 0.5+) always proceed.

Run standalone:
    python src/train.py --query "What is the onboarding policy?" --variant b
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from dotenv import load_dotenv

from utils import (
    build_context_window,
    cosine_similarity,
    mean_cosine_similarity,
    Timer,
    load_chunks,
    require_env,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25   # Anti-hallucination guardrail
TOP_K                = 5      # Chunks to retrieve
MAX_CONTEXT_TOKENS   = 2000
MODELS_DIR           = Path("models")

SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer ONLY using the "
    "provided context. If the answer is not in the context, respond "
    "with: 'I cannot find this information in the provided documents.' "
    "Never speculate or add outside knowledge. Always cite the source "
    "document and page number."
)

REFUSAL_MESSAGE = (
    "I cannot find this information in the provided documents. "
    "The query did not match any sufficiently relevant context "
    "(confidence below threshold)."
)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def embed_query_local(query: str) -> np.ndarray:
    """
    Embed a query string using the local sentence-transformer model.

    Args:
        query: Natural language query string.

    Returns:
        Normalised 1-D float32 embedding vector.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec   = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32)


def embed_query_openai(query: str) -> np.ndarray:
    """
    Embed a query using OpenAI text-embedding-3-small.

    Args:
        query: Natural language query string.

    Returns:
        Normalised 1-D float32 embedding vector.
    """
    import openai
    client   = openai.OpenAI(api_key=require_env("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    )
    vec  = np.array(response.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec) + 1e-10
    return vec


def retrieve_chunks(
    query_vec: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = TOP_K,
) -> tuple[list[dict], np.ndarray]:
    """
    Search the FAISS index and return top_k chunks with their similarity scores.

    Args:
        query_vec: Normalised query embedding.
        index:     FAISS index to search.
        metadata:  Chunk metadata aligned with index rows.
        top_k:     Number of results to retrieve.

    Returns:
        (list_of_chunk_dicts, similarity_scores_array)
    """
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)
    scores   = scores[0]      # shape (top_k,)
    indices  = indices[0]

    chunks:  list[dict]  = []
    sim_scores: list[float] = []

    for idx, score in zip(indices, scores):
        if idx < 0:          # FAISS returns -1 for unfilled slots
            continue
        chunks.append(metadata[idx])
        sim_scores.append(float(score))

    return chunks, np.array(sim_scores, dtype=np.float32)


# ── Variant A — Local LLM (Ollama) ───────────────────────────────────────────

def query_variant_a(query: str) -> dict:
    """
    Variant A: Local FAISS retrieval + Ollama llama3:8b-instruct generation.

    Args:
        query: User query string.

    Returns:
        Dict with keys: answer, confidence, sources, latency_ms, variant.
    """
    import requests

    index    = faiss.read_index(str(MODELS_DIR / "local.faiss"))
    with open(MODELS_DIR / "local_meta.json", encoding="utf-8") as f:
        metadata = json.load(f)

    query_vec = embed_query_local(query)

    with Timer() as t:
        chunks, scores = retrieve_chunks(query_vec, index, metadata)

        max_score = float(scores.max()) if len(scores) else 0.0
        if max_score < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[A] Guardrail triggered — max_sim={max_score:.3f} < {CONFIDENCE_THRESHOLD}"
            )
            return {
                "answer":     REFUSAL_MESSAGE,
                "confidence": max_score,
                "sources":    [],
                "latency_ms": t.elapsed_ms,
                "variant":    "A-local",
                "refused":    True,
            }

        context = build_context_window(chunks, MAX_CONTEXT_TOKENS)
        prompt  = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":  "llama3:8b-instruct",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
        except Exception as exc:
            logger.error(f"[A] Ollama request failed: {exc}")
            answer = f"[LLM ERROR] {exc}"

    sources = [
        {
            "file":     c["source_file"],
            "page":     c["page_num"],
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]

    return {
        "answer":     answer,
        "confidence": float(np.mean(scores)),
        "sources":    sources,
        "latency_ms": t.elapsed_ms,
        "variant":    "A-local",
        "refused":    False,
    }


# ── Variant B — API LLM (Google Gemini 2.0 Flash) ────────────────────────────

def _call_gemini(prompt: str, system: str) -> str:
    """
    Call the Gemini 2.0 Flash API with a system + user prompt.

    Args:
        prompt: User-facing prompt (context + question).
        system: System instruction string.

    Returns:
        Generated text response.
    """
    import google.generativeai as genai

    genai.configure(api_key=require_env("GEMINI_API_KEY"))
    model    = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system,
        generation_config={"temperature": 0.0, "max_output_tokens": 512},
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def query_variant_b(query: str) -> dict:
    """
    Variant B: Local FAISS retrieval + Gemini 1.5 Flash generation.

    Args:
        query: User query string.

    Returns:
        Dict with keys: answer, confidence, sources, latency_ms, variant.
    """
    index = faiss.read_index(str(MODELS_DIR / "local.faiss"))
    with open(MODELS_DIR / "local_meta.json", encoding="utf-8") as f:
        metadata = json.load(f)

    query_vec = embed_query_local(query)

    with Timer() as t:
        chunks, scores = retrieve_chunks(query_vec, index, metadata)

        max_score = float(scores.max()) if len(scores) else 0.0
        if max_score < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[B] Guardrail triggered — max_sim={max_score:.3f} < {CONFIDENCE_THRESHOLD}"
            )
            return {
                "answer":     REFUSAL_MESSAGE,
                "confidence": max_score,
                "sources":    [],
                "latency_ms": t.elapsed_ms,
                "variant":    "B-gemini",
                "refused":    True,
            }

        context = build_context_window(chunks, MAX_CONTEXT_TOKENS)
        prompt  = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            answer = _call_gemini(prompt, SYSTEM_PROMPT)
        except Exception as exc:
            logger.error(f"[B] Gemini request failed: {exc}")
            answer = f"[LLM ERROR] {exc}"

    sources = [
        {"file": c["source_file"], "page": c["page_num"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]

    return {
        "answer":     answer,
        "confidence": float(np.mean(scores)),
        "sources":    sources,
        "latency_ms": t.elapsed_ms,
        "variant":    "B-gemini",
        "refused":    False,
    }


# ── Variant C — Reranked (Gemini + cross-encoder) ────────────────────────────

def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Rerank candidate chunks using a cross-encoder (ms-marco-MiniLM-L-6-v2).

    Cross-encoders jointly encode the query and each passage, producing
    a relevance score that is more accurate than bi-encoder similarity.

    Args:
        query:  Original query string.
        chunks: Candidate chunks retrieved by FAISS.
        top_k:  Final number of chunks to keep after reranking.

    Returns:
        Top-k reranked chunks (most relevant first).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    pairs = [[query, c["text"]] for c in chunks]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)
        scores = logits.numpy()

    ranked_indices = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in ranked_indices]


def query_variant_c(query: str) -> dict:
    """
    Variant C: FAISS retrieval → cross-encoder reranking → Gemini generation.

    Retrieves top 2*TOP_K candidates, reranks with cross-encoder, then
    passes the best TOP_K chunks to Gemini.

    Args:
        query: User query string.

    Returns:
        Dict with keys: answer, confidence, sources, latency_ms, variant.
    """
    index = faiss.read_index(str(MODELS_DIR / "local.faiss"))
    with open(MODELS_DIR / "local_meta.json", encoding="utf-8") as f:
        metadata = json.load(f)

    query_vec = embed_query_local(query)

    with Timer() as t:
        # Retrieve more candidates for reranking
        candidate_chunks, candidate_scores = retrieve_chunks(
            query_vec, index, metadata, top_k=TOP_K * 2
        )

        max_score = float(candidate_scores.max()) if len(candidate_scores) else 0.0
        if max_score < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[C] Guardrail triggered — max_sim={max_score:.3f} < {CONFIDENCE_THRESHOLD}"
            )
            return {
                "answer":     REFUSAL_MESSAGE,
                "confidence": max_score,
                "sources":    [],
                "latency_ms": t.elapsed_ms,
                "variant":    "C-reranked",
                "refused":    True,
            }

        # Cross-encoder rerank
        try:
            chunks = rerank_chunks(query, candidate_chunks, top_k=TOP_K)
        except Exception as exc:
            logger.warning(f"[C] Reranking failed ({exc}), using FAISS order.")
            chunks = candidate_chunks[:TOP_K]

        context = build_context_window(chunks, MAX_CONTEXT_TOKENS)
        prompt  = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            answer = _call_gemini(prompt, SYSTEM_PROMPT)
        except Exception as exc:
            logger.error(f"[C] Gemini request failed: {exc}")
            answer = f"[LLM ERROR] {exc}"

    sources = [
        {"file": c["source_file"], "page": c["page_num"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]

    return {
        "answer":     answer,
        "confidence": float(np.mean(candidate_scores[:TOP_K])),
        "sources":    sources,
        "latency_ms": t.elapsed_ms,
        "variant":    "C-reranked",
        "refused":    False,
    }


# ── Public dispatch ───────────────────────────────────────────────────────────

VARIANT_MAP = {
    "a": query_variant_a,
    "b": query_variant_b,
    "c": query_variant_c,
}


def run_query(query: str, variant: str = "b") -> dict:
    """
    Dispatch a query to the specified pipeline variant.

    Args:
        query:   Natural language question.
        variant: One of 'a' (local), 'b' (gemini), 'c' (reranked).

    Returns:
        Result dict from the chosen variant.
    """
    if variant not in VARIANT_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: a, b, c")
    logger.info(f"Running variant '{variant.upper()}' for query: {query!r}")
    return VARIANT_MAP[variant](query)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3 — LLM Inference & Context Orchestration"
    )
    parser.add_argument("--query",   required=True, help="Query string")
    parser.add_argument(
        "--variant", default="b", choices=["a", "b", "c"],
        help="Pipeline variant: a=local, b=gemini, c=reranked (default: b)",
    )
    args = parser.parse_args()

    result = run_query(args.query, args.variant)

    print("\n" + "=" * 60)
    print(f"Variant  : {result['variant']}")
    print(f"Refused  : {result.get('refused', False)}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Latency  : {result['latency_ms']:.1f} ms")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  • {s['file']} page {s['page']}  [{s['chunk_id']}]")
    print("=" * 60)
