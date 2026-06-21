"""
Stage 2: Embedding Generation & Indexing
-----------------------------------------
Implements three embedding strategies:
  1. LOCAL  — sentence-transformers (all-MiniLM-L6-v2) → FAISS
  2. API    — groq → FAISS
  3. HYBRID — BM25 sparse + dense vectors via Reciprocal Rank Fusion

All indices are serialised to models/ for reuse without re-embedding.

Run standalone:
    python src/features.py --chunks data/chunks.json
"""

import os
import json
import pickle
import logging
import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("features")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LOCAL — sentence-transformers ─────────────────────────────────────────

def build_local_index(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> tuple[faiss.Index, list[dict]]:
    """
    Encode chunks with a local sentence-transformer and build a FAISS L2 index.

    Args:
        chunks:     List of chunk dicts from Stage 1.
        model_name: Sentence-transformers model identifier.
        batch_size: Encoding batch size.

    Returns:
        (faiss_index, chunk_metadata_list)
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"[LOCAL] Loading model '{model_name}' ...")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    logger.info(f"[LOCAL] Encoding {len(texts)} chunks (batch={batch_size}) ...")

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    elapsed = time.perf_counter() - t0

    dim = embeddings.shape[1]
    logger.info(
        f"[LOCAL] Embeddings shape: {embeddings.shape}, "
        f"dim={dim}, took {elapsed:.1f}s"
    )

    index = faiss.IndexFlatIP(dim)   # Inner product on normalised vecs = cosine
    index.add(embeddings.astype(np.float32))
    logger.info(f"[LOCAL] FAISS index size: {index.ntotal} vectors")

    return index, chunks


def save_local_index(index: faiss.Index, metadata: list[dict]) -> None:
    """Serialise the local FAISS index and its metadata sidecar."""
    faiss.write_index(index, str(MODELS_DIR / "local.faiss"))
    with open(MODELS_DIR / "local_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("[LOCAL] Index saved → models/local.faiss + local_meta.json")


def load_local_index() -> tuple[faiss.Index, list[dict]]:
    """Load a previously serialised local FAISS index."""
    index = faiss.read_index(str(MODELS_DIR / "local.faiss"))
    with open(MODELS_DIR / "local_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    logger.info(f"[LOCAL] Loaded index — {index.ntotal} vectors")
    return index, meta


# ── 2. API — groq ───────────────────────────────────

def _groq_embed_batch(texts: list[str], client: Any) -> np.ndarray:
    """
    Embed a batch of texts with groq and return an (N, D) float32 array.

    Args:
        texts:  List of text strings.
        client: Initialised groq.Groq client.

    Returns:
        Numpy array of shape (len(texts), embedding_dim).
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype=np.float32)


def build_api_index(
    chunks: list[dict],
    batch_size: int = 100,
) -> tuple[faiss.Index, list[dict]]:
    """
    Encode chunks via groq text-embedding-3-small and build a FAISS index.

    Requires GROQ_API_KEY in environment / .env.

    Args:
        chunks:     Chunk list from Stage 1.
        batch_size: API call batch size (max 2048 for text-embedding-3-small).

    Returns:
        (faiss_index, chunk_metadata_list)
    """
    import groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    client = groq.Groq(api_key=api_key)
    texts  = [c["text"] for c in chunks]
    all_embeddings: list[np.ndarray] = []

    logger.info(f"[API] Embedding {len(texts)} chunks via groq ...")
    t0 = time.perf_counter()

    for i in tqdm(range(0, len(texts), batch_size), desc="groq embed"):
        batch = texts[i : i + batch_size]
        try:
            vecs = _groq_embed_batch(batch, client)
            # Normalise for cosine via inner product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs  = vecs / np.clip(norms, 1e-10, None)
            all_embeddings.append(vecs)
        except Exception as exc:
            logger.error(f"[API] Batch {i} failed: {exc}")
            raise

    embeddings = np.vstack(all_embeddings)
    elapsed    = time.perf_counter() - t0
    dim        = embeddings.shape[1]
    logger.info(
        f"[API] Embeddings shape: {embeddings.shape}, dim={dim}, took {elapsed:.1f}s"
    )

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"[API] FAISS index size: {index.ntotal} vectors")

    return index, chunks


def save_api_index(index: faiss.Index, metadata: list[dict]) -> None:
    """Serialise the API FAISS index and metadata sidecar."""
    faiss.write_index(index, str(MODELS_DIR / "api.faiss"))
    with open(MODELS_DIR / "api_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("[API] Index saved → models/api.faiss + api_meta.json")


def load_api_index() -> tuple[faiss.Index, list[dict]]:
    """Load a previously serialised API FAISS index."""
    index = faiss.read_index(str(MODELS_DIR / "api.faiss"))
    with open(MODELS_DIR / "api_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    logger.info(f"[API] Loaded index — {index.ntotal} vectors")
    return index, meta


# ── 3. HYBRID — BM25 + Dense via Reciprocal Rank Fusion ─────────────────────

def build_hybrid_index(
    chunks: list[dict],
    dense_index: faiss.Index,
    dense_meta: list[dict],
) -> tuple[Any, faiss.Index, list[dict]]:
    """
    Build a hybrid retriever combining BM25 sparse search with a dense FAISS
    index via Reciprocal Rank Fusion (RRF).

    RRF score = Σ_r  1 / (k + rank_r)   where k=60 is the RRF constant.

    Args:
        chunks:      Chunk list from Stage 1.
        dense_index: Pre-built FAISS dense index (local or API).
        dense_meta:  Metadata list aligned with dense_index.

    Returns:
        (bm25_model, dense_index, metadata_list)
    """
    from rank_bm25 import BM25Okapi

    texts     = [c["text"] for c in chunks]
    tokenised = [t.lower().split() for t in texts]

    logger.info(f"[HYBRID] Building BM25 index over {len(tokenised)} documents ...")
    bm25 = BM25Okapi(tokenised)
    logger.info("[HYBRID] BM25 index built.")

    return bm25, dense_index, dense_meta


def save_hybrid_index(bm25_model: Any) -> None:
    """Pickle the BM25 model to disk."""
    with open(MODELS_DIR / "hybrid_bm25.pkl", "wb") as f:
        pickle.dump(bm25_model, f)
    logger.info("[HYBRID] BM25 model saved → models/hybrid_bm25.pkl")


def load_hybrid_index() -> tuple[Any, faiss.Index, list[dict]]:
    """Load BM25 model and reuse the local FAISS index for the dense component."""
    with open(MODELS_DIR / "hybrid_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    dense_index, meta = load_local_index()
    logger.info("[HYBRID] Loaded BM25 + local dense index")
    return bm25, dense_index, meta


def reciprocal_rank_fusion(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    top_k: int = 5,
    k: int = 60,
) -> list[int]:
    """
    Combine BM25 and dense rankings via Reciprocal Rank Fusion.

    Args:
        bm25_scores:   BM25 scores for every document (shape: N).
        dense_scores:  Cosine similarity scores for every document (shape: N).
        top_k:         Number of final results to return.
        k:             RRF damping constant (default 60, per literature).

    Returns:
        List of top_k document indices sorted by fused score descending.
    """
    n = len(bm25_scores)

    bm25_ranks  = np.argsort(-bm25_scores)           # descending
    dense_ranks = np.argsort(-dense_scores)

    # Build rank lookup tables
    bm25_rank_of  = np.empty(n, dtype=int)
    dense_rank_of = np.empty(n, dtype=int)
    bm25_rank_of[bm25_ranks]   = np.arange(n)
    dense_rank_of[dense_ranks] = np.arange(n)

    rrf_scores = 1.0 / (k + bm25_rank_of) + 1.0 / (k + dense_rank_of)
    top_indices = np.argsort(-rrf_scores)[:top_k]
    return top_indices.tolist()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 — Embedding Generation & Indexing"
    )
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/chunks.json"),
        help="Path to chunks JSON from Stage 1 (default: data/chunks.json)",
    )
    parser.add_argument(
        "--strategy", choices=["local", "api", "hybrid", "all"],
        default="all",
        help="Embedding strategy to build (default: all)",
    )
    args = parser.parse_args()

    with open(args.chunks, encoding="utf-8") as fh:
        chunks = json.load(fh)
    logger.info(f"Loaded {len(chunks)} chunks from '{args.chunks}'")

    if args.strategy in ("local", "all"):
        idx, meta = build_local_index(chunks)
        save_local_index(idx, meta)

    if args.strategy in ("api", "all"):
        idx, meta = build_api_index(chunks)
        save_api_index(idx, meta)

    if args.strategy in ("hybrid", "all"):
        # Hybrid reuses the local dense index
        local_idx, local_meta = load_local_index()
        bm25_model, _, _ = build_hybrid_index(chunks, local_idx, local_meta)
        save_hybrid_index(bm25_model)

    print("\n✅  Stage 2 complete — indices saved to models/")
