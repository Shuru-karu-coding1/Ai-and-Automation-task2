"""
utils.py — Shared helpers, logging configuration, and text utilities.

Imported by all other pipeline stages.
"""

import os
import logging
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a standard formatter.

    Args:
        name:  Logger name (typically __name__ of the calling module).
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Embedding helpers ─────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mean_cosine_similarity(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
) -> float:
    """
    Mean cosine similarity between a query vector and multiple chunk vectors.

    Args:
        query_vec:  1-D query embedding.
        chunk_vecs: 2-D array of shape (N, D) — retrieved chunk embeddings.

    Returns:
        Mean cosine similarity score.
    """
    scores = [cosine_similarity(query_vec, v) for v in chunk_vecs]
    return float(np.mean(scores)) if scores else 0.0


# ── Token helpers ─────────────────────────────────────────────────────────────

def truncate_to_token_limit(text: str, max_tokens: int = 2000) -> str:
    """
    Truncate *text* to at most *max_tokens* tokens using tiktoken.

    Args:
        text:       Input text.
        max_tokens: Maximum token budget.

    Returns:
        Truncated text (decoded back to string).
    """
    import tiktoken
    enc    = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def build_context_window(chunks: list[dict], max_tokens: int = 2000) -> str:
    """
    Concatenate retrieved chunks into a single context string within budget.

    Args:
        chunks:     List of chunk dicts (must have 'text', 'source_file', 'page_num').
        max_tokens: Maximum token budget for the context window.

    Returns:
        Formatted context string with source citations.
    """
    import tiktoken
    enc    = tiktoken.get_encoding("cl100k_base")
    parts: list[str] = []
    used   = 0

    for chunk in chunks:
        header = f"[Source: {chunk['source_file']}, Page {chunk['page_num']}]\n"
        body   = chunk["text"].strip()
        block  = f"{header}{body}\n"
        block_tokens = len(enc.encode(block))

        if used + block_tokens > max_tokens:
            # Try to fit a truncated version
            remaining = max_tokens - used
            if remaining > 50:
                truncated = enc.decode(enc.encode(block)[:remaining])
                parts.append(truncated)
                used += remaining
            break
        parts.append(block)
        used += block_tokens

    return "\n---\n".join(parts)


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    """
    Simple context manager for measuring elapsed wall-clock time.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ms / 1000


# ── File helpers ──────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    """
    Load chunk list from a JSON file produced by Stage 1.

    Args:
        path: Path to chunks.json.

    Returns:
        List of chunk dicts.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: Path) -> None:
    """
    Serialise *obj* to JSON at *path*, creating parent directories as needed.

    Args:
        obj:  JSON-serialisable object.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    """Load and return a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Environment helpers ───────────────────────────────────────────────────────

def require_env(key: str) -> str:
    """
    Return the value of an environment variable, raising if missing.

    Args:
        key: Environment variable name.

    Returns:
        String value of the variable.

    Raises:
        EnvironmentError: If the variable is not set.
    """
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Add it to your .env file."
        )
    return val
