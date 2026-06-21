"""
Stage 1: Document Ingestion & Text Segmentation
------------------------------------------------
Loads multi-page PDFs, extracts clean text, strips layout noise,
detects section types, chunks with tiktoken, and saves JSON metadata.

Run standalone:
    python src/preprocess.py --data-dir data/ --output data/chunks.json
"""

import os
import re
import json
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import fitz          # PyMuPDF
import tiktoken

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess")

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens per chunk
CHUNK_OVERLAP = 64    # overlapping tokens between adjacent chunks
ENCODING_NAME = "cl100k_base"  # tiktoken encoding (GPT-4 / text-embedding-3)

# Section-type keyword patterns — first match wins
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"\b(standard operating procedure|SOP|work instruction)\b", "SOP"),
    (r"\b(policy|policies|corporate policy|framework)\b",        "POLICY"),
    (r"\b(compliance|regulation|GDPR|ISO|audit|legal requirement)\b", "COMPLIANCE"),
    (r"\b(troubleshoot|error log|incident|diagnostic|debug|fault)\b", "TROUBLESHOOTING"),
]

# Regex patterns for boilerplate noise to strip
NOISE_PATTERNS: list[str] = [
    r"Page\s+\d+\s*(of\s+\d+)?",   # page numbers
    r"(confidential|internal use only)",
    r"^\s*\d+\s*$",                  # lone digit lines
    r"[-=]{5,}",                     # horizontal rules
    r"©.*?\d{4}",                    # copyright notices
    r"www\.[^\s]+",                  # bare URLs
    r"<[^>]+>",                      # stray HTML tags
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_tokenizer() -> tiktoken.Encoding:
    """Return the shared tiktoken encoding instance."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    """Return the number of tokens in *text*."""
    return len(enc.encode(text))


def clean_text(raw: str) -> str:
    """
    Remove layout noise from raw PDF-extracted text.

    Applies boilerplate regex patterns, collapses excess blank lines,
    and normalises whitespace.

    Args:
        raw: Raw text string from PyMuPDF.

    Returns:
        Cleaned text string.
    """
    text = raw
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def detect_section_type(text: str) -> str:
    """
    Classify text block into a document section type.

    Args:
        text: Cleaned text block.

    Returns:
        One of: SOP | POLICY | COMPLIANCE | TROUBLESHOOTING | GENERAL
    """
    for pattern, label in SECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "GENERAL"


def chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping token-bounded chunks.

    Rationale for defaults:
      - 512 tokens: fits within most embedding model context windows
        while preserving meaningful semantic units.
      - 64 tokens (~12.5% overlap): ensures cross-chunk context is
        preserved at boundaries without excessive redundancy.

    Args:
        text:       Input text to chunk.
        enc:        Tiktoken encoding for tokenisation.
        chunk_size: Max tokens per chunk (default 512).
        overlap:    Token overlap between consecutive chunks (default 64).

    Returns:
        List of decoded text strings.
    """
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def make_chunk_id(source_file: str, page_num: int, idx: int) -> str:
    """
    Generate a short deterministic ID for a chunk.

    Args:
        source_file: Filename of the source PDF.
        page_num:    Page number within the PDF.
        idx:         Global chunk index.

    Returns:
        12-character hex digest string.
    """
    raw = f"{source_file}::{page_num}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Open a PDF and return (page_number, raw_text) for each page.

    Args:
        pdf_path: Absolute or relative path to a PDF file.

    Returns:
        List of (1-indexed page number, raw text) tuples.

    Raises:
        Exception: Propagated from PyMuPDF on corrupt or missing files.
    """
    pages: list[tuple[int, str]] = []
    try:
        doc = fitz.open(str(pdf_path))
        logger.info(f"Opened '{pdf_path.name}' — {len(doc)} page(s)")
        for page_num, page in enumerate(doc, start=1):
            pages.append((page_num, page.get_text("text")))
        doc.close()
    except Exception as exc:
        logger.error(f"Cannot open '{pdf_path}': {exc}")
        raise
    return pages


def process_pdf(pdf_path: Path, enc: tiktoken.Encoding) -> list[dict]:
    """
    Full ingestion pipeline for a single PDF.

    For each page: extract → clean → detect section → chunk → attach metadata.

    Args:
        pdf_path: Path to the PDF file.
        enc:      Tiktoken encoding instance.

    Returns:
        List of chunk dicts with keys:
        {chunk_id, text, source_file, page_num, section_type, token_count}
    """
    pages = extract_pdf_pages(pdf_path)
    source_file = pdf_path.name
    all_chunks: list[dict] = []
    global_idx = 0

    for page_num, raw_text in pages:
        clean = clean_text(raw_text)
        if not clean:
            logger.debug(f"  Page {page_num}: empty after cleaning, skipping.")
            continue

        section_type = detect_section_type(clean)
        page_chunks  = chunk_text(clean, enc)

        for chunk_str in page_chunks:
            all_chunks.append(
                {
                    "chunk_id":     make_chunk_id(source_file, page_num, global_idx),
                    "text":         chunk_str,
                    "source_file":  source_file,
                    "page_num":     page_num,
                    "section_type": section_type,
                    "token_count":  count_tokens(chunk_str, enc),
                }
            )
            global_idx += 1

        logger.info(
            f"  Page {page_num}: section={section_type}, "
            f"chunks={len(page_chunks)}"
        )

    logger.info(f"'{source_file}' → {len(all_chunks)} chunk(s)")
    return all_chunks


def ingest_directory(
    data_dir: Path,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """
    Ingest every PDF in *data_dir* recursively.

    Args:
        data_dir:    Root directory containing source PDFs.
        output_path: If provided, write combined chunk list as JSON.

    Returns:
        Combined list of chunk dicts from all PDFs.
    """
    enc       = get_tokenizer()
    pdf_files = sorted(data_dir.glob("**/*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in '{data_dir}'.")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in '{data_dir}'")
    all_chunks: list[dict] = []

    for pdf_path in pdf_files:
        try:
            all_chunks.extend(process_pdf(pdf_path, enc))
        except Exception as exc:
            logger.error(f"Skipping '{pdf_path.name}': {exc}")

    logger.info(f"Total chunks: {len(all_chunks)}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
        logger.info(f"Saved chunks → '{output_path}'")

    return all_chunks


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 — Document Ingestion & Text Segmentation"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory with source PDFs (default: data/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/chunks.json"),
        help="Output JSON path (default: data/chunks.json)",
    )
    args = parser.parse_args()

    chunks = ingest_directory(args.data_dir, args.output)
    print(f"\n✅  Stage 1 complete — {len(chunks)} chunks → '{args.output}'")
