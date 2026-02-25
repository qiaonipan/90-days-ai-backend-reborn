"""
Ingest knowledge base Markdown files into a local vector store.

Steps:
- Load all Markdown files from knowledge_base/
- Split into ~500-character chunks with 50-character overlap
- Embed with a local sentence-transformers model
- Persist embeddings and metadata under vector_db/ (optionally with a FAISS index)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent
KB_DIR = ROOT_DIR / "knowledge_base"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


class RecursiveCharacterTextSplitter:
    """Simple character-based splitter with fixed-size chunks and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        n = len(text)
        step = self.chunk_size - self.chunk_overlap
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = text[start:end]
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start += step
        return chunks


def load_markdown_files(kb_dir: Path) -> List[Tuple[Path, str]]:
    """Load all Markdown files (UTF-8) from the knowledge base directory."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    docs: List[Tuple[Path, str]] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append((path, text))
    return docs


def build_chunks() -> Tuple[List[Chunk], int]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = load_markdown_files(KB_DIR)

    chunks: List[Chunk] = []
    for path, text in docs:
        pieces = splitter.split_text(text)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source=str(path.relative_to(ROOT_DIR)),
                    chunk_index=idx,
                )
            )
    return chunks, len(docs)


def embed_chunks(chunks: List[Chunk]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 0), dtype="float32")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def persist_vector_db(chunks: List[Chunk], embeddings: np.ndarray) -> None:
    VECTOR_DB_DIR.mkdir(exist_ok=True)

    # Save embeddings
    emb_path = VECTOR_DB_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)

    # Save metadata (JSONL)
    meta_path = VECTOR_DB_DIR / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for idx, c in enumerate(chunks):
            rec = {
                "id": idx,
                "source": c.source,
                "chunk_index": c.chunk_index,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Optional: build FAISS index if library is available
    try:
        import faiss  # type: ignore

        if embeddings.size == 0:
            return
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss_path = VECTOR_DB_DIR / "faiss.index"
        faiss.write_index(index, str(faiss_path))
        print(f"FAISS index written to: {faiss_path}")
    except ImportError:
        print("faiss not installed; saved raw embeddings and metadata only.")


def main() -> None:
    print(f"Loading knowledge base from: {KB_DIR}")
    chunks, num_docs = build_chunks()
    if not chunks:
        print("No chunks generated; nothing to ingest.")
        return

    print(f"Loaded {num_docs} Markdown files. Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)
    persist_vector_db(chunks, embeddings)

    processed_files = sorted({c.source for c in chunks})
    print("\nIngestion completed.")
    print("Processed files:")
    for src in processed_files:
        print(f"  - {src}")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error during ingestion: {exc}", file=sys.stderr)
        raise

