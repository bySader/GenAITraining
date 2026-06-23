"""
GEN AI Upskilling Training -- Exercise 04
Advanced RAG: Multi-Format Documents + Metadata + Ranking/Re-ranking

Key improvements over Exercise 03:
  - Multi-format document loading: .pdf, .docx, .txt, .json, .md
  - Custom metadata schema per chunk (document_name, author, topic, file_type, pages)
  - Two-stage retrieval: vector similarity (FAISS) + BM25 re-ranking (hybrid score)
  - Scope guard: rejects out-of-domain queries BEFORE hitting the index
  - Source attribution: LLM answer cites which documents were used
"""

import os
import json
import pickle
import re
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
load_dotenv()

DOCS_DIR      = Path(__file__).parent / "documents"
INDEX_DIR     = Path(__file__).parent / "faiss_index"
INDEX_PATH    = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "llama-3.1-8b-instant"
CHUNK_SIZE    = 400     # ~words per chunk
CHUNK_OVERLAP = 60      # words of overlap between consecutive chunks
TOP_K_VECTOR  = 10      # candidates from FAISS before re-ranking
TOP_K_FINAL   = 3       # final chunks after re-ranking
ALPHA         = 0.65    # weight for vector score in hybrid: alpha*vector + (1-alpha)*bm25

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────
# Known topics for scope guard
# ─────────────────────────────────────────────────────────────
KNOWN_TOPICS = {
    "Programming":          ["python", "decorator", "class", "function", "oop", "object", "inheritance",
                              "polymorphism", "api", "rest", "http", "endpoint", "code", "programming", "language"],
    "DevOps":               ["git", "docker", "container", "ci", "cd", "pipeline", "deploy", "branch",
                              "commit", "jenkins", "github", "devops", "kubernetes", "image", "workflow"],
    "Project Management":   ["agile", "scrum", "kanban", "sprint", "backlog", "standup", "retrospective",
                              "wip", "velocity", "burndown", "product owner", "scrum master", "iteration"],
    "Machine Learning":     ["machine learning", "neural network", "deep learning", "model", "training",
                              "overfitting", "backpropagation", "activation", "gradient", "regression",
                              "classification", "clustering", "feature", "dataset", "algorithm", "ml"],
    "Databases":            ["sql", "database", "nosql", "mongodb", "redis", "query", "table", "index",
                              "join", "normalization", "transaction", "schema", "primary key", "acid",
                              "postgresql", "relational", "document store"],
}

ALL_KEYWORDS = {kw for kws in KNOWN_TOPICS.values() for kw in kws}


# ─────────────────────────────────────────────────────────────
# STEP 1 -- Document Loaders (multi-format)
# ─────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> tuple[str, dict]:
    """Load text from a PDF file. Returns (text, metadata)."""
    metadata = {"file_type": "pdf", "author": "Unknown", "topic": "Unknown"}
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        full_text = "\n".join(pages)
        metadata["number_of_pages"] = len(reader.pages)
    except Exception:
        # Fallback: read as plain text (our text-fallback PDFs)
        full_text = path.read_text(encoding="utf-8", errors="ignore")
        metadata["number_of_pages"] = 1

    _extract_inline_metadata(full_text, metadata)
    return full_text, metadata


def load_docx(path: Path) -> tuple[str, dict]:
    """Load text from a Word document. Returns (text, metadata)."""
    from docx import Document
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    metadata = {"file_type": "docx", "number_of_pages": max(1, len(paragraphs) // 30)}
    _extract_inline_metadata(full_text, metadata)
    return full_text, metadata


def load_txt(path: Path) -> tuple[str, dict]:
    """Load a plain text file."""
    full_text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = {"file_type": "txt", "number_of_pages": max(1, len(full_text) // 3000)}
    _extract_inline_metadata(full_text, metadata)
    return full_text, metadata


def load_json(path: Path) -> tuple[str, dict]:
    """Load a JSON knowledge document. Flattens sections into readable text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "file_type": "json",
        "author":    data.get("author", "Unknown"),
        "topic":     data.get("topic", "Unknown"),
        "number_of_pages": 1,
    }
    parts = [data.get("title", ""), f"Author: {metadata['author']}", f"Topic: {metadata['topic']}", ""]
    for section in data.get("sections", []):
        parts.append(section.get("heading", ""))
        parts.append(section.get("content", ""))
        parts.append("")
    full_text = "\n".join(parts)
    return full_text, metadata


def load_markdown(path: Path) -> tuple[str, dict]:
    """Load a Markdown file."""
    full_text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = {"file_type": "md", "number_of_pages": max(1, len(full_text) // 3000)}
    _extract_inline_metadata(full_text, metadata)
    return full_text, metadata


def _extract_inline_metadata(text: str, metadata: dict) -> None:
    """Parse Author: and Topic: lines embedded in document text."""
    for line in text.splitlines()[:10]:
        line = line.strip().lstrip("#").strip()
        if line.lower().startswith("author:"):
            metadata["author"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("topic:"):
            metadata["topic"] = line.split(":", 1)[1].strip()


LOADERS = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".txt":  load_txt,
    ".json": load_json,
    ".md":   load_markdown,
}


def load_document(path: Path) -> tuple[str, dict]:
    """Dispatch to the correct loader based on file extension."""
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    text, meta = loader(path)
    meta["document_name"] = path.name
    return text, meta


# ─────────────────────────────────────────────────────────────
# STEP 1b -- Text Chunking
# ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping word-based chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 2 -- Build / Load FAISS Index with Rich Metadata
# ─────────────────────────────────────────────────────────────

def build_index(docs_dir: Path, model: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
    """
    Loads all documents, chunks them, embeds chunks, and builds a FAISS index.

    Each chunk stored as metadata dict:
    {
        "document_name": "python_intro.md",
        "file_type":     "md",
        "author":        "Jane Smith",
        "topic":         "Programming",
        "number_of_pages": 2,
        "chunk_index":   0,
        "total_chunks":  5,
        "content":       "..."
    }
    """
    all_chunks = []

    doc_files = sorted([
        p for p in docs_dir.iterdir()
        if p.suffix.lower() in LOADERS and p.is_file()
    ])

    print(f"  Loading {len(doc_files)} documents...")
    for doc_path in doc_files:
        try:
            text, meta = load_document(doc_path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    **meta,
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "content":      chunk,
                }
                all_chunks.append(chunk_meta)
            print(f"    [OK] {doc_path.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"    [ERROR] {doc_path.name}: {e}")

    print(f"\n  Embedding {len(all_chunks)} total chunks...")
    texts = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, all_chunks


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  Index saved to {INDEX_DIR}/")


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"  Index loaded: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


def get_or_build_index(force_rebuild: bool = False) -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    model = SentenceTransformer(EMBED_MODEL)
    if not force_rebuild and INDEX_PATH.exists() and METADATA_PATH.exists():
        print("[INFO] Loading existing FAISS index from disk...")
        index, chunks = load_index()
    else:
        print("[INFO] Building FAISS index from documents...")
        index, chunks = build_index(DOCS_DIR, model)
        save_index(index, chunks)
    return index, chunks, model


# ─────────────────────────────────────────────────────────────
# STEP 2b -- Scope Guard
# ─────────────────────────────────────────────────────────────

def is_in_scope(query: str) -> bool:
    """
    Fast keyword-based scope check BEFORE hitting the vector index.
    Returns True if the query likely relates to known topics.
    This prevents wasted API calls and irrelevant retrievals.
    """
    q_lower = query.lower()
    for keyword in ALL_KEYWORDS:
        if keyword in q_lower:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# STEP 2c -- Retrieval + Re-ranking
# ─────────────────────────────────────────────────────────────

def retrieve_and_rerank(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k_vector: int = TOP_K_VECTOR,
    top_k_final: int = TOP_K_FINAL,
    alpha: float = ALPHA,
) -> list[dict]:
    """
    Two-stage retrieval with hybrid re-ranking:

    Stage 1 -- Vector Similarity Search (FAISS):
      Embed the query, search the FAISS index, retrieve top_k_vector candidates.

    Stage 2 -- BM25 Re-ranking:
      Apply BM25 keyword matching on the candidate pool.
      Compute hybrid score = alpha * cosine_score + (1-alpha) * bm25_normalized_score.
      Return top_k_final chunks by hybrid score.

    WHY RE-RANK?
      Vector search captures semantic similarity but can miss exact keyword matches.
      BM25 excels at keyword matching but misses synonyms and paraphrases.
      Hybrid combines both strengths.
    """
    # --- Stage 1: FAISS vector search ---
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k_vector)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({**chunks[idx], "vector_score": float(score)})

    if not candidates:
        return []

    # --- Stage 2: BM25 re-ranking on candidate pool ---
    tokenize = lambda text: re.findall(r"\w+", text.lower())
    corpus_tokens = [tokenize(c["content"]) for c in candidates]
    query_tokens  = tokenize(query)

    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize BM25 scores to [0, 1]
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_norm = [s / bm25_max for s in bm25_scores]

    # Compute hybrid score
    for i, candidate in enumerate(candidates):
        candidate["bm25_score"]    = float(bm25_scores[i])
        candidate["bm25_norm"]     = bm25_norm[i]
        candidate["hybrid_score"]  = alpha * candidate["vector_score"] + (1 - alpha) * bm25_norm[i]

    # Sort by hybrid score descending
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return candidates[:top_k_final]


# ─────────────────────────────────────────────────────────────
# STEP 3 -- Grounded LLM Call with Source Attribution
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a precise technical knowledge assistant.

You MUST answer questions ONLY based on the context provided below.
Rules:
- If the context contains enough information, give a clear, accurate, and complete answer.
- If the context does NOT contain enough information, respond exactly with:
  "I don't know, that is not part of my context."
- Do NOT use any prior knowledge outside the provided context.
- Do NOT hallucinate facts, file names, or details not in the context.
- Keep your answer focused and well-structured.

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END OF CONTEXT ---
""".strip()


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block with source citations."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[Source {i}: {chunk['document_name']} | Topic: {chunk['topic']} | Author: {chunk.get('author','Unknown')}]"
        blocks.append(f"{source}\n{chunk['content']}")
    return "\n\n".join(blocks)


def ask_rag(
    question: str,
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline:
      0. Scope guard -- reject out-of-domain queries immediately
      1. Retrieve top-10 candidates via FAISS
      2. Re-rank with BM25 hybrid scoring
      3. Build grounded prompt with top-3 chunks
      4. Call Groq LLM
      Returns dict with 'answer', 'sources', 'in_scope' keys.
    """
    # Step 0: Scope guard
    if not is_in_scope(question):
        return {
            "answer":   "I'm sorry, that question is outside the scope of my knowledge base. "
                        "I can only answer questions about Programming, DevOps, Project Management, "
                        "Machine Learning, and Databases.",
            "sources":  [],
            "in_scope": False,
        }

    # Step 1 + 2: Retrieve and re-rank
    retrieved = retrieve_and_rerank(question, index, chunks, model)

    if not retrieved:
        return {
            "answer":   "I don't know, that is not part of my context.",
            "sources":  [],
            "in_scope": True,
        }

    if verbose:
        print("\n[RETRIEVAL] Re-ranked results:")
        for r in retrieved:
            print(f"  doc={r['document_name']}  chunk={r['chunk_index']}  "
                  f"vector={r['vector_score']:.3f}  bm25={r['bm25_norm']:.3f}  "
                  f"hybrid={r['hybrid_score']:.3f}")

    # Step 3: Build grounded prompt
    context_block = build_context_block(retrieved)
    system_prompt = SYSTEM_PROMPT.format(context=context_block)

    # Step 4: LLM call
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
    )

    sources = list({r["document_name"] for r in retrieved})
    return {
        "answer":   response.choices[0].message.content.strip(),
        "sources":  sources,
        "in_scope": True,
    }


# ─────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  GEN AI Upskilling -- Exercise 04: Advanced RAG Q&A")
    print("=" * 65)
    print("  Topics: Programming | DevOps | Project Mgmt | ML | Databases")
    print("=" * 65)

    index, chunks, embed_model = get_or_build_index()

    docs = len({c["document_name"] for c in chunks})
    print(f"\n[OK] Ready! {docs} documents | {len(chunks)} chunks indexed.")
    print("     Commands: 'verbose on/off' | 'rebuild' | 'quit'\n")

    verbose = False

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif question.lower() == "verbose on":
            verbose = True
            print("  Verbose mode ON.")
            continue
        elif question.lower() == "verbose off":
            verbose = False
            print("  Verbose mode OFF.")
            continue
        elif question.lower() == "rebuild":
            index, chunks, embed_model = get_or_build_index(force_rebuild=True)
            continue

        print("\n[...] Thinking...\n")
        result = ask_rag(question, index, chunks, embed_model, verbose=verbose)

        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print(f"\nAnswer:\n{result['answer']}\n")
        print("-" * 65)


if __name__ == "__main__":
    main()
