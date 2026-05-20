"""
GEN AI Upskilling Training — Exercise 03
Retrieval-Augmented Generation (RAG) with FAISS + Groq

Architecture:
  1. Load CSV knowledge base (synthetic_knowledge_items.csv)
  2. Embed each knowledge item with sentence-transformers (local, no API cost)
  3. Build a FAISS index for fast similarity search
  4. At query time: embed the question → retrieve top-k chunks → inject as context
  5. Send grounded prompt to Groq LLM → hallucination-free answer
"""

import os
import csv
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
load_dotenv()

CSV_PATH      = Path(__file__).parent / "archive" / "synthetic_knowledge_items.csv"
INDEX_PATH    = Path(__file__).parent / "faiss_index" / "index.faiss"
METADATA_PATH = Path(__file__).parent / "faiss_index" / "metadata.pkl"

EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, lightweight, 384-dim embeddings
LLM_MODEL     = "llama-3.1-8b-instant"
TOP_K         = 3                     # number of chunks to retrieve per query

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load knowledge base from CSV
# ─────────────────────────────────────────────────────────────

def load_knowledge_base(csv_path: Path) -> list[dict]:
    """
    Reads the CSV and returns a list of knowledge items.
    We index ki_topic + ki_text (the authoritative, well-written version).
    alt_ki_text and bad_ki_text are intentionally excluded to keep quality high.
    """
    items = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row["ki_topic"].strip()
            text  = row["ki_text"].strip()
            # Combine topic as a heading so the chunk is self-contained
            combined = f"Topic: {topic}\n\n{text}"
            items.append({
                "topic":    topic,
                "text":     text,
                "combined": combined,
            })
    return items


# ─────────────────────────────────────────────────────────────
# STEP 2 — Build (or load) FAISS index
# ─────────────────────────────────────────────────────────────

def build_index(items: list[dict], model: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
    """
    Embeds all knowledge items and builds a FAISS L2 index.
    Returns (index, items) — items are stored as metadata alongside the index.
    """
    print(f"  Embedding {len(items)} knowledge items with '{EMBED_MODEL}'...")
    texts = [item["combined"] for item in items]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalise → cosine similarity becomes equivalent to L2 on unit vectors
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product on unit vectors = cosine similarity
    index.add(embeddings)
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return index, items


def save_index(index: faiss.Index, items: list[dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(items, f)
    print(f"  Index saved to {INDEX_PATH}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        items = pickle.load(f)
    print(f"  Index loaded: {index.ntotal} vectors")
    return index, items


def get_or_build_index(force_rebuild: bool = False) -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    """
    Returns (index, metadata, embedding_model).
    Builds from scratch on first run; loads from disk on subsequent runs.
    """
    model = SentenceTransformer(EMBED_MODEL)

    if not force_rebuild and INDEX_PATH.exists() and METADATA_PATH.exists():
        print("\n[INFO] Loading existing FAISS index from disk...")
        index, items = load_index()
    else:
        print("\n[INFO] Building FAISS index from CSV...")
        items = load_knowledge_base(CSV_PATH)
        index, items = build_index(items, model)
        save_index(index, items)

    return index, items, model


# ─────────────────────────────────────────────────────────────
# STEP 2b — Query / Retrieval function
# ─────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    index: faiss.Index,
    items: list[dict],
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embeds the query string, performs similarity search in the FAISS index,
    and returns the top_k most relevant knowledge items.

    Args:
        query:  The user's question (string)
        index:  FAISS index
        items:  Metadata list aligned with the index
        model:  SentenceTransformer embedding model
        top_k:  Number of chunks to return

    Returns:
        List of dicts with keys: topic, text, score
    """
    # Embed query
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # Vector similarity search
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "topic": items[idx]["topic"],
            "text":  items[idx]["text"],
            "score": float(score),
        })

    return results


# ─────────────────────────────────────────────────────────────
# STEP 3 — Grounded prompt + LLM call
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful IT Knowledge Base assistant.

You MUST answer questions ONLY based on the context provided below.
- If the context contains enough information to answer the question, give a clear, accurate answer.
- If the context does NOT contain enough information, respond exactly with: "I don't know, that is not part of my context."
- Do NOT use any external knowledge. Do NOT hallucinate facts.
- Do NOT mention that you are using a context or knowledge base — just answer naturally.

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END OF CONTEXT ---
""".strip()


def build_context_block(retrieved: list[dict]) -> str:
    """Formats retrieved chunks into a readable context block."""
    blocks = []
    for i, item in enumerate(retrieved, 1):
        blocks.append(f"[{i}] Topic: {item['topic']}\n{item['text']}")
    return "\n\n".join(blocks)


def ask_rag(
    question: str,
    index: faiss.Index,
    items: list[dict],
    model: SentenceTransformer,
    top_k: int = TOP_K,
    verbose: bool = False,
) -> str:
    """
    Full RAG pipeline:
      1. Retrieve relevant context from FAISS
      2. Build grounded prompt
      3. Call Groq LLM
      4. Return answer string

    Args:
        question: User's question
        verbose:  If True, print retrieved chunks and similarity scores
    """
    # --- Retrieve ---
    retrieved = retrieve_context(question, index, items, model, top_k=top_k)

    if verbose:
        print("\n[CONTEXT] Retrieved chunks:")
        for r in retrieved:
            print(f"  [{r['score']:.3f}] {r['topic']}")

    # --- Build prompt ---
    context_block = build_context_block(retrieved)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_block)

    # --- LLM call ---
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,     # low → deterministic, grounded answers
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  GEN AI Upskilling — Exercise 03: RAG Q&A System")
    print("=" * 60)
    print("\nBuilding / loading knowledge index...")

    index, items, embed_model = get_or_build_index()

    print(f"\n[OK] Ready! Knowledge base: {len(items)} IT topics indexed.")
    print("   Type your question and press Enter.")
    print("   Commands: 'verbose on/off' | 'rebuild' | 'quit'\n")

    verbose = False

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        # Special commands
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif question.lower() == "verbose on":
            verbose = True
            print("  Verbose mode ON — retrieved chunks will be shown.")
            continue
        elif question.lower() == "verbose off":
            verbose = False
            print("  Verbose mode OFF.")
            continue
        elif question.lower() == "rebuild":
            index, items, embed_model = get_or_build_index(force_rebuild=True)
            print(f"  ✅ Index rebuilt: {len(items)} items.")
            continue

        # RAG answer
        print("\n[...] Thinking...\n")
        answer = ask_rag(question, index, items, embed_model, verbose=verbose)
        print(f"Answer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
