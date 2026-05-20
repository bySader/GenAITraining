# Exercise 03 — RAG Q&A System

## Overview
This exercise implements a **Retrieval-Augmented Generation (RAG)** system that answers IT support questions based **only** on a provided knowledge base — no hallucinations allowed.

## Key Concepts

### What is RAG and why is it necessary?
RAG (Retrieval-Augmented Generation) grounds an LLM's responses in a specific document corpus rather than relying on its pre-trained weights. This is necessary to:
- **Eliminate hallucinations** — the model can only answer from retrieved facts
- **Keep knowledge fresh** — update the corpus without retraining the model
- **Scope the domain** — prevent the model from answering outside its mandate

### Similarity Search vs. Vector Search
| Term | Meaning |
|------|---------|
| **Vector Search** | Finding nearest neighbors by comparing dense embedding vectors (e.g., FAISS ANN search) |
| **Similarity Search** | A broader term covering any search that measures semantic or geometric closeness — vector search is one implementation. Others include BM25 (term frequency) or hybrid (BM25 + vector) |

In this exercise we use **cosine similarity** via FAISS `IndexFlatIP` on L2-normalised vectors — a standard and effective vector search approach.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  Embedding Model                │
│  (all-MiniLM-L6-v2, local)      │   ← No API cost, runs offline
└──────────────┬──────────────────┘
               │  query vector (384-dim)
               ▼
┌─────────────────────────────────┐
│  FAISS Index (cosine similarity)│   ← 100 IT knowledge items
│  IndexFlatIP on unit vectors    │
└──────────────┬──────────────────┘
               │  top-3 chunks + scores
               ▼
┌─────────────────────────────────┐
│  Grounded Prompt Builder        │   ← Injects retrieved text as context
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Groq LLM (llama-3.1-8b-instant)│   ← temperature=0.1 for determinism
└──────────────┬──────────────────┘
               │
               ▼
         Grounded Answer
```

## Setup

### 1. Prerequisites
- Python 3.10+
- A [Groq](https://console.groq.com/) API key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run — this is cached locally.

### 3. Configure environment
```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run the interactive Q&A
```bash
python rag_engine.py
```

On first run it will **build and save the FAISS index** (`faiss_index/`). Subsequent runs load it instantly.

**Available commands inside the prompt:**
| Command | Action |
|---------|--------|
| `verbose on` | Show retrieved chunks and similarity scores |
| `verbose off` | Hide debug info |
| `rebuild` | Force rebuild of the FAISS index |
| `quit` / `exit` | Exit |

### 5. Run the evaluation suite
```bash
python evaluate.py
```
Runs 12 test cases (8 in-scope + 4 out-of-scope) and saves `test_results.json`.

## File Structure
```
Exercise_03/
├── rag_engine.py          ← Main RAG script (interactive Q&A)
├── evaluate.py            ← Automated test suite (12 Q&A pairs)
├── requirements.txt
├── .env.example
├── README.md              ← This file
├── archive/
│   └── synthetic_knowledge_items.csv   ← 100 IT knowledge items
└── faiss_index/           ← Generated on first run
    ├── index.faiss
    └── metadata.pkl
```

## Example Interaction
```
❓ Your question: How do I reset a jammed printer?
💬 Answer:
To reset a jammed printer, follow these steps: ...

❓ Your question: What is the weather in Mexico City?
💬 Answer:
I don't know, that is not part of my context.
```
