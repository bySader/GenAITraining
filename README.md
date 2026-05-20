# GenAI Training — Upskilling Exercises

A hands-on training repository for learning **Generative AI** fundamentals through practical Python exercises. Each exercise is self-contained with its own setup, dataset, and instructions.

## Exercises

| # | Folder | Topic | Stack |
|---|--------|-------|-------|
| 01 | [Exercise_01](./Exercise_01/) | Text Summarization with Structured JSON Output | Python · Groq · LLaMA3 |
| 02 | [Exercise_02](./Exercise_02/) | Text Classification with Prompt Engineering | Python · Groq · LLaMA3 |
| 03 | [Exercise_03](./Exercise_03/) | RAG Q&A System — FAISS + Embeddings + Grounded Answers | Python · Groq · FAISS · sentence-transformers |

## Prerequisites (Global)
- **Python 3.10+**
- **Groq API key** — Get a free key at [console.groq.com](https://console.groq.com/)
- **VS Code** (recommended)

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/GenAITraining.git
   cd GenAITraining
   ```

2. Navigate to the exercise you want to work on:
   ```bash
   cd Exercise_02
   ```

3. Install its dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` → `.env` and fill in your API key, then run the script.

## Conventions
Each exercise folder contains:
- `README.md` — Exercise-specific instructions and usage
- `requirements.txt` — Python dependencies
- `.env.example` — Environment variable template (never commit `.env`)
- Main Python script(s)
- `dataset/` or sample data (when applicable)

## Adding a New Exercise
1. Create a new folder: `Exercise_03/`
2. Follow the same structure as the existing exercises
3. Add an entry to the table above

## License
This repository is intended for internal training purposes.
