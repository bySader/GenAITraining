# GenAI Training — Upskilling Exercises

A hands-on training repository for learning **Generative AI** fundamentals through practical Python exercises. Each exercise is self-contained with its own setup, dataset, and instructions.

## Exercises

| # | Folder | Topic | Stack |
|---|--------|-------|-------|
| 01 | [Exercise_01](./Exercise_01/) | Text Summarization with Structured JSON Output | Python · Groq · LLaMA3 |
| 02 | [Exercise_02](./Exercise_02/) | Text Classification with Prompt Engineering | Python · Groq · LLaMA3 |
| 03 | [Exercise_03](./Exercise_03/) | RAG Q&A System — FAISS + Embeddings + Grounded Answers | Python · Groq · FAISS · sentence-transformers |
| 04 | [Exercise_04](./Exercise_04/) | Advanced RAG — Multi-Format Docs + Metadata + Re-ranking | Python · Groq · FAISS · BM25 · pypdf · python-docx |
| 05 | [Exercise_05](./Exercise_05/) | Tool Calling Agent — 7 Tools + Multi-Tool + Agentic Loop | Python · Groq · Function Calling |
| 06 | [Exercise_06](./Exercise_06/) | LangGraph NL-to-SQL Workflow — Router, SQL Guard, RAG Node | Python · LangGraph · LangChain · Groq · SQLite |

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
   cd Exercise_XX
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
1. Create a new folder: `Exercise_XX/`
2. Follow the same structure as the existing exercises
3. Add an entry to the table above

## Interface gráfica de ejercicios

También puedes abrir una interfaz visual para explorar todo el repositorio y ejecutar cada ejercicio individualmente desde un navegador.

### Ejecutar la dashboard

```bash
pip install -r requirements.txt
python app.py
```

> La instalación raíz incluye las dependencias principales de todos los ejercicios para que la dashboard pueda lanzar cada script sin errores de `ModuleNotFoundError`.

Luego abre:

```text
http://localhost:5000
```

La interfaz incluye:
- navegación por todos los ejercicios del repositorio
- descripción resumida de cada ejercicio
- acceso a su README y scripts disponibles
- ejecución individual de cada script con salida en tiempo real
- soporte para entrada interactiva de prompts: si un ejercicio solicita respuestas en consola, escribe cada línea en el campo de entrada de la dashboard antes de ejecutar.
- diseño responsive para escritorio y mobile

## License
This repository is intended for internal training purposes.
