from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from html import escape
from pathlib import Path
from typing import TextIO, TypedDict, cast

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
class ProcessMeta(TypedDict):
  process: subprocess.Popen[str]
  log_file: TextIO
  log_path: str
  exercise_id: str
  script_name: str


PROCESSES: dict[str, ProcessMeta] = {}


def load_global_api_key() -> str:
    env_candidates = [
        ROOT / ".env",
        ROOT / "Exercise_06" / ".env",
        ROOT / "Exercise_06" / ".env.local",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)

    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    return groq_key


GLOBAL_GROQ_API_KEY = load_global_api_key()

EXERCISE_OVERRIDES = {
    "Exercise_01": {
        "title": "Exercise 01 — Text Summarization",
        "description": "Genera resúmenes estructurados, puntos clave y un análisis de sentimiento con prompts de diferentes niveles de complejidad usando Groq.",
        "stack": ["Python", "Groq", "LLaMA3"],
        "main_script": "summarizer.py",
    },
    "Exercise_02": {
        "title": "Exercise 02 — Text Classifier",
        "description": "Clasifica documentos cortos en categorías jerárquicas con validación JSON estricta, comparando técnicas de prompting y un pipeline de dos llamadas.",
        "stack": ["Python", "Groq", "LLM", "JSON"],
        "main_script": "classifier.py",
    },
    "Exercise_03": {
        "title": "Exercise 03 — RAG Q&A",
        "description": "Construye un sistema RAG con embeddings, búsqueda vectorial FAISS y respuestas ancladas en un conjunto de conocimiento específico para evitar alucinaciones.",
        "stack": ["Python", "FAISS", "Embeddings", "RAG"],
        "main_script": "rag_engine.py",
    },
    "Exercise_04": {
        "title": "Exercise 04 — Advanced RAG",
        "description": "Extiende el patrón RAG con documentos multimodal, metadata, chunks, ranking y re-ranking para responder con contexto relevante y grounded.",
        "stack": ["Python", "RAG", "BM25", "pypdf", "docx"],
        "main_script": "rag_engine.py",
    },
    "Exercise_05": {
        "title": "Exercise 05 — Tool Calling Agent",
        "description": "Crea un asistente con invocación de herramientas, decisiones de ejecución y respuesta final basada en datos reales del entorno y herramientas del sistema.",
        "stack": ["Python", "Groq", "Function Calling", "Agent"],
        "main_script": "agent.py",
    },
    "Exercise_06": {
        "title": "Exercise 06 — LangGraph SQL Agent",
        "description": "Interpreta preguntas en lenguaje natural, decide si requieren SQL o RAG, y responde con trazabilidad usando un flujo de LangGraph sobre SQLite.",
        "stack": ["Python", "LangGraph", "SQLite", "RAG"],
        "main_script": "workflow.py",
    },
}


def find_exercise_dir(exercise_id: str) -> Path | None:
    target = exercise_id.lower()
    for path in sorted(ROOT.glob("Exercise_*")):
        if path.is_dir() and path.name.lower() == target:
            return path
    return None


def find_readme(folder: Path) -> Path | None:
    candidates = [folder / "README.md", *sorted(folder.glob("*.md"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def derive_summary(folder: Path, fallback: str) -> str:
    readme = find_readme(folder)
    if not readme:
        return fallback

    text = readme.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("## overview") or stripped.lower().startswith("### objective") or stripped.lower().startswith("## exercise description"):
            result: list[str] = []
            for next_line in lines[index + 1 :]:
                if next_line.startswith("##") or next_line.startswith("#"):
                    break
                clean = next_line.strip()
                if clean:
                    result.append(clean)
            if result:
                return " ".join(result[:3])
    return fallback


def build_exercise_record(folder: Path) -> dict[str, object]:
    folder_name = folder.name
    override = EXERCISE_OVERRIDES.get(folder_name, {})
    title = override.get("title") or folder_name
    description = override.get("description") or derive_summary(folder, "Ejercicio de entrenamiento para explorar IA generativa con prompts y flujo de trabajo práctico.")
    stack = override.get("stack") or ["Python", "Generative AI"]

    python_files = sorted(path for path in folder.glob("*.py") if path.is_file())
    main_script = override.get("main_script")
    if not main_script:
        candidates = [p.name for p in python_files if p.name.lower() not in {"evaluate.py", "setup_db.py"}]
        main_script = candidates[0] if candidates else (python_files[0].name if python_files else "")

    scripts: list[dict[str, str]] = []
    for py_file in python_files:
        kind = "main" if py_file.name == main_script else "support"
        scripts.append({"name": py_file.name, "kind": kind})

    readme = find_readme(folder)
    status = "listo" if (folder / "requirements.txt").exists() else "configuración"

    return {
        "id": folder_name.lower(),
        "name": folder_name,
        "title": title,
        "description": description,
        "stack": stack,
        "scripts": scripts,
        "main_script": main_script,
        "readme": str(readme.relative_to(ROOT)) if readme else "",
        "status": status,
    }


def list_exercises() -> list[dict[str, object]]:
    return [
        build_exercise_record(folder)
        for folder in sorted(ROOT.glob("Exercise_*"))
        if folder.is_dir()
    ]


def render_exercise_navigation(exercises: list[dict[str, object]]) -> str:
    if not exercises:
        return '<p class="description-text">No hay ejercicios disponibles en el repositorio.</p>'

    cards = []
    for exercise in exercises:
        exercise_id = escape(str(exercise["id"]), quote=True)
        name = escape(str(exercise["name"]).replace("Exercise_", "Ex."))
        title = escape(str(exercise["title"]))
        description = escape(str(exercise["description"]))
        cards.append(
            f'<button class="exercise-button" type="button" data-id="{exercise_id}">'
            f'<span class="exercise-chip">{name}</span>'
            f'<h3>{title}</h3>'
            f'<p>{description}</p>'
            "</button>"
        )
    return "".join(cards)


def list_allowed_scripts(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.py") if p.is_file())


EXERCISE_06_REQUIRED_COLUMNS = {
    "tickets": {"ticket_id", "title", "priority", "status", "owner", "owner_id", "project_id", "created_date"},
    "employees": {"employee_id", "name", "department", "hire_date", "project", "skills"},
    "projects": {"project_id", "name", "duration", "team", "termination_date", "has_open_tickets", "open_tickets"},
    "clients": {"client_id", "client_name", "industry", "country", "account_manager", "active_projects"},
    "skill_certifications": {"certification_id", "employee_id", "certification_name", "provider", "issue_date", "expiration_date"},
}


def exercise06_prerequisite_error() -> str | None:
    if not (os.environ.get("GROQ_API_KEY") or GLOBAL_GROQ_API_KEY):
        return "No se encontró GROQ_API_KEY. Configúrala en .env o como variable de entorno y reinicia el dashboard."

    missing_modules = [
        module
        for module in ("langchain_groq", "langchain_core", "langgraph", "pydantic", "dotenv")
        if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        return (
            "Faltan dependencias de Exercise 06: "
            f"{', '.join(missing_modules)}. Instálalas con el requirements.txt de Exercise_06."
        )

    database_path = ROOT / "Exercise_06" / "company.db"
    if not database_path.is_file():
        return (
            "No existe Exercise_06/company.db. setup_db.py puede crearla, pero "
            "elimina y recrea las tablas; ejecútalo solo si confirmas que no perderás datos."
        )

    try:
        connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
        try:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            missing_tables = EXERCISE_06_REQUIRED_COLUMNS.keys() - existing_tables
            if missing_tables:
                return (
                    f"La base de datos no contiene las tablas requeridas: {', '.join(sorted(missing_tables))}. "
                    "setup_db.py elimina y recrea las tablas; ejecútalo solo después de respaldar o confirmar que "
                    "puedes reemplazar los datos."
                )

            for table, required_columns in EXERCISE_06_REQUIRED_COLUMNS.items():
                actual_columns = {
                    row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
                }
                missing_columns = required_columns - actual_columns
                if missing_columns:
                    return (
                        f"La tabla {table} no contiene las columnas requeridas: "
                        f"{', '.join(sorted(missing_columns))}. No se modificó la base de datos."
                    )
        finally:
            connection.close()
    except sqlite3.Error:
        app.logger.exception("No se pudo validar la base de datos de Exercise 06.")
        return "No se pudo abrir o validar Exercise_06/company.db. Revisa que sea una base SQLite válida."

    return None


def is_safe_script_path(folder: Path, script_name: str) -> bool:
    if not script_name or script_name in {".", ".."}:
        return False
    candidate = (folder / script_name).resolve()
    base = folder.resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return False
    return candidate.is_file()


INDEX_HTML = '''
<!DOCTYPE html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GenAI Training Dashboard</title>
    <style>
      :root {
        --bg: #0f172a;
        --bg-alt: #111827;
        --panel: rgba(15, 23, 42, 0.88);
        --panel-soft: #1e293b;
        --card: #111827;
        --border: rgba(148, 163, 184, 0.2);
        --muted: #94a3b8;
        --text: #e2e8f0;
        --heading: #f8fafc;
        --primary: #38bdf8;
        --primary-strong: #0ea5e9;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #f87171;
        --shadow: 0 18px 38px rgba(8, 15, 29, 0.42);
      }

      * { box-sizing: border-box; }

      html, body {
        margin: 0;
        min-height: 100%;
        font-family: Arial, Helvetica, sans-serif;
        background: linear-gradient(135deg, #020817 0%, #0f172a 42%, #111827 100%);
        color: var(--text);
      }

      body {
        min-height: 100vh;
      }

      button, input, textarea {
        font: inherit;
      }

      .app-shell {
        display: grid;
        grid-template-columns: 320px 1fr;
        min-height: 100vh;
      }

      .sidebar {
        background: rgba(15, 23, 42, 0.8);
        border-right: 1px solid var(--border);
        padding: 24px 18px;
        backdrop-filter: blur(12px);
        align-self: start;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
      }

      .brand-block {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--border);
      }

      .brand-badge {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--primary), #8b5cf6);
        color: white;
        font-weight: 700;
      }

      .eyebrow {
        margin: 0 0 4px;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
      }

      h1, h2, h3, p {
        margin: 0;
      }

      .brand-block h1 {
        font-size: 1.15rem;
        color: var(--heading);
      }

      .search-box {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 16px;
        color: var(--muted);
        font-size: 0.8rem;
      }

      .search-box input {
        border: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        color: var(--text);
        padding: 0.8rem 0.9rem;
      }

      .search-box input:focus-visible,
      .primary-button:focus-visible,
      .secondary-button:focus-visible,
      .ghost-button:focus-visible,
      .script-button:focus-visible,
      .exercise-button:focus-visible,
      .stdin-panel textarea:focus-visible {
        outline: 3px solid rgba(56, 189, 248, 0.5);
        outline-offset: 2px;
      }

      .exercise-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .exercise-button {
        width: 100%;
        text-align: left;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.72);
        padding: 14px 14px 12px;
        color: var(--text);
        cursor: pointer;
        transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
      }

      .exercise-button:hover {
        transform: translateY(-1px);
        border-color: rgba(56, 189, 248, 0.5);
      }

      .exercise-button.active {
        background: rgba(14, 165, 233, 0.12);
        border-color: rgba(56, 189, 248, 0.62);
      }

      .exercise-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 54px;
        padding: 0.22rem 0.5rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.12);
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 10px;
      }

      .exercise-button h3 {
        font-size: 0.96rem;
        margin-bottom: 6px;
      }

      .exercise-button p {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
      }

      .content {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        padding: 24px 28px;
      }

      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 16px;
      }

      .topbar h2 {
        font-size: clamp(1.3rem, 2vw, 1.8rem);
        color: var(--heading);
      }

      .detail-description {
        max-width: 760px;
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.5;
      }

      .exercise-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 10px;
      }

      .meta-chip {
        padding: 0.3rem 0.55rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.58);
        color: var(--muted);
        font-size: 0.72rem;
      }

      .header-actions {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
      }

      .primary-button,
      .secondary-button,
      .ghost-button,
      .script-button {
        border-radius: 12px;
        border: 1px solid transparent;
        padding: 0.72rem 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: filter 0.2s ease, transform 0.2s ease;
      }

      .primary-button:hover,
      .secondary-button:hover,
      .ghost-button:hover,
      .script-button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
      }

      .primary-button {
        background: linear-gradient(135deg, var(--primary), var(--primary-strong));
        color: #062942;
      }

      .secondary-button {
        background: rgba(148, 163, 184, 0.08);
        border-color: var(--border);
        color: var(--text);
      }

      .ghost-button {
        background: transparent;
        border-color: var(--border);
        color: var(--muted);
      }

      .panel {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
      }

      .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
      }

      .panel-label {
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .description-text {
        color: var(--muted);
        line-height: 1.6;
        font-size: 0.9rem;
      }

      .visually-hidden {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      }

      .script-toolbar {
        display: flex;
        align-items: center;
        gap: 12px;
        min-height: 48px;
        margin-bottom: 12px;
      }

      .script-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }

      .script-button {
        background: rgba(15, 23, 42, 0.8);
        color: var(--text);
        border-color: var(--border);
        padding: 0.55rem 0.8rem;
        font-size: 0.82rem;
      }

      .script-button.primary {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(14, 165, 233, 0.3));
        border-color: rgba(56, 189, 248, 0.52);
      }

      .chat-panel {
        display: flex;
        flex: 1;
        flex-direction: column;
        min-height: 480px;
        overflow: hidden;
      }

      .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
      }

      .chat-heading {
        color: var(--heading);
        font-size: 0.92rem;
        font-weight: 700;
      }

      .chat-persistence {
        margin-top: 4px;
        color: var(--muted);
        font-size: 0.75rem;
      }

      .chat-messages {
        display: flex;
        flex: 1;
        flex-direction: column;
        gap: 18px;
        min-height: 260px;
        overflow: auto;
        padding: 22px clamp(14px, 5vw, 64px);
        overscroll-behavior: contain;
      }

      .empty-chat {
        display: grid;
        flex: 1;
        align-content: center;
        justify-items: center;
        gap: 10px;
        color: var(--muted);
        text-align: center;
      }

      .empty-chat strong {
        color: var(--heading);
        font-size: 1.05rem;
      }

      .empty-chat p {
        max-width: 440px;
        font-size: 0.86rem;
        line-height: 1.6;
      }

      .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        max-width: min(820px, 88%);
      }

      .chat-message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
      }

      .message-avatar {
        display: grid;
        flex: 0 0 32px;
        width: 32px;
        height: 32px;
        place-items: center;
        border: 1px solid var(--border);
        border-radius: 50%;
        background: var(--panel-soft);
        color: var(--primary);
        font-size: 0.75rem;
        font-weight: 700;
      }

      .chat-message.user .message-avatar {
        background: rgba(14, 165, 233, 0.2);
      }

      .message-content {
        min-width: 0;
      }

      .message-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0 4px 6px;
        color: var(--muted);
        font-size: 0.72rem;
      }

      .chat-message.user .message-meta {
        justify-content: flex-end;
      }

      .message-bubble {
        padding: 12px 15px;
        border: 1px solid var(--border);
        border-radius: 4px 16px 16px 16px;
        background: rgba(30, 41, 59, 0.72);
        color: var(--text);
        font-size: 0.9rem;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }

      .chat-message.user .message-bubble {
        border-color: rgba(56, 189, 248, 0.28);
        border-radius: 16px 4px 16px 16px;
        background: rgba(14, 165, 233, 0.16);
      }

      .chat-message.failed .message-bubble {
        border-color: rgba(248, 113, 113, 0.42);
      }

      .structured-response {
        display: grid;
        gap: 14px;
      }

      .result-section-label {
        margin-bottom: 7px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .sql-query {
        max-height: 260px;
        margin: 0;
        overflow: auto;
        padding: 12px 14px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: rgba(2, 6, 23, 0.55);
        color: #dbeafe;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }

      .answer-text {
        color: var(--text);
        font-size: 0.92rem;
        line-height: 1.7;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }

      .message-status {
        margin: 7px 4px 0;
        color: var(--muted);
        font-size: 0.72rem;
      }

      .chat-message.failed .message-status {
        color: var(--danger);
      }

      .stdin-panel {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 10px;
        align-items: end;
        padding: 14px 18px 10px;
        border-top: 1px solid var(--border);
      }

      .stdin-panel textarea {
        grid-column: 1 / -1;
        width: 100%;
        min-height: 62px;
        max-height: 180px;
        resize: vertical;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.72);
        color: var(--text);
        padding: 0.8rem 0.9rem;
      }

      .stdin-caption {
        grid-column: 1;
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.5;
      }

      .input-feedback {
        grid-column: 1 / -1;
        color: var(--danger);
        font-size: 0.8rem;
      }

      .stdin-panel .primary-button {
        grid-column: 2;
        grid-row: 2;
        min-width: 132px;
      }

      button:disabled,
      textarea:disabled {
        cursor: not-allowed;
        opacity: 0.58;
      }

      @media (max-width: 980px) {
        .app-shell {
          grid-template-columns: 1fr;
        }

        .sidebar {
          position: static;
          height: auto;
          max-height: none;
          border-right: none;
          border-bottom: 1px solid var(--border);
        }

      }

      @media (max-width: 560px) {
        .content {
          min-height: 75vh;
          padding: 18px 14px 22px;
        }

        .topbar {
          align-items: flex-start;
          flex-direction: column;
        }

        .header-actions {
          width: 100%;
        }

        .header-actions > * {
          flex: 1 1 auto;
        }

        .chat-panel {
          min-height: 65vh;
        }

        .chat-header {
          align-items: flex-start;
          flex-direction: column;
        }

        .chat-messages {
          padding: 18px 12px;
        }

        .chat-message {
          max-width: 96%;
        }

        .stdin-panel {
          grid-template-columns: 1fr;
          padding: 12px;
        }

        .stdin-caption,
        .stdin-panel .primary-button {
          grid-column: 1;
          grid-row: auto;
        }

        .stdin-panel .primary-button {
          width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <aside class="sidebar" aria-label="Navegación de ejercicios">
        <div class="brand-block">
          <span class="brand-badge">AI</span>
          <div>
            <p class="eyebrow">Repositorio</p>
            <h1>GenAI Training</h1>
          </div>
        </div>

        <label class="search-box" for="exerciseSearch">
          <span>Buscar</span>
          <input id="exerciseSearch" type="search" placeholder="Ej. RAG, SQL, agente..." aria-label="Buscar ejercicios" />
        </label>

        <nav id="exerciseList" class="exercise-list" aria-live="polite">__INITIAL_EXERCISE_NAV__</nav>
      </aside>

      <main class="content">
        <header class="topbar">
          <div>
            <p class="eyebrow">Dashboard</p>
            <h2 id="detailTitle">Selecciona un ejercicio</h2>
            <p id="detailDescription" class="detail-description">Elige un ejercicio para comenzar una conversación.</p>
            <div id="exerciseMeta" class="exercise-meta" aria-label="Resumen del ejercicio"></div>
          </div>
          <div class="header-actions">
            <button id="openReadmeButton" class="secondary-button" type="button">Abrir README</button>
          </div>
        </header>

        <section class="script-toolbar" aria-label="Scripts disponibles">
          <span class="panel-label">Scripts</span>
          <div id="scriptActions" class="script-actions" aria-label="Scripts disponibles"></div>
        </section>

        <section class="panel chat-panel" aria-label="Conversación del ejercicio">
          <div class="chat-header">
            <div>
              <p class="chat-heading">Conversación</p>
              <p id="chatPersistence" class="chat-persistence" role="status">El historial se guarda en este navegador.</p>
            </div>
            <button id="clearHistoryButton" class="ghost-button" type="button">Borrar historial</button>
          </div>
          <div id="chatMessages" class="chat-messages" role="log" aria-live="polite" aria-relevant="additions text" aria-label="Historial de la conversación"></div>
          <form id="chatForm" class="stdin-panel">
            <label for="stdinInput" class="visually-hidden">Mensaje o respuestas para el script</label>
            <textarea id="stdinInput" aria-describedby="stdinHelp" placeholder="Escribe un mensaje o las respuestas que solicita el script..."></textarea>
            <small id="stdinHelp" class="stdin-caption">Enter ejecuta el script principal. Shift+Enter agrega una línea. El historial queda guardado en este navegador.</small>
            <small id="inputFeedback" class="input-feedback" role="alert" hidden></small>
            <button id="sendButton" class="primary-button" type="submit">Enviar y ejecutar</button>
          </form>
        </section>
      </main>
    </div>

    <script>
      const HISTORY_STORAGE_KEY = 'genai-training-chat-history-v1';
      const INITIAL_EXERCISES = __INITIAL_EXERCISES__;

      const state = {
        exercises: INITIAL_EXERCISES,
        selectedExerciseId: null,
        conversations: Object.create(null),
        activeProcessPid: null,
        activePoller: null,
        isLaunching: false,
      };

      const elements = {
        exerciseList: document.getElementById('exerciseList'),
        exerciseSearch: document.getElementById('exerciseSearch'),
        detailTitle: document.getElementById('detailTitle'),
        detailDescription: document.getElementById('detailDescription'),
        exerciseMeta: document.getElementById('exerciseMeta'),
        scriptActions: document.getElementById('scriptActions'),
        stdinInput: document.getElementById('stdinInput'),
        chatForm: document.getElementById('chatForm'),
        chatMessages: document.getElementById('chatMessages'),
        chatPersistence: document.getElementById('chatPersistence'),
        inputFeedback: document.getElementById('inputFeedback'),
        sendButton: document.getElementById('sendButton'),
        openReadmeButton: document.getElementById('openReadmeButton'),
        clearHistoryButton: document.getElementById('clearHistoryButton'),
      };

      function setPersistenceStatus(message) {
        elements.chatPersistence.textContent = message;
      }

      function loadConversationHistory() {
        try {
          const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
          if (!stored) {
            return;
          }

          const parsed = JSON.parse(stored);
          if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw new Error('El formato del historial guardado no es válido.');
          }

          let shouldSaveHistory = false;
          for (const [exerciseId, messages] of Object.entries(parsed)) {
            if (!Array.isArray(messages)) {
              continue;
            }

            state.conversations[exerciseId] = messages.filter((message) =>
              message &&
              typeof message.id === 'string' &&
              (message.role === 'user' || message.role === 'assistant') &&
              typeof message.content === 'string'
            ).map((message) => ({
              id: message.id,
              role: message.role,
              content: message.content,
              createdAt: typeof message.createdAt === 'string' ? message.createdAt : new Date().toISOString(),
              scriptName: typeof message.scriptName === 'string' ? message.scriptName : '',
              sqlQuery: typeof message.sqlQuery === 'string' ? message.sqlQuery : '',
              status: ['running', 'completed', 'failed'].includes(message.status) ? message.status : 'completed',
            }));

            for (const message of state.conversations[exerciseId]) {
              if (message.role === 'assistant' && message.status === 'running') {
                message.status = 'failed';
                message.content = `${message.content}\n\nLa ejecución se interrumpió al salir o recargar la página.`;
                shouldSaveHistory = true;
              }

              if (exerciseId.toLowerCase() === 'exercise_06' && message.role === 'assistant' && message.scriptName === 'workflow.py') {
                const legacyOutput = message.content;
                const sqlMatch = legacyOutput.match(/(?:^|\n)SQL:\s*([\s\S]*?)(?=\nAnswer:\s*\n)/);
                const answerMatch = legacyOutput.match(/\nAnswer:\s*\n([\s\S]*?)(?=\n-{5,}\s*(?:\n|$)|\nEstado:|\n\nMenu:|$)/);
                if (sqlMatch || answerMatch) {
                  message.sqlQuery = sqlMatch ? sqlMatch[1].trim() : '';
                  message.content = answerMatch ? answerMatch[1].trim() : 'La salida anterior no contiene una respuesta legible.';
                  shouldSaveHistory = true;
                } else if (/Route:|Workflow running|GEN AI Upskilling/.test(legacyOutput)) {
                  message.content = 'La respuesta de esta ejecución anterior no está disponible en el formato nuevo.';
                  message.sqlQuery = '';
                  shouldSaveHistory = true;
                }
              }
            }
          }

          if (shouldSaveHistory) {
            saveConversationHistory();
          }
        } catch (error) {
          console.error('No se pudo cargar el historial de conversación.', error);
          setPersistenceStatus('No se pudo leer el historial guardado en este navegador.');
        }
      }

      function saveConversationHistory() {
        try {
          localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.conversations));
          setPersistenceStatus('Historial guardado en este navegador.');
          return true;
        } catch (error) {
          console.error('No se pudo guardar el historial de conversación.', error);
          setPersistenceStatus('No se pudo guardar el historial. Revisa el espacio disponible del navegador.');
          return false;
        }
      }

      function getConversation(exerciseId) {
        if (!state.conversations[exerciseId]) {
          state.conversations[exerciseId] = [];
        }
        return state.conversations[exerciseId];
      }

      function renderConversation(exerciseId) {
        elements.chatMessages.replaceChildren();
        const messages = getConversation(exerciseId);
        elements.clearHistoryButton.disabled = !messages.length || state.activeProcessPid !== null || state.isLaunching;

        if (!messages.length) {
          const emptyState = document.createElement('div');
          emptyState.className = 'empty-chat';
          const title = document.createElement('strong');
          title.textContent = 'Comienza una conversación';
          const description = document.createElement('p');
          description.textContent = 'Escribe una pregunta o las respuestas que solicita el script. La salida aparecerá aquí y quedará guardada para este ejercicio.';
          emptyState.append(title, description);
          elements.chatMessages.append(emptyState);
          return;
        }

        for (const message of messages) {
          const row = document.createElement('article');
          row.className = `chat-message ${message.role}${message.status === 'failed' ? ' failed' : ''}`;
          row.dataset.messageId = message.id;
          const avatar = document.createElement('span');
          avatar.className = 'message-avatar';
          avatar.setAttribute('aria-hidden', 'true');
          avatar.textContent = message.role === 'user' ? 'T' : 'AI';

          const content = document.createElement('div');
          content.className = 'message-content';
          const meta = document.createElement('div');
          meta.className = 'message-meta';
          const sender = document.createElement('span');
          sender.textContent = message.role === 'user' ? 'Tú' : 'GenAI Training';
          meta.append(sender);
          if (message.scriptName && !isSqlWorkflowMessage(exerciseId, message)) {
            const script = document.createElement('span');
            script.textContent = `· ${message.scriptName}`;
            meta.append(script);
          }

          const bubble = document.createElement('div');
          bubble.className = 'message-bubble';
          if (message.role === 'assistant' && isSqlWorkflowMessage(exerciseId, message) && (message.status === 'completed' || message.sqlQuery)) {
            const structured = document.createElement('div');
            structured.className = 'structured-response';
            if (message.sqlQuery) {
              const querySection = document.createElement('section');
              const queryLabel = document.createElement('p');
              queryLabel.className = 'result-section-label';
              queryLabel.textContent = 'Consulta SQL';
              const query = document.createElement('pre');
              query.className = 'sql-query';
              const queryCode = document.createElement('code');
              queryCode.textContent = message.sqlQuery;
              query.append(queryCode);
              querySection.append(queryLabel, query);
              structured.append(querySection);
            }

            const answerSection = document.createElement('section');
            const answerLabel = document.createElement('p');
            answerLabel.className = 'result-section-label';
            answerLabel.textContent = 'Respuesta';
            const answer = document.createElement('p');
            answer.className = 'answer-text';
            answer.textContent = message.content;
            answerSection.append(answerLabel, answer);
            structured.append(answerSection);
            bubble.append(structured);
          } else {
            bubble.textContent = message.content;
          }
          content.append(meta, bubble);

          if (message.role === 'assistant' && (!isSqlWorkflowMessage(exerciseId, message) || message.status !== 'completed')) {
            const status = document.createElement('p');
            status.className = 'message-status';
            status.textContent = message.status === 'running'
              ? isSqlWorkflowMessage(exerciseId, message) ? 'Preparando la respuesta…' : 'Ejecutando…'
              : message.status === 'failed'
                ? isSqlWorkflowMessage(exerciseId, message) ? 'No se pudo completar la consulta.' : 'La ejecución terminó con un error.'
                : 'Respuesta del script';
            content.append(status);
          }

          row.append(avatar, content);
          elements.chatMessages.append(row);
        }

        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
      }

      function updateRunningMessage(exerciseId, message) {
        if (state.selectedExerciseId !== exerciseId) {
          return;
        }

        const row = [...elements.chatMessages.querySelectorAll('.chat-message')]
          .find((item) => item.dataset.messageId === message.id);
        if (!row) {
          return;
        }

        const bubble = row.querySelector('.message-bubble');
        if (bubble) {
          bubble.textContent = message.content;
        }
      }

      function setExecutionBusy(isBusy) {
        state.isLaunching = isBusy && state.activeProcessPid === null;
        elements.stdinInput.disabled = isBusy;
        elements.sendButton.disabled = isBusy;
        const currentMessages = state.conversations[state.selectedExerciseId] || [];
        elements.clearHistoryButton.disabled = isBusy || !currentMessages.length;
        elements.sendButton.textContent = isBusy ? 'Ejecutando…' : 'Enviar y ejecutar';
        elements.scriptActions.querySelectorAll('button').forEach((button) => {
          button.disabled = isBusy;
        });
      }

      async function fetchExercises() {
        try {
          const response = await fetch('/api/exercises');
          if (!response.ok) {
            throw new Error(`No se pudieron cargar los ejercicios (${response.status}).`);
          }
          const data = await response.json();
          if (!Array.isArray(data.exercises)) {
            throw new Error('La respuesta de ejercicios no tiene el formato esperado.');
          }
          state.exercises = data.exercises.length ? data.exercises : state.exercises;
          if (!state.selectedExerciseId && state.exercises.length) {
            selectExercise(state.exercises[0].id);
          } else {
            renderExerciseList();
          }
        } catch (error) {
          console.error('No se pudieron actualizar los ejercicios.', error);
          if (!state.exercises.length) {
            elements.exerciseList.innerHTML = '<p class="description-text">No se pudieron cargar los ejercicios. Recarga la página para intentarlo de nuevo.</p>';
          }
        }
      }

      function renderExerciseList() {
        const term = (elements.exerciseSearch.value || '').trim().toLowerCase();
        const filtered = state.exercises.filter((exercise) => {
          const text = `${exercise.title} ${exercise.description} ${exercise.stack.join(' ')}`.toLowerCase();
          return !term || text.includes(term);
        });

        if (!filtered.length) {
          elements.exerciseList.innerHTML = '<p class="description-text">No se encontraron ejercicios con ese filtro.</p>';
          return;
        }

        elements.exerciseList.innerHTML = filtered
          .map((exercise) => `
            <button class="exercise-button ${exercise.id === state.selectedExerciseId ? 'active' : ''}" type="button" data-id="${exercise.id}">
              <span class="exercise-chip">${exercise.name.replace('Exercise_', 'Ex.')}</span>
              <h3>${exercise.title}</h3>
              <p>${exercise.description}</p>
            </button>
          `)
          .join('');

        elements.exerciseList.querySelectorAll('.exercise-button').forEach((button) => {
          button.addEventListener('click', () => {
            selectExercise(button.dataset.id);
          });
        });
      }

      function selectExercise(exerciseId) {
        const exercise = state.exercises.find((item) => item.id.toLowerCase() === exerciseId.toLowerCase());
        if (!exercise) {
          return;
        }
        state.selectedExerciseId = exercise.id;

        elements.detailTitle.textContent = exercise.title;
        elements.detailDescription.textContent = exercise.description;
        const metaItems = [
          `Stack: ${exercise.stack.join(' · ')}`,
          `Scripts: ${exercise.scripts.length}`,
          `Estado: ${exercise.status}`,
        ];
        elements.exerciseMeta.replaceChildren(...metaItems.map((item) => {
          const chip = document.createElement('span');
          chip.className = 'meta-chip';
          chip.textContent = item;
          return chip;
        }));

        const buttons = exercise.scripts
          .map((script) => `
            <button class="script-button ${script.kind === 'main' ? 'primary' : ''}" type="button" data-script="${script.name}" ${state.activeProcessPid !== null || state.isLaunching ? 'disabled' : ''}>
              ${script.kind === 'main' ? '▶ Ejecutar principal' : '◇'} ${script.name}
            </button>
          `)
          .join('');
        elements.scriptActions.innerHTML = buttons;

        document.querySelectorAll('.script-button').forEach((button) => {
          button.addEventListener('click', () => {
            runExercise(exercise.id, button.dataset.script);
          });
        });

        elements.openReadmeButton.onclick = () => window.open(`/readme/${exercise.id}`, '_blank', 'noopener,noreferrer');
        renderConversation(exercise.id);
        renderExerciseList();
      }

      function runSelectedExercise() {
        const exercise = state.exercises.find((item) => item.id === state.selectedExerciseId);
        if (exercise) {
          return runExercise(exercise.id, exercise.main_script);
        }
      }

      function isSqlWorkflowMessage(exerciseId, message) {
        return exerciseId.toLowerCase() === 'exercise_06' && message.scriptName === 'workflow.py';
      }

      async function runExercise(exerciseId, scriptName) {
        if (state.activeProcessPid !== null || state.isLaunching) {
          return;
        }

        const stdinText = (elements.stdinInput.value || '').trimEnd();
        const isSqlWorkflow = exerciseId.toLowerCase() === 'exercise_06' && scriptName === 'workflow.py';
        if (isSqlWorkflow && !stdinText.trim()) {
          elements.inputFeedback.textContent = 'Escribe una pregunta para ejecutar el agente SQL.';
          elements.inputFeedback.hidden = false;
          elements.stdinInput.focus();
          return;
        }
        elements.inputFeedback.textContent = '';
        elements.inputFeedback.hidden = true;

        const conversation = getConversation(exerciseId);
        const userContent = stdinText || (isSqlWorkflow ? 'Consultar información de la empresa' : `Ejecutar ${scriptName}`);
        const timestamp = new Date().toISOString();
        const assistantMessage = {
          id: `${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: isSqlWorkflow ? 'Analizando tu consulta…' : `Iniciando ${scriptName}…`,
          createdAt: timestamp,
          scriptName,
          sqlQuery: '',
          status: 'running',
        };
        conversation.push({
          id: `${Date.now()}-${Math.random()}-user`,
          role: 'user',
          content: userContent,
          createdAt: timestamp,
          scriptName: '',
          status: 'completed',
        });
        conversation.push(assistantMessage);
        elements.stdinInput.value = '';
        saveConversationHistory();
        renderConversation(exerciseId);
        setExecutionBusy(true);

        try {
          const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exerciseId: exerciseId, scriptName: scriptName, stdinText: stdinText }),
          });

          const payload = await response.json().catch(() => ({ error: 'No se pudo ejecutar el ejercicio.' }));
          if (!response.ok) {
            assistantMessage.content = payload.error || 'No se pudo ejecutar el ejercicio.';
            assistantMessage.status = 'failed';
            saveConversationHistory();
            renderConversation(exerciseId);
            setExecutionBusy(false);
            return;
          }

          state.activeProcessPid = payload.pid;
          pollProcess(payload.pid, exerciseId, assistantMessage);
        } catch (error) {
          console.error('No se pudo iniciar el script.', error);
          assistantMessage.content = 'No se pudo iniciar el script. Verifica la conexión con el dashboard e inténtalo de nuevo.';
          assistantMessage.status = 'failed';
          saveConversationHistory();
          renderConversation(exerciseId);
          setExecutionBusy(false);
        }
      }

      async function pollProcess(pid, exerciseId, assistantMessage) {
        try {
          const response = await fetch(`/api/process/${pid}`);
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || 'No se pudo consultar el proceso.');
          }

          if (data.status === 'running') {
            assistantMessage.content = isSqlWorkflowMessage(exerciseId, assistantMessage)
              ? 'Analizando la pregunta y preparando la respuesta…'
              : data.output || `El script ${assistantMessage.scriptName} sigue ejecutándose…`;
            updateRunningMessage(exerciseId, assistantMessage);
            state.activePoller = setTimeout(() => pollProcess(pid, exerciseId, assistantMessage), 1200);
            return;
          }

          let succeeded = data.exitCode === 0;
          if (isSqlWorkflowMessage(exerciseId, assistantMessage)) {
            let result;
            try {
              result = JSON.parse(data.output);
            } catch (error) {
              console.error('La respuesta estructurada de SQL no es válida.', error);
              result = null;
            }

            if (!result || typeof result !== 'object' || Array.isArray(result)) {
              assistantMessage.content = 'Exercise 06 no devolvió una respuesta válida. Comprueba las dependencias y revisa los logs del dashboard.';
              succeeded = false;
            } else {
              assistantMessage.sqlQuery = typeof result.sql_query === 'string' ? result.sql_query : '';
              assistantMessage.content = typeof result.final_response === 'string' && result.final_response
                ? result.final_response
                : 'La consulta terminó sin una respuesta disponible.';
              succeeded = succeeded && !result.error;
            }
          } else {
            assistantMessage.content = `${data.output || 'El proceso terminó sin generar salida.'}\n\nEstado: ${succeeded ? 'completado correctamente.' : 'finalizó con errores.'}`;
          }
          assistantMessage.status = succeeded ? 'completed' : 'failed';
          saveConversationHistory();
          if (state.selectedExerciseId === exerciseId) {
            renderConversation(exerciseId);
          }
          state.activeProcessPid = null;
          state.activePoller = null;
          setExecutionBusy(false);
        } catch (error) {
          console.error('No se pudo consultar la salida del script.', error);
          assistantMessage.content = isSqlWorkflowMessage(exerciseId, assistantMessage)
            ? 'No se pudo obtener la respuesta. Verifica la conexión e inténtalo de nuevo.'
            : `No se pudo consultar la salida del script: ${error.message}`;
          assistantMessage.status = 'failed';
          saveConversationHistory();
          if (state.selectedExerciseId === exerciseId) {
            renderConversation(exerciseId);
          }
          state.activeProcessPid = null;
          state.activePoller = null;
          setExecutionBusy(false);
        }
      }

      elements.exerciseSearch.addEventListener('input', renderExerciseList);
      elements.chatForm.addEventListener('submit', (event) => {
        event.preventDefault();
        runSelectedExercise();
      });
      elements.stdinInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          elements.chatForm.requestSubmit();
        }
      });
      elements.stdinInput.addEventListener('input', () => {
        elements.inputFeedback.textContent = '';
        elements.inputFeedback.hidden = true;
      });
      elements.clearHistoryButton.addEventListener('click', () => {
        if (!state.selectedExerciseId || !getConversation(state.selectedExerciseId).length) {
          return;
        }
        if (!window.confirm('¿Borrar todo el historial de este ejercicio?')) {
          return;
        }
        delete state.conversations[state.selectedExerciseId];
        saveConversationHistory();
        renderConversation(state.selectedExerciseId);
      });

      loadConversationHistory();
      if (state.exercises.length) {
        selectExercise(state.exercises[0].id);
      } else {
        elements.exerciseList.innerHTML = '<p class="description-text">No hay ejercicios disponibles en el repositorio.</p>';
      }
      fetchExercises();
    </script>
  </body>
</html>
'''


@app.route("/")
def index():
    exercises = list_exercises()
    initial_exercises = json.dumps(exercises, ensure_ascii=True).replace("</", "<\\/")
    exercise_navigation = render_exercise_navigation(exercises)
    html = INDEX_HTML.replace("__INITIAL_EXERCISES__", initial_exercises)
    html = html.replace("__INITIAL_EXERCISE_NAV__", exercise_navigation)
    return Response(html, mimetype="text/html; charset=utf-8")


@app.route("/api/exercises")
def api_exercises():
    return jsonify({"exercises": list_exercises()})


@app.route("/api/run", methods=["POST"])
def api_run():
    raw_payload = request.get_json(silent=True)
    payload: dict[str, object] = (
      cast(dict[str, object], raw_payload) if isinstance(raw_payload, dict) else {}
    )
    exercise_id = str(payload.get("exerciseId") or "").strip()
    script_name = str(payload.get("scriptName") or "").strip()
    stdin_text = payload.get("stdinText")
    if stdin_text is None:
        stdin_text = ""
    if not isinstance(stdin_text, str):
        stdin_text = str(stdin_text)

    if not exercise_id:
        return jsonify({"error": "Debes indicar un ejercicio."}), 400

    folder = find_exercise_dir(exercise_id)
    if not folder:
        return jsonify({"error": "No se encontró el ejercicio solicitado."}), 404

    allowed_scripts = list_allowed_scripts(folder)
    if not script_name:
        script_name = EXERCISE_OVERRIDES.get(folder.name, {}).get("main_script") or next(
            (p for p in allowed_scripts if p.lower() not in {"evaluate.py", "setup_db.py"}),
            allowed_scripts[0] if allowed_scripts else "",
        )

    if script_name not in allowed_scripts:
        return jsonify({"error": f"El script '{script_name}' no está permitido para este ejercicio."}), 400

    script_path = (folder / script_name).resolve()
    if not is_safe_script_path(folder, script_name) or not script_path.is_file():
        return jsonify({"error": f"Ruta no válida para el script '{script_name}'."}), 400

    if folder.name == "Exercise_06" and script_name == "workflow.py" and not stdin_text.strip():
        return jsonify({"error": "Escribe una pregunta para ejecutar el agente SQL."}), 400

    if folder.name == "Exercise_06" and script_name == "workflow.py":
        prerequisite_error = exercise06_prerequisite_error()
        if prerequisite_error:
            return jsonify({"error": prerequisite_error}), 503
    elif not (folder.name == "Exercise_06" and script_name == "setup_db.py") and not (
        os.environ.get("GROQ_API_KEY") or GLOBAL_GROQ_API_KEY
    ):
        return jsonify({"error": "No se encontró GROQ_API_KEY. Configúrala en .env o como variable de entorno."}), 503

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{exercise_id}-{uuid.uuid4().hex}.log"
    log_file = log_path.open("w", encoding="utf-8")

    stdin_stream = subprocess.PIPE if stdin_text else subprocess.DEVNULL
    is_sql_dashboard_run = folder.name == "Exercise_06" and script_name == "workflow.py"
    command = [sys.executable, script_name]
    if is_sql_dashboard_run:
        command.append("--dashboard")

    try:
        process = subprocess.Popen(
            command,
            cwd=str(folder),
            stdin=stdin_stream,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    except OSError as exc:
        log_file.close()
        app.logger.exception("No se pudo iniciar %s para %s.", script_name, exercise_id)
        return jsonify({"error": f"No se pudo iniciar el script solicitado: {exc}"}), 500

    if stdin_text:
        try:
            if process.stdin is None:
                raise OSError("El proceso no abrió el canal de entrada.")
            process.stdin.write(stdin_text)
            process.stdin.close()
        except (OSError, ValueError) as exc:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
            log_file.close()
            app.logger.exception("No se pudo enviar la entrada a %s para %s.", script_name, exercise_id)
            return jsonify({"error": "No se pudo enviar la pregunta al proceso. Inténtalo de nuevo."}), 500

    PROCESSES[str(process.pid)] = {
        "process": process,
        "log_file": log_file,
        "log_path": str(log_path),
        "exercise_id": exercise_id,
        "script_name": script_name,
    }

    return jsonify({"pid": process.pid, "exerciseId": exercise_id, "scriptName": script_name, "status": "running"})


@app.route("/api/process/<int:pid>")
def process_status(pid: int):
    meta = PROCESSES.get(str(pid))
    if not meta:
        return jsonify({"error": "Proceso no encontrado."}), 404

    process = meta["process"]
    status = "running" if process.poll() is None else ("completed" if process.returncode == 0 else "failed")

    if process.poll() is not None:
        meta["log_file"].flush()
        meta["log_file"].close()

    log_path = Path(meta["log_path"])
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
        output = "\n".join(lines)
    else:
        output = ""

    return jsonify(
        {
            "pid": pid,
            "status": status,
            "exitCode": process.returncode,
            "exerciseId": meta["exercise_id"],
            "scriptName": meta["script_name"],
            "output": output,
        }
    )


@app.route("/readme/<exercise_id>")
def readme_view(exercise_id: str):
    folder = find_exercise_dir(exercise_id)
    if not folder:
        return Response("Ejercicio no encontrado", mimetype="text/plain; charset=utf-8")

    readme = find_readme(folder)
    if not readme:
        return Response("No se encontró un README para este ejercicio.", mimetype="text/plain; charset=utf-8")

    content = readme.read_text(encoding="utf-8", errors="ignore")
    return Response(content, mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
