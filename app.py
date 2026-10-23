from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from contextlib import closing
from html import escape
from pathlib import Path
from typing import Any, TextIO, TypedDict, cast

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
CONVERSATION_DB_PATH = ROOT / "instance" / "conversation_history.sqlite3"
MAX_CONVERSATION_MESSAGES = 500
MAX_CONVERSATION_TEXT_LENGTH = 20_000


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
        "placeholder": "Pega aquí el texto que deseas resumir...",
        "sample_prompts": [
            "Python es un lenguaje de programación de alto nivel multiparadigma. Es ampliamente utilizado en ciencia de datos, inteligencia artificial, desarrollo web y automatización por su sintaxis legible.",
            "La arquitectura de microservicios descompone aplicaciones complejas en servicios independientes que se comunican mediante APIs ligeras, mejorando la escalabilidad y tolerancia a fallos.",
        ],
    },
    "Exercise_02": {
        "title": "Exercise 02 — Text Classifier",
        "description": "Clasifica documentos cortos en categorías jerárquicas con validación JSON estricta, comparando técnicas de prompting y un pipeline de dos llamadas.",
        "stack": ["Python", "Groq", "LLM", "JSON"],
        "main_script": "classifier.py",
        "placeholder": "Selecciona técnica (1: Zero-shot, 2: Few-shot, 3: CoT) o presiona Enviar para clasificar el dataset...",
        "sample_prompts": [
            "1",
            "2",
            "3",
        ],
    },
    "Exercise_03": {
        "title": "Exercise 03 — RAG Q&A",
        "description": "Construye un sistema RAG con embeddings, búsqueda vectorial FAISS y respuestas ancladas en un conjunto de conocimiento específico para evitar alucinaciones.",
        "stack": ["Python", "FAISS", "Embeddings", "RAG"],
        "main_script": "rag_engine.py",
        "placeholder": "Haz una pregunta sobre la base de conocimiento de TI...",
        "sample_prompts": [
            "¿Cómo solicito un reseteo de contraseña?",
            "¿Cuáles son los pasos para configurar la VPN?",
            "¿Cuál es el procedimiento para solicitar reembolso de gastos?",
        ],
    },
    "Exercise_04": {
        "title": "Exercise 04 — Advanced RAG",
        "description": "Extiende el patrón RAG con documentos multimodal, metadata, chunks, ranking y re-ranking para responder con contexto relevante y grounded.",
        "stack": ["Python", "RAG", "BM25", "pypdf", "docx"],
        "main_script": "rag_engine.py",
        "placeholder": "Pregunta sobre programación, DevOps, gestión ágil, ML o bases de datos...",
        "sample_prompts": [
            "¿Qué es un decorador en Python y cómo se usa?",
            "¿Cuáles son las diferencias clave entre Scrum y Kanban?",
            "¿Cómo se previene el sobreajuste (overfitting) en modelos de Machine Learning?",
            "¿Qué ventajas ofrece la normalización en bases de datos relacionales?",
        ],
    },
    "Exercise_05": {
        "title": "Exercise 05 — Tool Calling Agent",
        "description": "Crea un asistente con invocación de herramientas, decisiones de ejecución y respuesta final basada en datos reales del entorno y herramientas del sistema.",
        "stack": ["Python", "Groq", "Function Calling", "Agent"],
        "main_script": "agent.py",
        "placeholder": "Pregunta sobre empleados, tickets, clima, cálculos o documentos...",
        "sample_prompts": [
            "¿Cuál es el estatus del ticket INC-12345 y a quién está asignado?",
            "¿Qué información tienes sobre la empleada Alice Martinez (ID 1001)?",
            "¿Cuál es el clima actual en Monterrey?",
            "¿Cuántas personas trabajan en el departamento de IT?",
            "Calcula la raíz cuadrada de 144 multiplicada por 15.",
        ],
    },
    "Exercise_06": {
        "title": "Exercise 06 — LangGraph SQL Agent",
        "description": "Interpreta preguntas en lenguaje natural, decide si requieren SQL o RAG, y responde con trazabilidad usando un flujo de LangGraph sobre SQLite.",
        "stack": ["Python", "LangGraph", "SQLite", "RAG"],
        "main_script": "workflow.py",
        "placeholder": "Escribe una consulta en lenguaje natural sobre la empresa (tickets, empleados, proyectos)...",
        "sample_prompts": [
            "¿Cuáles son los tickets de alta prioridad que siguen abiertos?",
            "¿Cuántos proyectos activos tiene la empresa actualmente?",
            "¿Quién es el account manager de los clientes en el sector de tecnología?",
        ],
    },
}


def find_exercise_dir(exercise_id: str) -> Path | None:
    target = exercise_id.lower()
    for path in sorted(ROOT.glob("Exercise_*")):
        if path.is_dir() and path.name.lower() == target:
            return path
    return None


def validate_conversation_owner(owner_id: str) -> bool:
    """Accept only canonical UUIDs as browser-local history owners."""
    try:
        return str(uuid.UUID(owner_id)) == owner_id
    except ValueError:
        return False


def load_conversation_history(owner_id: str, exercise_id: str) -> list[dict[str, str]]:
    """Load a browser's saved messages for one exercise."""
    if not CONVERSATION_DB_PATH.exists():
        return []

    with closing(sqlite3.connect(CONVERSATION_DB_PATH)) as connection:
        rows = connection.execute(
            """
            SELECT message_id, role, content, created_at, script_name, sql_query, status
            FROM conversation_messages
            WHERE owner_id = ? AND exercise_id = ?
            ORDER BY created_at, rowid
            """,
            (owner_id, exercise_id),
        ).fetchall()

    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "createdAt": row[3],
            "scriptName": row[4],
            "sqlQuery": row[5],
            "status": row[6],
        }
        for row in rows
    ]


def save_conversation_history(
    owner_id: str,
    exercise_id: str,
    messages: list[dict[str, str]],
) -> None:
    """Replace an exercise conversation atomically in the separate history store."""
    CONVERSATION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(CONVERSATION_DB_PATH)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    owner_id TEXT NOT NULL,
                    exercise_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    script_name TEXT NOT NULL,
                    sql_query TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                    PRIMARY KEY (owner_id, exercise_id, message_id)
                )
                """
            )
            connection.execute(
                "DELETE FROM conversation_messages WHERE owner_id = ? AND exercise_id = ?",
                (owner_id, exercise_id),
            )
            connection.executemany(
                """
                INSERT INTO conversation_messages (
                    owner_id, exercise_id, message_id, role, content,
                    created_at, script_name, sql_query, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        owner_id,
                        exercise_id,
                        message["id"],
                        message["role"],
                        message["content"],
                        message["createdAt"],
                        message["scriptName"],
                        message["sqlQuery"],
                        message["status"],
                    )
                    for message in messages
                ],
            )


def normalize_conversation_messages(raw_messages: object) -> list[dict[str, str]] | None:
    """Validate and limit client-supplied messages before storing them."""
    if not isinstance(raw_messages, list) or len(raw_messages) > MAX_CONVERSATION_MESSAGES:
        return None

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            return None

        message_id = raw_message.get("id")
        role = raw_message.get("role")
        content = raw_message.get("content")
        created_at = raw_message.get("createdAt")
        script_name = raw_message.get("scriptName", "")
        sql_query = raw_message.get("sqlQuery", "")
        status = raw_message.get("status", "completed")
        if (
            not isinstance(message_id, str)
            or not message_id
            or len(message_id) > 100
            or message_id in seen_ids
            or not isinstance(role, str)
            or role not in {"user", "assistant"}
            or not isinstance(content, str)
            or len(content) > MAX_CONVERSATION_TEXT_LENGTH
            or not isinstance(created_at, str)
            or not created_at
            or len(created_at) > 64
            or not isinstance(script_name, str)
            or len(script_name) > 255
            or not isinstance(sql_query, str)
            or len(sql_query) > MAX_CONVERSATION_TEXT_LENGTH
            or not isinstance(status, str)
            or status not in {"running", "completed", "failed"}
        ):
            return None

        seen_ids.add(message_id)
        normalized.append(
            {
                "id": message_id,
                "role": role,
                "content": content,
                "createdAt": created_at,
                "scriptName": script_name,
                "sqlQuery": sql_query,
                "status": status,
            }
        )
    return normalized


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
    placeholder = override.get("placeholder") or "Escribe un mensaje o respuestas para el script..."
    sample_prompts = override.get("sample_prompts") or []

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
        "placeholder": placeholder,
        "sample_prompts": sample_prompts,
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

    cards: list[str] = []
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


def prepare_script_stdin(folder_name: str, script_name: str, stdin_text: str) -> str:
    """Format stdin text so terminal-interactive scripts execute cleanly in web headless mode."""
    cleaned = (stdin_text or "").strip()
    folder_lower = folder_name.lower()
    script_lower = script_name.lower()

    if folder_lower == "exercise_01" and script_lower == "summarizer.py":
        if not cleaned:
            cleaned = (
                "Python es un lenguaje de programación multiparadigma reconocido por su sintaxis clara y legible. "
                "Es ampliamente utilizado en ciencia de datos, inteligencia artificial, desarrollo web y automatización de procesos."
            )
        first_line = cleaned.splitlines()[0].strip()
        if first_line in {"1", "2", "3"}:
            rest = "\n".join(cleaned.splitlines()[1:]).strip()
            return f"{first_line}\n{rest}\n\n\n"
        return f"1\n{cleaned}\n\n\n"

    if folder_lower == "exercise_02" and script_lower == "classifier.py":
        if cleaned in {"1", "2", "3"}:
            return f"{cleaned}\n"
        return "1\n"

    if folder_lower == "exercise_03" and script_lower == "rag_engine.py":
        if not cleaned:
            cleaned = "¿Cómo solicito un reseteo de contraseña?"
        return f"{cleaned}\nquit\n"

    if folder_lower == "exercise_04" and script_lower == "rag_engine.py":
        if not cleaned:
            cleaned = "¿Qué es un decorador en Python y cómo se usa?"
        return f"{cleaned}\nquit\n"

    if folder_lower == "exercise_05" and script_lower == "agent.py":
        if not cleaned:
            cleaned = "¿Cuál es el estatus del ticket INC-12345?"
        return f"{cleaned}\nquit\n"

    if folder_lower == "exercise_06" and script_lower == "workflow.py":
        return cleaned

    return cleaned


def parse_exercise_output(exercise_id: str, script_name: str, raw_output: str) -> dict[str, Any]:
    """Parse terminal output into clean structured data for rich web formatting."""
    if not raw_output:
        return {"kind": "empty", "raw": ""}

    ex_id = exercise_id.lower()
    script = script_name.lower()

    # 1. Exercise 01 — Summarizer
    if ex_id == "exercise_01" and script == "summarizer.py":
        json_match = re.search(r"\{\s*\"summary\"[\s\S]*?\}", raw_output)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return {
                    "kind": "summary",
                    "summary": data.get("summary", ""),
                    "key_points": data.get("key_points", []),
                    "sentiment": data.get("sentiment", "neutral"),
                    "word_count": data.get("word_count", 0),
                    "technique": data.get("technique", "zero-shot"),
                    "raw_json": json_match.group(0),
                }
            except Exception:
                pass

    # 2. Exercise 02 — Classifier
    if ex_id == "exercise_02" and script == "classifier.py":
        json_match = re.search(r"\[\s*\{\s*\"document_name\"[\s\S]*?\}\s*\]", raw_output)
        if json_match:
            try:
                items = json.loads(json_match.group(0))
                return {
                    "kind": "classifier",
                    "total": len(items),
                    "items": items,
                    "raw_json": json_match.group(0),
                }
            except Exception:
                pass

    # 3. Exercise 03 — RAG Q&A
    if ex_id == "exercise_03" and script == "rag_engine.py":
        ans_match = re.search(r"Answer:\s*\n([\s\S]*?)(?=\n-{5,}|\nGoodbye!|\nYour question:|$)", raw_output)
        if ans_match:
            return {
                "kind": "rag",
                "answer": ans_match.group(1).strip(),
            }

    # 4. Exercise 04 — Advanced RAG
    if ex_id == "exercise_04" and script == "rag_engine.py":
        src_match = re.search(r"Sources:\s*([^\n]+)", raw_output)
        ans_match = re.search(r"Answer:\s*\n([\s\S]*?)(?=\n-{5,}|\nGoodbye!|\nYour question:|$)", raw_output)
        sources = [s.strip() for s in src_match.group(1).split(",")] if src_match else []
        answer = ans_match.group(1).strip() if ans_match else ""
        if answer:
            is_out_of_scope = "outside the scope" in answer.lower()
            return {
                "kind": "advanced_rag",
                "sources": sources,
                "answer": answer,
                "in_scope": not is_out_of_scope,
            }

    # 5. Exercise 05 — Tool Calling Agent
    if ex_id == "exercise_05" and script == "agent.py":
        tool_match = re.search(r"\[Tool\(s\) used:\s*([^\]]+)\]", raw_output)
        no_tool_match = re.search(r"\[No tool needed\]", raw_output)
        ans_match = re.search(r"Answer:\s*\n([\s\S]*?)(?=\nStructured JSON:|\n-{5,}|\nGoodbye!|$)", raw_output)
        json_match = re.search(r"Structured JSON:\s*\n(\{[\s\S]*?\})", raw_output)
        tool_name = tool_match.group(1).strip() if tool_match else (None if no_tool_match else None)
        answer = ans_match.group(1).strip() if ans_match else ""
        structured = None
        if json_match:
            try:
                structured = json.loads(json_match.group(1))
            except Exception:
                pass
        if answer:
            return {
                "kind": "agent",
                "tool": tool_name,
                "answer": answer,
                "structured": structured,
            }

    # 6. Exercise 06 — SQL Workflow
    if ex_id == "exercise_06" and script == "workflow.py":
        try:
            dashboard_data = json.loads(raw_output.strip())
            if isinstance(dashboard_data, dict):
                return {
                    "kind": "sql",
                    "sql_query": dashboard_data.get("sql_query", ""),
                    "answer": dashboard_data.get("final_response", ""),
                    "error": dashboard_data.get("error", ""),
                }
        except Exception:
            pass
        sql_match = re.search(r"(?:^|\n)SQL:\s*([\s\S]*?)(?=\nAnswer:\s*\n)", raw_output)
        ans_match = re.search(r"\nAnswer:\s*\n([\s\S]*?)(?=\n-{5,}\s*(?:\n|$)|\nEstado:|\n\nMenu:|$)", raw_output)
        if sql_match or ans_match:
            return {
                "kind": "sql",
                "sql_query": sql_match.group(1).strip() if sql_match else "",
                "answer": ans_match.group(1).strip() if ans_match else "",
                "error": "",
            }

    if "evaluate.py" in script:
        return {"kind": "evaluate", "text": raw_output.strip()}

    return {"kind": "raw", "text": raw_output.strip()}


INDEX_HTML = '''
<!DOCTYPE html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GenAI Training Dashboard</title>
    <style>
      :root {
        --bg: #090e17;
        --bg-alt: #0f172a;
        --panel: rgba(15, 23, 42, 0.88);
        --panel-soft: #1e293b;
        --card: #111827;
        --border: rgba(148, 163, 184, 0.18);
        --border-hover: rgba(56, 189, 248, 0.45);
        --muted: #94a3b8;
        --text: #e2e8f0;
        --heading: #f8fafc;
        --primary: #38bdf8;
        --primary-strong: #0ea5e9;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #f87171;
        --purple: #c084fc;
        --shadow: 0 18px 38px rgba(8, 15, 29, 0.45);
        --code-bg: rgba(2, 6, 23, 0.7);
        --code-border: rgba(56, 189, 248, 0.16);
      }

      * { box-sizing: border-box; }

      html, body {
        margin: 0;
        min-height: 100%;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 40%, #020617 100%);
        color: var(--text);
      }

      body { min-height: 100vh; }
      button, input, textarea { font: inherit; }

      .app-shell {
        display: grid;
        grid-template-columns: 340px 1fr;
        min-height: 100vh;
      }

      .sidebar {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid var(--border);
        padding: 24px 18px;
        backdrop-filter: blur(14px);
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
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, #0284c7, #8b5cf6);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.35);
      }

      .eyebrow {
        margin: 0 0 4px;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
      }

      h1, h2, h3, p { margin: 0; }

      .brand-block h1 {
        font-size: 1.15rem;
        color: var(--heading);
        font-weight: 700;
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
        padding: 0.75rem 0.9rem;
        transition: border-color 0.2s;
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
        gap: 10px;
      }

      .exercise-button {
        width: 100%;
        text-align: left;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.68);
        padding: 13px 14px 12px;
        color: var(--text);
        cursor: pointer;
        transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
      }

      .exercise-button:hover {
        transform: translateY(-2px);
        border-color: var(--border-hover);
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.12);
      }

      .exercise-button.active {
        background: rgba(14, 165, 233, 0.12);
        border-color: rgba(56, 189, 248, 0.62);
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.2), 0 6px 18px rgba(14, 165, 233, 0.14);
      }

      .exercise-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 54px;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.12);
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 8px;
      }

      .exercise-button h3 {
        font-size: 0.94rem;
        margin-bottom: 5px;
        font-weight: 600;
      }

      .exercise-button p {
        color: var(--muted);
        font-size: 0.75rem;
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
        margin-bottom: 14px;
      }

      .topbar h2 {
        font-size: clamp(1.3rem, 2vw, 1.75rem);
        color: var(--heading);
        font-weight: 700;
      }

      .detail-description {
        max-width: 780px;
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
      }

      .exercise-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 10px;
      }

      .meta-chip {
        padding: 0.28rem 0.6rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.55);
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 500;
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
        padding: 0.65rem 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: filter 0.2s ease, transform 0.15s ease, box-shadow 0.15s ease;
      }

      .primary-button:hover,
      .secondary-button:hover,
      .ghost-button:hover,
      .script-button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
      }

      .primary-button {
        background: linear-gradient(135deg, var(--primary), var(--primary-strong));
        color: #032030;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.28);
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
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
      }

      .panel-label {
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
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
        flex-direction: column;
        gap: 10px;
        margin-bottom: 12px;
      }

      .script-row {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
      }

      .script-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }

      .script-button {
        background: rgba(15, 23, 42, 0.85);
        color: var(--text);
        border-color: var(--border);
        padding: 0.5rem 0.85rem;
        font-size: 0.82rem;
      }

      .script-button.primary {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(14, 165, 233, 0.3));
        border-color: rgba(56, 189, 248, 0.55);
        color: #e0f2fe;
      }

      .quick-prompts-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        overflow-x: auto;
        padding: 2px 0 6px;
      }

      .quick-prompt-label {
        font-size: 0.72rem;
        color: var(--muted);
        white-space: nowrap;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .quick-prompt-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        font-size: 0.75rem;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
      }

      .quick-prompt-chip:hover {
        background: rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        transform: translateY(-1px);
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
        padding: 14px 20px;
        border-bottom: 1px solid var(--border);
      }

      .chat-heading {
        color: var(--heading);
        font-size: 0.94rem;
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
        padding: 20px clamp(14px, 4vw, 48px);
        overscroll-behavior: contain;
      }

      .empty-chat {
        display: grid;
        flex: 1;
        align-content: center;
        justify-items: center;
        gap: 12px;
        color: var(--muted);
        text-align: center;
        padding: 30px 20px;
      }

      .empty-chat-icon {
        font-size: 2.2rem;
        opacity: 0.5;
      }

      .empty-chat strong {
        color: var(--heading);
        font-size: 1.05rem;
      }

      .empty-chat p {
        max-width: 480px;
        font-size: 0.86rem;
        line-height: 1.6;
      }

      .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        max-width: min(880px, 92%);
        animation: fadeIn 0.2s ease-out;
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .chat-message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
      }

      .message-avatar {
        display: grid;
        flex: 0 0 34px;
        width: 34px;
        height: 34px;
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
        color: #7dd3fc;
      }

      .message-content {
        min-width: 0;
        max-width: 100%;
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
        padding: 14px 18px;
        border: 1px solid var(--border);
        border-radius: 4px 16px 16px 16px;
        background: rgba(30, 41, 59, 0.75);
        color: var(--text);
        font-size: 0.9rem;
        line-height: 1.65;
        overflow-wrap: anywhere;
      }

      .chat-message.user .message-bubble {
        border-color: rgba(56, 189, 248, 0.28);
        border-radius: 16px 4px 16px 16px;
        background: rgba(14, 165, 233, 0.16);
        white-space: pre-wrap;
      }

      .chat-message.failed .message-bubble {
        border-color: rgba(248, 113, 113, 0.42);
        background: rgba(239, 68, 68, 0.08);
      }

      .card-header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
      }

      .badge-sentiment-positive {
        background: rgba(34, 197, 94, 0.15);
        color: var(--success);
        border: 1px solid rgba(34, 197, 94, 0.35);
      }

      .badge-sentiment-neutral {
        background: rgba(56, 189, 248, 0.12);
        color: var(--primary);
        border: 1px solid rgba(56, 189, 248, 0.3);
      }

      .badge-sentiment-negative {
        background: rgba(248, 113, 113, 0.15);
        color: var(--danger);
        border: 1px solid rgba(248, 113, 113, 0.35);
      }

      .badge-tool {
        background: rgba(192, 132, 252, 0.15);
        color: var(--purple);
        border: 1px solid rgba(192, 132, 252, 0.35);
      }

      .summary-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid var(--primary);
        padding: 10px 14px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 12px;
        font-size: 0.94rem;
        line-height: 1.6;
        color: #f1f5f9;
      }

      .key-points-list {
        margin: 8px 0 12px;
        padding-left: 0;
        list-style: none;
        display: grid;
        gap: 6px;
      }

      .key-point-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        background: rgba(15, 23, 42, 0.45);
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        font-size: 0.85rem;
      }

      .key-point-bullet {
        color: var(--primary);
        font-weight: 700;
      }

      .classification-table-wrapper {
        overflow-x: auto;
        margin: 10px 0;
        border-radius: 10px;
        border: 1px solid var(--border);
      }

      .classification-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        text-align: left;
      }

      .classification-table th {
        background: rgba(30, 41, 59, 0.9);
        color: var(--primary);
        font-weight: 700;
        padding: 8px 12px;
        border-bottom: 1px solid var(--border);
      }

      .classification-table td {
        padding: 8px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      }

      .classification-table tr:last-child td {
        border-bottom: none;
      }

      .category-pill {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
      }

      .category-pill.error {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
      }

      .sources-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px dashed var(--border);
      }

      .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #93c5fd;
        font-size: 0.75rem;
        font-family: monospace;
      }

      .structured-response {
        display: grid;
        gap: 14px;
      }

      .result-section-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      .sql-query {
        max-height: 240px;
        margin: 0;
        overflow: auto;
        padding: 12px 14px;
        border: 1px solid var(--code-border);
        border-radius: 10px;
        background: var(--code-bg);
        color: #bae6fd;
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.82rem;
        line-height: 1.55;
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

      .terminal-details {
        margin-top: 12px;
        border-top: 1px solid var(--border);
        padding-top: 8px;
      }

      .terminal-details summary {
        font-size: 0.75rem;
        color: var(--muted);
        cursor: pointer;
        padding: 4px 0;
        user-select: none;
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }

      .terminal-details summary:hover {
        color: var(--primary);
      }

      .terminal-raw-code {
        margin-top: 6px;
        max-height: 220px;
        overflow: auto;
        padding: 10px 12px;
        border-radius: 8px;
        background: rgba(2, 6, 23, 0.75);
        border: 1px solid var(--border);
        color: #94a3b8;
        font-family: Consolas, monospace;
        font-size: 0.76rem;
        line-height: 1.5;
        white-space: pre-wrap;
      }

      .copy-btn {
        background: transparent;
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: var(--muted);
        border-radius: 6px;
        padding: 0.15rem 0.45rem;
        font-size: 0.68rem;
        cursor: pointer;
        transition: all 0.15s ease;
      }

      .copy-btn:hover {
        border-color: var(--primary);
        color: var(--primary);
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
        min-height: 64px;
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
        min-width: 140px;
      }

      button:disabled,
      textarea:disabled {
        cursor: not-allowed;
        opacity: 0.58;
      }

      @media (max-width: 980px) {
        .app-shell { grid-template-columns: 1fr; }
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
          padding: 16px 12px 20px;
        }
        .topbar {
          align-items: flex-start;
          flex-direction: column;
        }
        .header-actions { width: 100%; }
        .header-actions > * { flex: 1 1 auto; }
        .chat-panel { min-height: 65vh; }
        .chat-header {
          align-items: flex-start;
          flex-direction: column;
        }
        .chat-messages { padding: 16px 10px; }
        .chat-message { max-width: 96%; }
        .stdin-panel {
          grid-template-columns: 1fr;
          padding: 12px;
        }
        .stdin-caption,
        .stdin-panel .primary-button {
          grid-column: 1;
          grid-row: auto;
        }
        .stdin-panel .primary-button { width: 100%; }
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
          <div class="script-row">
            <span class="panel-label">Scripts</span>
            <div id="scriptActions" class="script-actions" aria-label="Scripts disponibles"></div>
          </div>
          <div id="quickPromptsContainer" class="quick-prompts-bar" hidden>
            <span class="quick-prompt-label">Sugerencias:</span>
            <div id="quickPromptChips" style="display:flex; gap:6px; flex-wrap:wrap;"></div>
          </div>
        </section>

        <section class="panel chat-panel" aria-label="Conversación del ejercicio">
          <div class="chat-header">
            <div>
              <p class="chat-heading">Conversación</p>
              <p id="chatPersistence" class="chat-persistence" role="status">La conversación se guarda para consultarla después.</p>
            </div>
            <button id="clearHistoryButton" class="ghost-button" type="button">Borrar historial</button>
          </div>
          <div id="chatMessages" class="chat-messages" role="log" aria-live="polite" aria-relevant="additions text" aria-label="Historial de la conversación"></div>
          <form id="chatForm" class="stdin-panel">
            <label for="stdinInput" class="visually-hidden">Mensaje o respuestas para el script</label>
            <textarea id="stdinInput" aria-describedby="stdinHelp" placeholder="Escribe un mensaje o las respuestas que solicita el script..."></textarea>
            <small id="stdinHelp" class="stdin-caption">Enter ejecuta el script principal. Shift+Enter agrega una línea. La conversación se guarda para que puedas volver a consultarla.</small>
            <small id="inputFeedback" class="input-feedback" role="alert" hidden></small>
            <button id="sendButton" class="primary-button" type="submit">Enviar y ejecutar</button>
          </form>
        </section>
      </main>
    </div>

    <script>
      const HISTORY_STORAGE_KEY = 'genai-training-chat-history-v1';
      const HISTORY_OWNER_STORAGE_KEY = 'genai-training-history-owner-v1';
      const MAX_CONVERSATION_MESSAGES = 500;
      const INITIAL_EXERCISES = __INITIAL_EXERCISES__;

      const state = {
        exercises: INITIAL_EXERCISES,
        selectedExerciseId: null,
        conversations: Object.create(null),
        historyOwnerId: null,
        historyWriteQueues: Object.create(null),
        historyRestorePromises: Object.create(null),
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
        quickPromptsContainer: document.getElementById('quickPromptsContainer'),
        quickPromptChips: document.getElementById('quickPromptChips'),
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

      function createHistoryOwnerId() {
        if (crypto.randomUUID) {
          return crypto.randomUUID();
        }
        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
        return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
      }

      function initializeHistoryOwner() {
        try {
          const storedOwnerId = localStorage.getItem(HISTORY_OWNER_STORAGE_KEY);
          if (storedOwnerId && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(storedOwnerId)) {
            state.historyOwnerId = storedOwnerId;
            return;
          }
          state.historyOwnerId = createHistoryOwnerId();
          localStorage.setItem(HISTORY_OWNER_STORAGE_KEY, state.historyOwnerId);
        } catch (error) {
          state.historyOwnerId = null;
          console.error('No se pudo inicializar el identificador del historial.', error);
          setPersistenceStatus('No se pudo preparar el guardado del historial en este navegador.');
        }
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
                message.content = `${message.content}

La ejecución se interrumpió al salir o recargar la página.`;
                shouldSaveHistory = true;
              }

              if (exerciseId.toLowerCase() === 'exercise_06' && message.role === 'assistant' && message.scriptName === 'workflow.py') {
                const legacyOutput = message.content;
                const sqlMatch = legacyOutput.match(/(?:^|\\n)SQL:\\s*([\\s\\S]*?)(?=\\nAnswer:\\s*\\n)/);
                const answerMatch = legacyOutput.match(/\\nAnswer:\\s*\\n([\\s\\S]*?)(?=\\n-{5,}\\s*(?:\\n|$)|\\nEstado:|\\n\\nMenu:|$)/);
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

      function saveConversationHistory(exerciseId = null) {
        let localSaveSucceeded = true;
        try {
          localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.conversations));
          setPersistenceStatus('Historial guardado en este navegador.');
        } catch (error) {
          console.error('No se pudo guardar el historial de conversación.', error);
          setPersistenceStatus('No se pudo guardar el historial. Revisa el espacio disponible del navegador.');
          localSaveSucceeded = false;
        }
        if (exerciseId) {
          syncConversationHistory(exerciseId);
        }
        return localSaveSucceeded;
      }

      function syncConversationHistory(exerciseId) {
        if (!state.historyOwnerId) {
          return;
        }
        const messages = [...getConversation(exerciseId)];
        const previousWrite = state.historyWriteQueues[exerciseId] || Promise.resolve();
        const write = previousWrite.catch(() => {}).then(async () => {
          const hasMessages = messages.length > 0;
          const response = await fetch(
            `/api/conversations/${encodeURIComponent(state.historyOwnerId)}/${encodeURIComponent(exerciseId)}`,
            {
              method: hasMessages ? 'PUT' : 'DELETE',
              headers: hasMessages ? { 'Content-Type': 'application/json' } : {},
              body: hasMessages ? JSON.stringify({ messages }) : undefined,
            }
          );
          const result = await response.json();
          if (!response.ok) {
            throw new Error(result.error || 'No se pudo guardar el historial.');
          }
        });
        state.historyWriteQueues[exerciseId] = write;
        setPersistenceStatus('Guardando la conversación…');
        write.then(() => {
          if (state.selectedExerciseId === exerciseId) {
            setPersistenceStatus('Conversación guardada. Puedes consultarla aquí cuando vuelvas.');
          }
        }).catch((error) => {
          console.error('No se pudo sincronizar el historial de conversación.', error);
          if (state.selectedExerciseId === exerciseId) {
            setPersistenceStatus('No se pudo guardar en el servidor; se conserva la copia de este navegador.');
          }
        });
      }

      function restoreConversationHistory(exerciseId) {
        if (!state.historyOwnerId || state.historyRestorePromises[exerciseId]) {
          return state.historyRestorePromises[exerciseId] || Promise.resolve();
        }

        const restore = (async () => {
          try {
            const response = await fetch(
              `/api/conversations/${encodeURIComponent(state.historyOwnerId)}/${encodeURIComponent(exerciseId)}`
            );
            const result = await response.json();
            if (!response.ok || !Array.isArray(result.messages)) {
              throw new Error(result.error || 'El servidor devolvió un historial no válido.');
            }

            const localMessages = getConversation(exerciseId);
            const messagesById = new Map();
            for (const message of result.messages) {
              if (
                message &&
                typeof message.id === 'string' &&
                (message.role === 'user' || message.role === 'assistant') &&
                typeof message.content === 'string'
              ) {
                messagesById.set(message.id, message);
              }
            }
            for (const message of localMessages) {
              messagesById.set(message.id, message);
            }

            const restoredMessages = [...messagesById.values()]
              .sort((left, right) => Date.parse(left.createdAt) - Date.parse(right.createdAt))
              .slice(-MAX_CONVERSATION_MESSAGES);
            for (const message of restoredMessages) {
              if (message.role === 'assistant' && message.status === 'running') {
                message.status = 'failed';
                message.content = `${message.content}

La ejecución se interrumpió antes de terminar.`;
              }
            }
            state.conversations[exerciseId] = restoredMessages;
            try {
              localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.conversations));
            } catch (error) {
              console.error('No se pudo actualizar la copia local del historial.', error);
            }
            if (restoredMessages.length) {
              syncConversationHistory(exerciseId);
            } else if (state.selectedExerciseId === exerciseId) {
              setPersistenceStatus('Todavía no hay mensajes guardados en esta conversación.');
            }
            if (state.selectedExerciseId === exerciseId) {
              renderConversation(exerciseId);
            }
          } catch (error) {
            console.error('No se pudo consultar el historial guardado.', error);
            if (state.selectedExerciseId === exerciseId) {
              setPersistenceStatus('No se pudo consultar el historial del servidor; se muestra la copia de este navegador.');
            }
          } finally {
            delete state.historyRestorePromises[exerciseId];
          }
        })();
        state.historyRestorePromises[exerciseId] = restore;
        return restore;
      }

      function getConversation(exerciseId) {
        if (!state.conversations[exerciseId]) {
          state.conversations[exerciseId] = [];
        }
        const conversation = state.conversations[exerciseId];
        if (conversation.length > MAX_CONVERSATION_MESSAGES) {
          conversation.splice(0, conversation.length - MAX_CONVERSATION_MESSAGES);
        }
        return conversation;
      }

      function isSqlWorkflowMessage(exerciseId, message) {
        return exerciseId.toLowerCase() === 'exercise_06' && message.scriptName === 'workflow.py';
      }

      function copyTextToClipboard(text, btnElement) {
        navigator.clipboard.writeText(text).then(() => {
          const original = btnElement.textContent;
          btnElement.textContent = '✓ Copiado';
          setTimeout(() => { btnElement.textContent = original; }, 1800);
        }).catch(() => {
          btnElement.textContent = 'Error';
        });
      }

      const NL = String.fromCharCode(10);

      function extractAnswerFromRaw(rawText) {
        if (!rawText) return '';
        const lower = rawText.toLowerCase();
        const marker = 'answer:';
        const idx = lower.lastIndexOf(marker);
        if (idx === -1) return '';
        let ans = rawText.slice(idx + marker.length).trim();
        const delimiters = [NL + '---', NL + 'Structured JSON:', NL + 'Goodbye!', NL + 'Menu:', NL + 'Estado:'];
        for (const delim of delimiters) {
          const cut = ans.indexOf(delim);
          if (cut !== -1) {
            ans = ans.slice(0, cut).trim();
          }
        }
        return ans;
      }

      function extractSourcesFromRaw(rawText) {
        if (!rawText) return [];
        const lines = rawText.split(NL);
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('Sources:')) {
            return trimmed.replace('Sources:', '').split(',').map((s) => s.trim()).filter(Boolean);
          }
        }
        return [];
      }

      function extractToolFromRaw(rawText) {
        if (!rawText) return null;
        if (rawText.includes('[No tool needed]')) return 'Sin herramientas requeridas';
        const start = rawText.indexOf('[Tool(s) used:');
        if (start !== -1) {
          const end = rawText.indexOf(']', start);
          if (end !== -1) {
            return rawText.slice(start + 14, end).trim();
          }
        }
        return null;
      }

      function renderWebFormattedBubble(exerciseId, message) {
        const container = document.createElement('div');
        const exId = (exerciseId || '').toLowerCase();
        const content = message.content || '';
        const rawOutput = message.rawOutput || content;

        // Exercise 06 (SQL Workflow)
        if (isSqlWorkflowMessage(exerciseId, message) && (message.status === 'completed' || message.sqlQuery)) {
          const structured = document.createElement('div');
          structured.className = 'structured-response';
          if (message.sqlQuery) {
            const querySection = document.createElement('section');
            const queryLabel = document.createElement('div');
            queryLabel.className = 'result-section-label';
            const labelText = document.createElement('span');
            labelText.textContent = 'Consulta SQL';
            const copyBtn = document.createElement('button');
            copyBtn.type = 'button';
            copyBtn.className = 'copy-btn';
            copyBtn.textContent = 'Copiar SQL';
            copyBtn.onclick = () => copyTextToClipboard(message.sqlQuery, copyBtn);
            queryLabel.append(labelText, copyBtn);

            const query = document.createElement('pre');
            query.className = 'sql-query';
            const queryCode = document.createElement('code');
            queryCode.textContent = message.sqlQuery;
            query.append(queryCode);
            querySection.append(queryLabel, query);
            structured.append(querySection);
          }

          const answerSection = document.createElement('section');
          const answerLabel = document.createElement('div');
          answerLabel.className = 'result-section-label';
          answerLabel.textContent = 'Respuesta';
          const answer = document.createElement('p');
          answer.className = 'answer-text';
          answer.textContent = content;
          answerSection.append(answerLabel, answer);
          structured.append(answerSection);
          container.append(structured);
          return container;
        }

        // Exercise 01 (Summarizer)
        if (exId === 'exercise_01' && rawOutput.includes('"summary"')) {
          let data = null;
          try {
            const startBrace = rawOutput.indexOf('{');
            const endBrace = rawOutput.lastIndexOf('}');
            if (startBrace !== -1 && endBrace > startBrace) {
              data = JSON.parse(rawOutput.slice(startBrace, endBrace + 1));
            }
          } catch (e) {}

          if (data && data.summary) {
            const sentiment = (data.sentiment || 'neutral').toLowerCase();
            const badgeClass = sentiment === 'positive' ? 'badge-sentiment-positive' : (sentiment === 'negative' ? 'badge-sentiment-negative' : 'badge-sentiment-neutral');

            const badgesDiv = document.createElement('div');
            badgesDiv.style.display = 'flex';
            badgesDiv.style.gap = '8px';
            badgesDiv.style.flexWrap = 'wrap';
            badgesDiv.style.marginBottom = '10px';

            const sentimentBadge = document.createElement('span');
            sentimentBadge.className = `card-header-badge ${badgeClass}`;
            sentimentBadge.textContent = `Sentimiento: ${sentiment}`;

            const techBadge = document.createElement('span');
            techBadge.className = 'card-header-badge badge-sentiment-neutral';
            techBadge.textContent = `Técnica: ${data.technique || 'Zero-shot'}`;

            const wordsBadge = document.createElement('span');
            wordsBadge.className = 'card-header-badge badge-sentiment-neutral';
            wordsBadge.textContent = `${data.word_count || 0} palabras`;

            badgesDiv.append(sentimentBadge, techBadge, wordsBadge);
            container.append(badgesDiv);

            const summaryBox = document.createElement('div');
            summaryBox.className = 'summary-box';
            summaryBox.textContent = data.summary;
            container.append(summaryBox);

            if (Array.isArray(data.key_points) && data.key_points.length) {
              const pointsTitle = document.createElement('div');
              pointsTitle.className = 'result-section-label';
              pointsTitle.textContent = 'Puntos Clave';
              container.append(pointsTitle);

              const pointsList = document.createElement('ul');
              pointsList.className = 'key-points-list';
              data.key_points.forEach((point) => {
                const li = document.createElement('li');
                li.className = 'key-point-item';
                li.textContent = `• ${point}`;
                pointsList.append(li);
              });
              container.append(pointsList);
            }
            return container;
          }
        }

        // Exercise 02 (Classifier)
        if (exId === 'exercise_02' && rawOutput.includes('"document_name"')) {
          let items = null;
          try {
            const startBracket = rawOutput.indexOf('[');
            const endBracket = rawOutput.lastIndexOf(']');
            if (startBracket !== -1 && endBracket > startBracket) {
              items = JSON.parse(rawOutput.slice(startBracket, endBracket + 1));
            }
          } catch (e) {}

          if (Array.isArray(items) && items.length) {
            const heading = document.createElement('div');
            heading.className = 'result-section-label';
            heading.textContent = `Documentos Clasificados (${items.length})`;
            container.append(heading);

            const wrapper = document.createElement('div');
            wrapper.className = 'classification-table-wrapper';
            const table = document.createElement('table');
            table.className = 'classification-table';

            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            for (const colTitle of ['Documento', 'Categoría', 'Subcategoría']) {
              const th = document.createElement('th');
              th.textContent = colTitle;
              headerRow.append(th);
            }
            thead.append(headerRow);

            const tbody = document.createElement('tbody');
            for (const item of items) {
              const tr = document.createElement('tr');
              const tdDoc = document.createElement('td');
              const code = document.createElement('code');
              code.textContent = item.document_name || '';
              tdDoc.append(code);

              const tdCat = document.createElement('td');
              const pill = document.createElement('span');
              pill.className = `category-pill ${item.category === 'ERROR' ? 'error' : ''}`;
              pill.textContent = item.category || '';
              tdCat.append(pill);

              const tdSub = document.createElement('td');
              tdSub.textContent = item.sub_category || '';

              tr.append(tdDoc, tdCat, tdSub);
              tbody.append(tr);
            }
            table.append(thead, tbody);
            wrapper.append(table);
            container.append(wrapper);
            return container;
          }
        }

        // Exercise 04 (Advanced RAG)
        if (exId === 'exercise_04') {
          const answer = extractAnswerFromRaw(rawOutput);
          const sources = extractSourcesFromRaw(rawOutput);

          if (answer) {
            const ansP = document.createElement('p');
            ansP.className = 'answer-text';
            ansP.textContent = answer;
            container.append(ansP);

            if (sources.length) {
              const sourcesDiv = document.createElement('div');
              sourcesDiv.className = 'sources-container';
              const label = document.createElement('span');
              label.style.fontSize = '0.72rem';
              label.style.color = 'var(--muted)';
              label.textContent = 'Fuentes consultadas:';
              sourcesDiv.append(label);
              sources.forEach((src) => {
                const tag = document.createElement('span');
                tag.className = 'source-tag';
                tag.textContent = `📄 ${src}`;
                sourcesDiv.append(tag);
              });
              container.append(sourcesDiv);
            }
            return container;
          }
        }

        // Exercise 05 (Tool Calling Agent)
        if (exId === 'exercise_05') {
          const toolName = extractToolFromRaw(rawOutput);
          const answer = extractAnswerFromRaw(rawOutput);

          if (toolName) {
            const badge = document.createElement('span');
            badge.className = 'card-header-badge badge-tool';
            badge.textContent = `🛠️ ${toolName}`;
            container.append(badge);
          }

          if (answer) {
            const ansP = document.createElement('p');
            ansP.className = 'answer-text';
            ansP.textContent = answer;
            container.append(ansP);
            return container;
          }
        }

        // Exercise 03 (Basic RAG)
        if (exId === 'exercise_03') {
          const answer = extractAnswerFromRaw(rawOutput);
          if (answer) {
            const ansP = document.createElement('p');
            ansP.className = 'answer-text';
            ansP.textContent = answer;
            container.append(ansP);
            return container;
          }
        }

        // Default text fallback
        const defaultP = document.createElement('p');
        defaultP.className = 'answer-text';
        defaultP.textContent = content;
        container.append(defaultP);
        return container;
      }

      function renderConversation(exerciseId) {
        elements.chatMessages.replaceChildren();
        const messages = getConversation(exerciseId);
        elements.clearHistoryButton.disabled = !messages.length || state.activeProcessPid !== null || state.isLaunching;

        if (!messages.length) {
          const emptyState = document.createElement('div');
          emptyState.className = 'empty-chat';
          const icon = document.createElement('div');
          icon.className = 'empty-chat-icon';
          icon.textContent = '💬';
          const title = document.createElement('strong');
          title.textContent = 'Comienza una conversación';
          const description = document.createElement('p');
          description.textContent = 'Escribe una pregunta o utiliza una de las sugerencias rápidas. El script procesará la solicitud y presentará los resultados en formato web estructurado.';
          emptyState.append(icon, title, description);
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
          if (message.role === 'assistant') {
            const formattedContent = renderWebFormattedBubble(exerciseId, message);
            bubble.append(formattedContent);

            const rawText = message.rawOutput || message.content || '';
            if (rawText && rawText.length > 30 && (rawText.includes(String.fromCharCode(10)) || rawText.includes('='))) {
              const details = document.createElement('details');
              details.className = 'terminal-details';
              const summary = document.createElement('summary');
              summary.textContent = '🖥️ Ver salida original de terminal';
              const pre = document.createElement('pre');
              pre.className = 'terminal-raw-code';
              pre.textContent = rawText;
              details.append(summary, pre);
              bubble.append(details);
            }
          } else {
            bubble.textContent = message.content;
          }
          content.append(meta, bubble);

          if (message.role === 'assistant' && (!isSqlWorkflowMessage(exerciseId, message) || message.status !== 'completed')) {
            const status = document.createElement('p');
            status.className = 'message-status';
            status.textContent = message.status === 'running'
              ? (isSqlWorkflowMessage(exerciseId, message) ? 'Preparando la respuesta…' : 'Ejecutando en segundo plano…')
              : message.status === 'failed'
                ? (isSqlWorkflowMessage(exerciseId, message) ? 'No se pudo completar la consulta.' : 'La ejecución terminó con un error.')
                : 'Respuesta completada';
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
        elements.quickPromptChips.querySelectorAll('button').forEach((chip) => {
          chip.disabled = isBusy;
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

        if (exercise.placeholder) {
          elements.stdinInput.placeholder = exercise.placeholder;
        } else {
          elements.stdinInput.placeholder = 'Escribe un mensaje o las respuestas que solicita el script...';
        }

        const prompts = Array.isArray(exercise.sample_prompts) ? exercise.sample_prompts : [];
        if (prompts.length) {
          elements.quickPromptsContainer.hidden = false;
          elements.quickPromptChips.replaceChildren(
            ...prompts.map((promptText) => {
              const chip = document.createElement('button');
              chip.className = 'quick-prompt-chip';
              chip.type = 'button';
              chip.disabled = state.activeProcessPid !== null || state.isLaunching;
              chip.textContent = promptText.length > 58 ? `${promptText.slice(0, 55)}...` : promptText;
              chip.addEventListener('click', () => {
                elements.stdinInput.value = promptText;
                elements.stdinInput.focus();
              });
              return chip;
            })
          );
        } else {
          elements.quickPromptsContainer.hidden = true;
          elements.quickPromptChips.replaceChildren();
        }

        elements.openReadmeButton.onclick = () => window.open(`/readme/${exercise.id}`, '_blank', 'noopener,noreferrer');
        renderConversation(exercise.id);
        restoreConversationHistory(exercise.id);
        renderExerciseList();
      }

      function runSelectedExercise() {
        const exercise = state.exercises.find((item) => item.id === state.selectedExerciseId);
        if (exercise) {
          return runExercise(exercise.id, exercise.main_script);
        }
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
        getConversation(exerciseId);
        elements.stdinInput.value = '';
        saveConversationHistory(exerciseId);
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
            saveConversationHistory(exerciseId);
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
          saveConversationHistory(exerciseId);
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
            assistantMessage.content = `${data.output || 'El proceso terminó sin generar salida.'}

Estado: ${succeeded ? 'completado correctamente.' : 'finalizó con errores.'}`;
          }

          assistantMessage.status = succeeded ? 'completed' : 'failed';
          saveConversationHistory(exerciseId);
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
          saveConversationHistory(exerciseId);
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
        saveConversationHistory(state.selectedExerciseId);
        renderConversation(state.selectedExerciseId);
      });

      initializeHistoryOwner();
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


@app.route(
    "/api/conversations/<owner_id>/<exercise_id>",
    methods=["GET", "PUT", "DELETE"],
)
def api_conversation_history(owner_id: str, exercise_id: str):
    if not validate_conversation_owner(owner_id):
        return jsonify({"error": "La identificación del historial no es válida."}), 400

    exercise_folder = find_exercise_dir(exercise_id)
    if not exercise_folder:
        return jsonify({"error": "No se encontró el ejercicio solicitado."}), 404
    normalized_exercise_id = exercise_folder.name.lower()

    try:
        if request.method == "GET":
            messages = load_conversation_history(owner_id, normalized_exercise_id)
            return jsonify({"messages": messages})

        if request.method == "DELETE":
            save_conversation_history(owner_id, normalized_exercise_id, [])
            return jsonify({"saved": True, "messages": []})

        raw_payload = request.get_json(silent=True)
        if not isinstance(raw_payload, dict):
            return jsonify({"error": "El formato del historial no es válido."}), 400
        messages = normalize_conversation_messages(raw_payload.get("messages"))
        if messages is None:
            return jsonify({"error": "Los mensajes del historial no son válidos."}), 400

        save_conversation_history(owner_id, normalized_exercise_id, messages)
        return jsonify({"saved": True, "messages": messages})
    except (OSError, sqlite3.Error):
        app.logger.exception("No se pudo acceder al historial de conversación.")
        return jsonify({"error": "No se pudo guardar o consultar el historial de conversación."}), 500


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

    prepared_stdin = prepare_script_stdin(folder.name, script_name, stdin_text)
    needs_pipe = bool(prepared_stdin)
    stdin_stream = subprocess.PIPE if needs_pipe else subprocess.DEVNULL
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
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
    except Exception as exc:
        log_file.close()
        app.logger.exception("No se pudo iniciar %s para %s.", script_name, exercise_id)
        return jsonify({"error": f"No se pudo iniciar el script solicitado: {exc}"}), 500

    if needs_pipe:
        try:
            if process.stdin is None:
                raise OSError("El proceso no abrió el canal de entrada.")
            process.stdin.write(prepared_stdin)
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
        try:
            if not getattr(meta["log_file"], "closed", False):
                meta["log_file"].flush()
                meta["log_file"].close()
        except Exception:
            pass

    log_path = Path(meta["log_path"])
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
        output = "\n".join(lines)
    else:
        output = ""

    parsed = parse_exercise_output(meta["exercise_id"], meta["script_name"], output)

    return jsonify(
        {
            "pid": pid,
            "status": status,
            "exitCode": process.returncode,
            "exerciseId": meta["exercise_id"],
            "scriptName": meta["script_name"],
            "output": output,
            "parsed": parsed,
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