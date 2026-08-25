from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
PROCESSES: dict[str, dict] = {}

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
            result = []
            for next_line in lines[index + 1 :]:
                if next_line.startswith("##") or next_line.startswith("#"):
                    break
                clean = next_line.strip()
                if clean:
                    result.append(clean)
            if result:
                return " ".join(result[:3])
    return fallback


def build_exercise_record(folder: Path) -> dict:
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

    scripts = []
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

      button, input {
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
      .exercise-button:focus-visible {
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
        padding: 32px 28px 20px;
      }

      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 22px;
      }

      .topbar h2 {
        font-size: clamp(1.6rem, 2vw, 2.2rem);
        color: var(--heading);
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

      .overview-grid {
        display: grid;
        grid-template-columns: 1.5fr 0.9fr;
        gap: 18px;
        margin-bottom: 18px;
      }

      .panel {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
      }

      .info-panel,
      .stats-panel,
      .action-panel,
      .console-panel {
        padding: 18px 20px;
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
        color: var(--text);
        line-height: 1.7;
        font-size: 1rem;
      }

      .stat-list {
        display: grid;
        gap: 12px;
      }

      .stat-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 10px 12px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid var(--border);
        border-radius: 12px;
      }

      .stat-label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .stat-item strong {
        color: var(--heading);
        font-size: 0.92rem;
      }

      .action-panel {
        margin-bottom: 18px;
      }

      .script-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }

      .script-button {
        background: rgba(15, 23, 42, 0.8);
        color: var(--text);
        border-color: var(--border);
      }

      .script-button.primary {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(14, 165, 233, 0.3));
        border-color: rgba(56, 189, 248, 0.52);
      }

      .console-panel {
        min-height: 300px;
      }

      .terminal-header {
        margin-bottom: 14px;
      }

      .console-output {
        margin: 0;
        min-height: 210px;
        max-height: 420px;
        overflow: auto;
        padding: 14px 16px;
        border-radius: 12px;
        background: rgba(2, 6, 23, 0.82);
        border: 1px solid var(--border);
        color: #dbeafe;
        font-size: 0.82rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
      }

      @media (max-width: 980px) {
        .app-shell {
          grid-template-columns: 1fr;
        }

        .sidebar {
          border-right: none;
          border-bottom: 1px solid var(--border);
        }

        .overview-grid {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 560px) {
        .content {
          padding: 18px 14px 30px;
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

        <nav id="exerciseList" class="exercise-list" aria-live="polite"></nav>
      </aside>

      <main class="content" aria-live="polite">
        <header class="topbar">
          <div>
            <p class="eyebrow">Dashboard</p>
            <h2 id="detailTitle">Selecciona un ejercicio</h2>
          </div>
          <div class="header-actions">
            <button id="openReadmeButton" class="secondary-button" type="button">Abrir README</button>
            <button id="runButton" class="primary-button" type="button">Ejecutar</button>
          </div>
        </header>

        <section class="overview-grid">
          <article class="info-panel panel">
            <div class="panel-header">
              <span class="panel-label">Descripción</span>
            </div>
            <p id="detailDescription" class="description-text">
              Elige un ejercicio para revisar objetivos, stack tecnológico y cómo ejecutarlo.
            </p>
          </article>

          <article class="stats-panel panel">
            <div class="panel-header">
              <span class="panel-label">Resumen</span>
            </div>
            <div class="stat-list">
              <div class="stat-item">
                <span class="stat-label">Stack</span>
                <strong id="detailStack">—</strong>
              </div>
              <div class="stat-item">
                <span class="stat-label">Scripts</span>
                <strong id="detailScripts">—</strong>
              </div>
              <div class="stat-item">
                <span class="stat-label">Estado</span>
                <strong id="detailStatus">—</strong>
              </div>
            </div>
          </article>
        </section>

        <section class="panel action-panel">
          <div class="panel-header">
            <span class="panel-label">Acciones</span>
          </div>
          <div id="scriptActions" class="script-actions" aria-label="Scripts disponibles"></div>
        </section>

        <section class="panel console-panel">
          <div class="panel-header terminal-header">
            <span class="panel-label">Salida</span>
            <button id="clearConsoleButton" class="ghost-button" type="button">Limpiar</button>
          </div>
          <pre id="consoleOutput" class="console-output">Selecciona un ejercicio y ejecuta el script para ver el resultado aquí.</pre>
        </section>
      </main>
    </div>

    <script>
      const state = {
        exercises: [],
        selectedExerciseId: null,
        activeProcessPid: null,
        activePoller: null,
      };

      const elements = {
        exerciseList: document.getElementById('exerciseList'),
        exerciseSearch: document.getElementById('exerciseSearch'),
        detailTitle: document.getElementById('detailTitle'),
        detailDescription: document.getElementById('detailDescription'),
        detailStack: document.getElementById('detailStack'),
        detailScripts: document.getElementById('detailScripts'),
        detailStatus: document.getElementById('detailStatus'),
        scriptActions: document.getElementById('scriptActions'),
        consoleOutput: document.getElementById('consoleOutput'),
        runButton: document.getElementById('runButton'),
        openReadmeButton: document.getElementById('openReadmeButton'),
        clearConsoleButton: document.getElementById('clearConsoleButton'),
      };

      function setConsole(message) {
        elements.consoleOutput.textContent = message;
      }

      function clearActivePoller() {
        if (state.activePoller) {
          clearTimeout(state.activePoller);
          state.activePoller = null;
        }
      }

      async function fetchExercises() {
        const response = await fetch('/api/exercises');
        const data = await response.json();
        state.exercises = data.exercises || [];
        if (!state.selectedExerciseId && state.exercises.length) {
          selectExercise(state.exercises[0].id);
        }
        renderExerciseList();
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

        document.querySelectorAll('.exercise-button').forEach((button) => {
          button.addEventListener('click', () => {
            selectExercise(button.dataset.id);
          });
        });
      }

      function selectExercise(exerciseId) {
        state.selectedExerciseId = exerciseId;
        const exercise = state.exercises.find((item) => item.id === exerciseId);
        if (!exercise) {
          return;
        }

        elements.detailTitle.textContent = exercise.title;
        elements.detailDescription.textContent = exercise.description;
        elements.detailStack.textContent = exercise.stack.join(' • ');
        elements.detailScripts.textContent = `${exercise.scripts.length} script(s)`;
        elements.detailStatus.textContent = exercise.status;

        const buttons = exercise.scripts
          .map((script) => `
            <button class="script-button ${script.kind === 'main' ? 'primary' : ''}" type="button" data-script="${script.name}">
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

        elements.runButton.disabled = false;
        elements.runButton.onclick = () => runExercise(exercise.id, exercise.main_script);
        elements.openReadmeButton.onclick = () => window.open(`/readme/${exercise.id}`, '_blank', 'noopener,noreferrer');

        renderExerciseList();
      }

      async function runExercise(exerciseId, scriptName) {
        clearActivePoller();
        setConsole(`Iniciando ${scriptName}...\n\nEsperando respuesta del proceso...`);

        const response = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ exerciseId: exerciseId, scriptName: scriptName }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: 'No se pudo ejecutar el ejercicio.' }));
          setConsole(errorData.error || 'No se pudo ejecutar el ejercicio.');
          return;
        }

        const payload = await response.json();
        state.activeProcessPid = payload.pid;
        pollProcess(payload.pid);
      }

      async function pollProcess(pid) {
        const response = await fetch(`/api/process/${pid}`);
        const data = await response.json();

        if (response.ok && data.output) {
          setConsole(data.output);
        } else if (response.ok) {
          setConsole(`Proceso ${pid} en ejecución...`);
        }

        if (data.status === 'running') {
          state.activePoller = setTimeout(() => pollProcess(pid), 1200);
          return;
        }

        const percentText = data.exitCode === 0 ? 'completado correctamente.' : 'finalizó con errores.';
        setConsole(`${data.output || 'El proceso terminó.'}\n\nEstado: ${percentText}`);
      }

      elements.exerciseSearch.addEventListener('input', renderExerciseList);
      elements.clearConsoleButton.addEventListener('click', () => setConsole('Salida limpia. Selecciona un ejercicio para continuar.'));

      fetchExercises();
    </script>
  </body>
</html>
'''


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.route("/api/exercises")
def api_exercises():
    exercises = [
        build_exercise_record(folder)
        for folder in sorted(ROOT.glob("Exercise_*"))
        if folder.is_dir()
    ]
    return jsonify({"exercises": exercises})


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(silent=True) or {}
    exercise_id = (payload.get("exerciseId") or "").strip()
    script_name = (payload.get("scriptName") or "").strip()

    if not exercise_id:
        return jsonify({"error": "Debes indicar un ejercicio."}), 400

    folder = find_exercise_dir(exercise_id)
    if not folder:
        return jsonify({"error": "No se encontró el ejercicio solicitado."}), 404

    if not script_name:
        script_name = EXERCISE_OVERRIDES.get(folder.name, {}).get("main_script") or next(
            (p.name for p in sorted(folder.glob("*.py")) if p.name.lower() not in {"evaluate.py", "setup_db.py"}),
            "",
        )

    script_path = folder / script_name
    if not script_path.exists():
        return jsonify({"error": f"No se encontró el script {script_name}."}), 404

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{exercise_id}-{uuid.uuid4().hex}.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, script_name],
        cwd=str(folder),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

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
