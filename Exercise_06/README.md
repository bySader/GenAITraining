# Exercise 06 — LangGraph SQL Agent

## Overview
A **LangGraph Workflow** that takes natural language questions, optionally converts them to SQL, queries a SQLite database, and returns grounded natural language answers.

## Workflow Nodes

```
[Router] -> sql_query  -> [SQL Tool] -> [Formatter] -> END
         -> direct_answer            -> [Direct Answer] -> END
         -> rag                      -> [RAG Node] -> END
```

## Database Tables
- `tickets` — IT support tickets (priority, status, owner)
- `employees` — Employee profiles (department, skills)
- `projects` — Projects with team assignments
- `clients` — Client accounts with active projects
- `skill_certifications` — Employee certifications

## Setup

```bash
pip install -r requirements.txt
python setup_db.py          # creates company.db; drops and recreates its tables
copy .env.example .env      # add your GROQ_API_KEY
python workflow.py           # interactive CLI
python evaluate.py           # run automated tests
```

`setup_db.py` replaces the existing tables and their data. Run it only for a new
database or after confirming that replacing its contents is safe. The dashboard
checks the API key, dependencies, and expected database schema before launching
the Exercise 06 workflow.

En el CLI, escribe una pregunta directamente después de `You:` para ejecutarla
en el flujo. Después de cada respuesta aparecerá un menú para iniciar otra
pregunta, cambiar el modo `verbose` o salir. También puedes usar `quit` para
salir directamente.

En el dashboard, las respuestas se presentan en un formato más sencillo: la
consulta SQL generada y la respuesta en lenguaje natural, sin los mensajes del
menú interactivo de la terminal.

## Example Questions

```
What is the most urgent ticket?
Who is assigned to the highest priority ticket?
How many open tickets does Project Phoenix have?
Which employee has the most open tickets?
Show all critical tickets created this month.
How do I escalate a critical ticket?  (RAG)
What is LangGraph?                     (Direct answer)
```

## Challenges Implemented
- ✅ **Challenge 1**: RAG node for KB article queries
- ✅ **Challenge 2**: SQL safety guard (regex + prompt) blocks destructive statements
