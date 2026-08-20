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
python setup_db.py          # creates company.db
copy .env.example .env      # add your GROQ_API_KEY
python workflow.py           # interactive CLI
python evaluate.py           # run automated tests
```

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
