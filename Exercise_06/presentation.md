# Exercise 06 — Key Code Concepts for Presentation
## LangGraph Workflow: NL → SQL → Database → Natural Language Response

---

## 1. What Problem Are We Solving?

Users need to query a **private company database** using natural language, without knowing SQL. Classic approaches fail because:

| Approach | Problem |
|----------|---------|
| Direct LLM answer | Hallucinations — invents ticket numbers, employee names |
| Hardcoded SQL | Not flexible enough for ad-hoc business questions |
| Simple tool calling (Ex05) | No structured flow control, no state passing between steps |

**LangGraph Workflow** solves this by providing a **controllable graph of nodes** where state flows explicitly, each step has a clear responsibility, and routing is conditional.

---

## 2. Architecture: The LangGraph Workflow

```
User Question
      |
      v
[Node 1: ROUTER]
LLM classifies: sql_query | direct_answer | rag
      |              |              |
      v              v              v
[Node 2: SQL   [Node 4: DIRECT  [Node 5: RAG]
  TOOL]          ANSWER]         KB search +
LLM generates   LLM answers     LLM synthesis
SQL + safety    from training    (Challenge 1)
guard + runs    knowledge
query
      |
      v
[Node 3: FORMATTER]
LLM converts raw JSON
results into natural
language answer
      |
      v
  Final Response
```

**Why LangGraph instead of a simple function chain?**
- State is **explicit** — every node reads from and writes to `AgentState`
- Routing is **conditional** — the graph branches based on the question type
- Nodes are **isolated and testable** individually
- The graph is **visualizable and debuggable**

---

## 3. LangGraph State: The Backbone

```python
class AgentState(TypedDict):
    question:       str    # original user question (never changes)
    route:          str    # set by router: "sql_query" | "direct_answer" | "rag"
    sql_query:      str    # generated SQL (set by sql_tool_node)
    query_result:   str    # raw DB JSON result (set by sql_tool_node)
    final_response: str    # polished answer (set by formatter or direct/rag)
    error:          str    # error message if something went wrong
```

**Key design principle**: The state travels through the entire graph. Each node receives the current state and returns a **modified copy** — nodes are pure functions of state.

```python
def router_node(state: AgentState) -> AgentState:
    # ... call LLM ...
    return {**state, "route": route}   # return modified state
```

---

## 4. Node 1: Router — Conditional Branching

```python
def router_node(state: AgentState) -> AgentState:
    system = """
    Classify the user's question into exactly ONE of:
    - "sql_query":     requires data from the company database
    - "rag":           asks for documentation or procedures
    - "direct_answer": general knowledge, no DB or docs needed
    Respond with ONLY one of these exact strings.
    """
    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content=state["question"])])
    route = response.content.strip().lower()
    return {**state, "route": route}
```

**Then the conditional edge:**
```python
graph.add_conditional_edges(
    "router",
    route_decision,              # function that reads state["route"]
    {
        "sql_tool":      "sql_tool",
        "direct_answer": "direct_answer",
        "rag":           "rag",
    }
)
```

**Why a separate router?** The LLM should not invoke tools unnecessarily — the exercise requirement. A dedicated routing step prevents wasted DB calls on general questions.

---

## 5. Node 2: SQL Tool — LLM-Generated SQL + Safety Guard

### SQL Generation
```python
system = f"""
You are an expert SQLite query generator.
STRICT RULES:
- Output ONLY the SQL query, nothing else.
- ONLY use SELECT statements.
- NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER...
DATABASE SCHEMA:
{DB_SCHEMA}
"""
response = llm.invoke([system_msg, HumanMessage(content=state["question"])])
sql = response.content.strip()
```

### Challenge 2: SQL Safety Guard
```python
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE|...)\b",
    re.IGNORECASE,
)

def is_safe_sql(sql: str) -> bool:
    if not sql.strip().upper().startswith("SELECT"):
        return False
    if FORBIDDEN_KEYWORDS.search(sql):
        return False
    return True
```

**Two-layer protection:**
1. System prompt tells the LLM to only generate SELECT
2. Regex guard validates the output before it ever touches the database

**Result**: Even if a malicious user says "Delete all tickets", the LLM generates a destructive statement, which is caught and blocked with an error message.

---

## 6. Priority Ordering in SQL

SQL doesn't know that "Critical" > "High" > "Medium" > "Low" alphabetically. The LLM is instructed to use a CASE expression:

```sql
SELECT ticket_id, title, priority, status
FROM tickets
WHERE status = 'Open'
ORDER BY
  CASE priority
    WHEN 'Critical' THEN 1
    WHEN 'High'     THEN 2
    WHEN 'Medium'   THEN 3
    WHEN 'Low'      THEN 4
  END
LIMIT 1
```

This is part of the `DB_SCHEMA` prompt — the LLM learns the correct ordering convention from the schema description.

---

## 7. Node 3: Formatter — Grounded Natural Language Response

```python
def formatter_node(state: AgentState) -> AgentState:
    user_msg = f"""
    User question: {state['question']}
    SQL query used: {state['sql_query']}
    Database results: {state['query_result']}

    Provide a natural language answer based ONLY on the data above.
    """
    response = llm.invoke([system_msg, HumanMessage(content=user_msg)])
    return {**state, "final_response": response.content.strip()}
```

**Why a separate formatter node?**
- Separation of concerns: SQL execution ≠ natural language synthesis
- The formatter sees both the question AND the raw data
- Makes the answer grounded — the LLM cannot hallucinate because the data is right there
- Easy to swap formatting style (markdown, JSON, bullet points) without touching the SQL logic

---

## 8. Node 4: Direct Answer — No Tool Needed

```python
def direct_answer_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content="Answer the question concisely and accurately."),
        HumanMessage(content=state["question"]),
    ])
    return {**state, "sql_query": "", "final_response": response.content.strip()}
```

Example triggers:
- "What is LangGraph?" → direct_answer
- "Explain what a SQL JOIN is." → direct_answer
- "What is the capital of France?" → direct_answer

No DB call, no SQL generation — the LLM answers from training knowledge.

---

## 9. Node 5: RAG Node (Challenge 1)

```python
def rag_node(state: AgentState) -> AgentState:
    # 1. Keyword search across KB articles
    scored = [(score, article) for article in KB_ARTICLES if score > 0]
    top_articles = sorted(scored, reverse=True)[:2]

    # 2. Build context block
    context = "\n\n".join(f"[{a['id']}] {a['title']}\n{a['content']}"
                          for a in top_articles)

    # 3. Grounded LLM answer
    system = f"Answer using ONLY these KB articles:\n{context}"
    response = llm.invoke([system_msg, HumanMessage(content=state["question"])])
    return {**state, "final_response": f"{answer}\n\n(Sources: KB-001, KB-006)"}
```

**What it handles**: "How do I escalate a critical ticket?", "What is the VPN setup procedure?", etc. — questions about company policies that live in documentation, not in the database.

---

## 10. Building and Compiling the Graph

```python
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("router",        router_node)
graph.add_node("sql_tool",      sql_tool_node)
graph.add_node("formatter",     formatter_node)
graph.add_node("direct_answer", direct_answer_node)
graph.add_node("rag",           rag_node)

# Entry point
graph.set_entry_point("router")

# Conditional routing from router
graph.add_conditional_edges("router", route_decision,
    {"sql_tool": "sql_tool", "direct_answer": "direct_answer", "rag": "rag"})

# Linear edges
graph.add_edge("sql_tool",      "formatter")
graph.add_edge("formatter",     END)
graph.add_edge("direct_answer", END)
graph.add_edge("rag",           END)

app = graph.compile()    # returns a Runnable
```

`app.invoke(initial_state)` runs the entire graph and returns the final state.

---

## 11. Database Schema Design

The SQLite database (`company.db`) has **5 tables** with foreign key relationships:

```
tickets ──────────────────────┐
  ticket_id (PK)              │
  priority (Critical/High/...) │
  status (Open/In Progress/...)│
  owner_id ──> employees      │
  project_id ──> projects     │
                              │
employees                     │
  employee_id (PK)            │
  name, department            │
  skills (comma-separated)    │
                              │
projects ◄────────────────────┘
  project_id (PK)
  team (comma-separated EMP IDs)
  has_open_tickets, open_tickets

clients
  client_id (PK)
  account_manager (name ref)
  active_projects (comma-separated PRJ IDs)

skill_certifications
  certification_id (PK)
  employee_id ──> employees
  issue_date, expiration_date
```

---

## 12. Answering the 5 Required Questions

| Question | SQL Pattern |
|----------|------------|
| Most urgent ticket | `ORDER BY CASE priority... LIMIT 1` |
| Open tickets in Phoenix | `WHERE project_id='PRJ-001' AND status='Open'` |
| Owner of highest priority ticket | `JOIN employees ON owner_id` |
| Employee with most open tickets | `GROUP BY owner ORDER BY COUNT(*) DESC LIMIT 1` |
| Critical tickets this month | `WHERE priority='Critical' AND created_date LIKE '2026-08-%'` |

---

## 13. Key Takeaways

1. **LangGraph State** = the shared memory that flows through every node — explicit, typed, debuggable
2. **Conditional edges** = the routing logic lives in the graph, not inside node functions
3. **LLM-generated SQL** = the bridge between natural language and structured data
4. **SQL safety guard** (two layers: prompt + regex) = non-negotiable for any DB-connected agent
5. **Formatter node** = separates data retrieval from natural language synthesis — both are grounded
6. **Router prevents tool overuse** — general questions never hit the database
7. **RAG extends the workflow** to handle documentation queries alongside database queries

---

## File Structure

```
Exercise_06/
├── workflow.py      <- LangGraph workflow: 5 nodes, conditional routing
├── setup_db.py      <- Creates and seeds company.db (run once)
├── evaluate.py      <- 14-question automated test suite
├── company.db       <- SQLite database (5 tables, 51 rows)
├── presentation.md  <- This document
├── requirements.txt <- langgraph, langchain, langchain-groq, python-dotenv
├── .env.example
└── README.md
```
