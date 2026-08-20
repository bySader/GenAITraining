"""
GEN AI Upskilling Training -- Exercise 06
LangGraph Workflow: NL -> SQL -> Database -> Natural Language Response

Architecture:
  User Question
       |
       v
  [Node 1: Router]
  LLM decides: needs DB query OR can answer directly
       |                |
  "sql_query"     "direct_answer"
       |                |
  [Node 2: SQL Tool]   [Node 4: Direct Answer]
  LLM generates SQL        LLM answers directly
  + safety guard           (no DB call)
  + runs query
  + returns results
       |
       v
  [Node 3: Formatter]
  LLM crafts natural language
  answer from raw SQL results
       |
       v
  Final Response (with structured output)

Challenges implemented:
  - Challenge 1: RAG node (vector search on KB articles from Ex03/04)
  - Challenge 2: SQL safety guard blocks destructive statements
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent / "company.db"
MODEL   = "openai/gpt-oss-120b"

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model=MODEL,
    temperature=0.1,
)

# ─────────────────────────────────────────────────────────────
# DATABASE SCHEMA (injected into LLM prompts so it knows the tables)
# ─────────────────────────────────────────────────────────────
DB_SCHEMA = """
Tables and columns in the company.db SQLite database:

tickets(ticket_id TEXT, title TEXT, priority TEXT, status TEXT,
        owner TEXT, owner_id TEXT, project_id TEXT, created_date TEXT)
  priority values: Critical, High, Medium, Low
  status values:   Open, In Progress, Resolved, Closed

employees(employee_id TEXT, name TEXT, department TEXT, hire_date TEXT,
          project TEXT, skills TEXT)

projects(project_id TEXT, name TEXT, duration TEXT, team TEXT,
         termination_date TEXT, has_open_tickets INTEGER, open_tickets INTEGER)
  team: comma-separated employee_ids

clients(client_id TEXT, client_name TEXT, industry TEXT, country TEXT,
        account_manager TEXT, active_projects TEXT)
  active_projects: comma-separated project_ids

skill_certifications(certification_id TEXT, employee_id TEXT,
                     certification_name TEXT, provider TEXT,
                     issue_date TEXT, expiration_date TEXT)

Priority ordering (most urgent first): Critical > High > Medium > Low
For ordering by urgency, use:
  ORDER BY CASE priority WHEN 'Critical' THEN 1 WHEN 'High' THEN 2
           WHEN 'Medium' THEN 3 WHEN 'Low' THEN 4 END
""".strip()

# ─────────────────────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:       str          # original user question
    route:          str          # "sql_query" | "direct_answer" | "rag"
    sql_query:      str          # generated SQL
    query_result:   str          # raw DB result
    final_response: str          # human-readable answer
    error:          str          # error message if any


# ─────────────────────────────────────────────────────────────
# SQL SAFETY GUARD  (Challenge 2)
# ─────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE|REPLACE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

def is_safe_sql(sql: str) -> bool:
    """Returns True only if the SQL is a pure SELECT statement with no destructive keywords."""
    stripped = sql.strip().lstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if FORBIDDEN_KEYWORDS.search(stripped):
        return False
    return True


# ─────────────────────────────────────────────────────────────
# DATABASE EXECUTION TOOL
# ─────────────────────────────────────────────────────────────
def run_sql(sql: str) -> list[dict]:
    """Execute a SELECT query and return results as list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# NODE 1: ROUTER
# Decides: does this question need a DB query, direct answer, or RAG?
# ─────────────────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """
    Entry node: LLM classifies the question into one of three routes:
      - sql_query:     requires database lookup
      - direct_answer: general knowledge, no DB needed
      - rag:           requires document search (Challenge 1)
    """
    system = f"""
You are a routing assistant for a company internal AI system.

Classify the user's question into exactly ONE of these categories:
- "sql_query":     The question requires data from the company database
                   (tickets, employees, projects, clients, certifications)
- "rag":           The question asks for documentation, procedures, or policies
                   that would be found in knowledge base articles
- "direct_answer": General knowledge question that does not require the database
                   or any document search

Database contains: {DB_SCHEMA[:300]}...

Respond with ONLY one of these exact strings: sql_query | direct_answer | rag
Do NOT add any explanation.
""".strip()

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])

    route = response.content.strip().lower().strip('"').strip("'")
    if route not in ("sql_query", "direct_answer", "rag"):
        route = "sql_query"  # safe default

    return {**state, "route": route}


# ─────────────────────────────────────────────────────────────
# NODE 2: SQL TOOL NODE
# LLM generates SQL -> safety check -> DB execution
# ─────────────────────────────────────────────────────────────
def sql_tool_node(state: AgentState) -> AgentState:
    """
    1. LLM analyzes the question and generates a safe SELECT query
    2. Safety guard validates the SQL (Challenge 2)
    3. Query runs against SQLite; results stored in state
    """
    system = f"""
You are an expert SQLite query generator.

Given the user's question, generate a single valid SQLite SELECT query.

STRICT RULES:
- Output ONLY the SQL query, nothing else. No markdown, no explanation.
- ONLY use SELECT statements. NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, MERGE, or any other destructive statement.
- Use only the tables and columns listed in the schema below.
- Limit results to 20 rows maximum.
- For priority ordering use: ORDER BY CASE priority WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 WHEN 'Low' THEN 4 END

DATABASE SCHEMA:
{DB_SCHEMA}

Output the SQL query only. No markdown code blocks.
""".strip()

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])

    sql = response.content.strip()
    # Strip markdown code fences if LLM added them anyway
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip()

    # Challenge 2: Safety guard
    if not is_safe_sql(sql):
        return {
            **state,
            "sql_query":    sql,
            "query_result": "",
            "error":        f"BLOCKED: Generated SQL contains forbidden operations: {sql[:200]}",
        }

    # Execute query
    try:
        rows = run_sql(sql)
        result_str = json.dumps(rows, ensure_ascii=False, indent=2)
        return {**state, "sql_query": sql, "query_result": result_str, "error": ""}
    except Exception as exc:
        return {**state, "sql_query": sql, "query_result": "", "error": str(exc)}


# ─────────────────────────────────────────────────────────────
# NODE 3: FORMATTER NODE
# Converts raw SQL results into a polished natural language response
# ─────────────────────────────────────────────────────────────
def formatter_node(state: AgentState) -> AgentState:
    """
    Takes the raw JSON query result and crafts a clear, professional
    natural language answer grounded entirely in the DB data.
    """
    if state.get("error"):
        return {**state, "final_response": f"I encountered an issue: {state['error']}"}

    if not state.get("query_result") or state["query_result"] == "[]":
        return {**state, "final_response": "No results were found in the database for your query."}

    system = """
You are a professional business assistant.

Convert the following database query results into a clear, concise,
and well-formatted natural language response.

Rules:
- Base your answer ONLY on the data provided. Do NOT add information not in the results.
- Be conversational and professional.
- If there are multiple rows, present them as a numbered list or a brief table summary.
- Keep the response concise (under 200 words unless a list requires more).
""".strip()

    user_msg = f"""
User question: {state['question']}

SQL query used: {state['sql_query']}

Database results:
{state['query_result']}

Please provide a natural language answer based only on the data above.
""".strip()

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    return {**state, "final_response": response.content.strip()}


# ─────────────────────────────────────────────────────────────
# NODE 4: DIRECT ANSWER NODE
# For general-knowledge questions that don't need the database
# ─────────────────────────────────────────────────────────────
def direct_answer_node(state: AgentState) -> AgentState:
    """Answers general-knowledge questions directly without DB access."""
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the question concisely and accurately."),
        HumanMessage(content=state["question"]),
    ])
    return {**state, "sql_query": "", "query_result": "", "final_response": response.content.strip()}


# ─────────────────────────────────────────────────────────────
# NODE 5: RAG NODE (Challenge 1)
# Performs simple in-memory knowledge base search
# ─────────────────────────────────────────────────────────────
KB_ARTICLES = [
    {"id": "KB-001", "title": "Password Reset Procedure",
     "content": "To reset your password: 1) Go to IT portal. 2) Click Forgot Password. 3) Enter employee email. 4) Follow the link sent. 5) Set a new password with at least 12 characters including uppercase, lowercase, number, and symbol."},
    {"id": "KB-002", "title": "VPN Setup Guide",
     "content": "To configure VPN: 1) Download the VPN client from IT portal. 2) Install and launch. 3) Enter server address vpn.company.com. 4) Use company credentials. 5) Enable 2FA when prompted."},
    {"id": "KB-003", "title": "Onboarding Checklist",
     "content": "New employee onboarding: Complete I-9 form, set up company email, install required software from IT portal, attend orientation, meet manager for 90-day plan review."},
    {"id": "KB-004", "title": "Expense Reimbursement Policy",
     "content": "Submit expenses within 30 days of purchase. Receipts required for amounts over $25. Use finance.company.com portal. Manager approval required for expenses over $500."},
    {"id": "KB-005", "title": "Remote Work Policy",
     "content": "Employees may work remotely up to 3 days per week with manager approval. Core hours are 10am-3pm local time. VPN required for all company system access."},
    {"id": "KB-006", "title": "IT Ticket Escalation Process",
     "content": "Critical tickets must be escalated to IT manager within 1 hour. High priority tickets within 4 hours. Medium within 24 hours. Low within 5 business days."},
]

def rag_node(state: AgentState) -> AgentState:
    """
    Challenge 1: RAG node. Searches the knowledge base for relevant articles
    using keyword matching, then uses LLM to synthesize a grounded answer.
    """
    q_lower = state["question"].lower()
    scored = []
    for article in KB_ARTICLES:
        score = sum(
            (3 if kw in article["title"].lower() else 0) +
            (1 if kw in article["content"].lower() else 0)
            for kw in q_lower.split()
            if len(kw) > 3
        )
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_articles = [a for _, a in scored[:2]]

    if not top_articles:
        return {**state, "final_response": "I could not find relevant documentation for your question. Please contact the IT helpdesk."}

    context = "\n\n".join(f"[{a['id']}] {a['title']}\n{a['content']}" for a in top_articles)

    system = f"""
You are a company knowledge base assistant.
Answer the user's question using ONLY the documentation excerpts below.
If the excerpts do not contain enough information, say "I don't have that information in my knowledge base."

--- KNOWLEDGE BASE ---
{context}
--- END ---
""".strip()

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])

    sources = ", ".join(a["id"] for a in top_articles)
    return {
        **state,
        "sql_query":    "",
        "query_result": context,
        "final_response": f"{response.content.strip()}\n\n(Sources: {sources})",
    }


# ─────────────────────────────────────────────────────────────
# ROUTING FUNCTION (conditional edge)
# ─────────────────────────────────────────────────────────────
def route_decision(state: AgentState) -> Literal["sql_tool", "direct_answer", "rag"]:
    """Reads the route set by router_node and directs the graph."""
    route = state.get("route", "sql_query")
    if route == "direct_answer":
        return "direct_answer"
    elif route == "rag":
        return "rag"
    else:
        return "sql_tool"


# ─────────────────────────────────────────────────────────────
# BUILD THE LANGGRAPH WORKFLOW
# ─────────────────────────────────────────────────────────────
def build_workflow():
    """
    Constructs the LangGraph StateGraph:

        [router] --sql_query--> [sql_tool] --> [formatter] --> END
                 --direct----> [direct_answer] -----------> END
                 --rag-------> [rag_node] -----------------> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router",        router_node)
    graph.add_node("sql_tool",      sql_tool_node)
    graph.add_node("formatter",     formatter_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("rag",           rag_node)

    # Set entry point
    graph.set_entry_point("router")

    # Conditional edge from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "sql_tool":     "sql_tool",
            "direct_answer":"direct_answer",
            "rag":          "rag",
        }
    )

    # Linear edges
    graph.add_edge("sql_tool",      "formatter")
    graph.add_edge("formatter",     END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("rag",           END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
app = build_workflow()

def ask(question: str, verbose: bool = False) -> dict:
    """
    Run the workflow for a single question.
    Returns structured dict with route, sql_query, and final_response.
    """
    initial_state: AgentState = {
        "question":       question,
        "route":          "",
        "sql_query":      "",
        "query_result":   "",
        "final_response": "",
        "error":          "",
    }

    result = app.invoke(initial_state)

    if verbose:
        print(f"  Route:    {result['route']}")
        if result["sql_query"]:
            print(f"  SQL:      {result['sql_query']}")
        if result.get("error"):
            print(f"  Error:    {result['error']}")

    return {
        "route":          result["route"],
        "sql_query":      result["sql_query"],
        "query_result":   result["query_result"],
        "final_response": result["final_response"],
        "error":          result.get("error", ""),
    }


# ─────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  GEN AI Upskilling -- Exercise 06: LangGraph SQL Agent")
    print("=" * 65)
    print("  Ask anything about tickets, employees, projects, clients.")
    print("  Commands: 'verbose on/off' | 'quit'")
    print("=" * 65 + "\n")

    verbose = False

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif question.lower() == "verbose on":
            verbose = True
            print("  Verbose ON.\n")
            continue
        elif question.lower() == "verbose off":
            verbose = False
            print("  Verbose OFF.\n")
            continue

        print("\n[Workflow running...]\n")
        result = ask(question, verbose=verbose)

        if verbose:
            print()
        print(f"Route: {result['route']}")
        if result["sql_query"]:
            print(f"SQL:   {result['sql_query'][:120]}{'...' if len(result['sql_query']) > 120 else ''}")
        print(f"\nAnswer:\n{result['final_response']}\n")
        print("-" * 65)


if __name__ == "__main__":
    main()
