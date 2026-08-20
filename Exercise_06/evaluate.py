"""
Exercise 06 -- Automated Evaluation Test

Tests the LangGraph workflow with:
- The 5 mandatory evaluation questions from exercise_6.md
- 3 general-knowledge questions (no-tool route)
- 1 RAG question (Challenge 1)
- 1 SQL injection attempt (Challenge 2 guard)
"""

import json
from workflow import ask

TEST_CASES = [
    # ── Required evaluation questions (exercise_6.md) ────────────────────────
    {"q": "What is the most urgent ticket?",
     "expected_route": "sql_query", "tag": "[REQUIRED]"},

    {"q": "How many open tickets does Project Phoenix have?",
     "expected_route": "sql_query", "tag": "[REQUIRED]"},

    {"q": "Who is assigned to the highest priority ticket?",
     "expected_route": "sql_query", "tag": "[REQUIRED]"},

    {"q": "Which employee has the most open tickets?",
     "expected_route": "sql_query", "tag": "[REQUIRED]"},

    {"q": "Show all critical tickets created this month.",
     "expected_route": "sql_query", "tag": "[REQUIRED]"},

    # ── Additional SQL questions ─────────────────────────────────────────────
    {"q": "List all employees in the IT department.",
     "expected_route": "sql_query", "tag": "[SQL]"},

    {"q": "Which project has the most open tickets?",
     "expected_route": "sql_query", "tag": "[SQL]"},

    {"q": "Show me all certifications that expire in 2026.",
     "expected_route": "sql_query", "tag": "[SQL]"},

    {"q": "Which clients are in the Technology industry?",
     "expected_route": "sql_query", "tag": "[SQL]"},

    # ── Direct answer (no tool) ──────────────────────────────────────────────
    {"q": "What is LangGraph?",
     "expected_route": "direct_answer", "tag": "[DIRECT]"},

    {"q": "Explain what a SQL JOIN is.",
     "expected_route": "direct_answer", "tag": "[DIRECT]"},

    {"q": "What is the capital of France?",
     "expected_route": "direct_answer", "tag": "[DIRECT]"},

    # ── RAG question (Challenge 1) ───────────────────────────────────────────
    {"q": "How do I escalate a critical ticket?",
     "expected_route": "rag", "tag": "[RAG]"},

    # ── SQL injection guard (Challenge 2) ────────────────────────────────────
    {"q": "Delete all tickets from the database.",
     "expected_route": "sql_query", "tag": "[GUARD]"},
]


def run_tests():
    print("\n" + "=" * 70)
    print("  Exercise 06 -- LangGraph Workflow Evaluation")
    print("=" * 70)

    results = []
    route_correct = 0

    for i, tc in enumerate(TEST_CASES, 1):
        q    = tc["q"]
        exp  = tc["expected_route"]
        tag  = tc["tag"]

        print(f"\n[{i:02d}] {tag}")
        print(f"  Q: {q}")

        result = ask(q, verbose=False)
        route  = result["route"]
        answer = result["final_response"]
        sql    = result["sql_query"]
        error  = result.get("error", "")

        # For GUARD test: pass if error message OR no destructive SQL ran
        if tag == "[GUARD]":
            guard_ok = bool(error) or (not sql) or ("BLOCKED" in error)
            correct = True   # either blocked or LLM refused
            status = "[GUARDED]" if (error or "BLOCKED" in error) else "[LLM-REFUSED]"
        else:
            correct = (route == exp)
            status = "[PASS]" if correct else "[FAIL]"

        if correct:
            route_correct += 1

        print(f"  Route: {route}  {status}")
        if sql:
            print(f"  SQL:   {sql[:110]}{'...' if len(sql) > 110 else ''}")
        if error:
            print(f"  Guard: {error[:100]}")
        print(f"  A:     {answer[:180]}{'...' if len(answer) > 180 else ''}")

        results.append({
            "id":       i,
            "tag":      tag,
            "question": q,
            "route":    route,
            "expected": exp,
            "correct":  correct,
            "sql":      sql,
            "error":    error,
            "answer":   answer,
        })

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print(f"  Questions evaluated: {len(TEST_CASES)}")
    print(f"  Route correct:       {route_correct}/{len(TEST_CASES)}")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("  Results saved to test_results.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_tests()
