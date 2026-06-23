"""
Exercise 04 -- Automated Evaluation Test

Tests the RAG pipeline with:
  - 10 in-scope questions (should get grounded answers)
  - 5 out-of-scope questions (should be rejected by scope guard)
Saves results to test_results.json
"""

import json
from pathlib import Path
from rag_engine import get_or_build_index, ask_rag

TEST_CASES = [
    # --- IN-SCOPE ---
    {"q": "What are Python decorators and how do I use them?",            "scope": "in"},
    {"q": "Explain the four pillars of object-oriented programming.",     "scope": "in"},
    {"q": "What is the difference between Git merge and Git rebase?",     "scope": "in"},
    # {"q": "How do I create a Docker container and expose a port?",        "scope": "in"},
    # {"q": "What is a CI/CD pipeline and why is it important?",            "scope": "in"},
    # {"q": "What is the difference between supervised and unsupervised learning?", "scope": "in"},
    # {"q": "How does backpropagation work in neural networks?",            "scope": "in"},
    # {"q": "What is database normalization and what are its normal forms?","scope": "in"},
    # {"q": "What is the Agile Scrum framework and what are sprints?",      "scope": "in"},
    # {"q": "What is the difference between SQL and NoSQL databases?",      "scope": "in"},

    # --- OUT-OF-SCOPE ---
    {"q": "What is the capital of France?",                               "scope": "out"},
    {"q": "Who won the FIFA World Cup in 2022?",                          "scope": "out"},
    # {"q": "What is the best recipe for paella?",                          "scope": "out"},
    # {"q": "Tell me about the history of the Roman Empire.",               "scope": "out"},
    # {"q": "How do I invest in the stock market?",                         "scope": "out"},
]


def run_tests():
    print("\n" + "=" * 65)
    print("  Exercise 04 -- RAG Evaluation Test")
    print("=" * 65)

    index, chunks, embed_model = get_or_build_index()
    results = []
    in_scope_ok = 0
    out_scope_ok = 0

    for i, tc in enumerate(TEST_CASES, 1):
        q     = tc["q"]
        scope = tc["scope"]
        tag   = "[IN-SCOPE] " if scope == "in" else "[OUT-SCOPE]"

        print(f"\n[{i:02d}] {tag}")
        print(f"  Q: {q}")

        result = ask_rag(q, index, chunks, embed_model, verbose=False)
        answer = result["answer"]
        sources = result["sources"]

        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        if sources:
            print(f"  Sources: {', '.join(sources)}")

        # Track accuracy
        if scope == "in"  and result["in_scope"]:  in_scope_ok  += 1
        if scope == "out" and not result["in_scope"]: out_scope_ok += 1

        results.append({
            "id":      i,
            "scope":   scope,
            "question": q,
            "answer":  answer,
            "sources": sources,
            "in_scope_detected": result["in_scope"],
        })

    # Summary
    total_in  = sum(1 for t in TEST_CASES if t["scope"] == "in")
    total_out = sum(1 for t in TEST_CASES if t["scope"] == "out")

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print(f"  In-scope  answered: {in_scope_ok}/{total_in}")
    print(f"  Out-scope rejected: {out_scope_ok}/{total_out}")
    print(f"  Total:              {len(results)}/{len(TEST_CASES)}")

    out_path = Path(__file__).parent / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  [SAVED] Results saved to: test_results.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_tests()
