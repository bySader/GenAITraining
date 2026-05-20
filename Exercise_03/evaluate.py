"""
Exercise 03 — Automated test / demo script

Runs a fixed set of Q&A pairs through the RAG pipeline to verify:
  1. Grounded answers (questions the KB should know)
  2. Out-of-scope rejection (questions the KB should NOT answer)
  3. JSON output for easy evaluation
"""

import json
from pathlib import Path
from rag_engine import get_or_build_index, ask_rag

# ── Test cases ──────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    # In-scope — should answer from KB
    {"question": "How do I configure VPN access for remote workers?",          "expected_scope": "in"},
    {"question": "What steps do I follow to reset a forgotten PIN?",           "expected_scope": "in"},
    {"question": "How can I set up company email on my mobile device?",        "expected_scope": "in"},
    {"question": "How do I troubleshoot issues with Microsoft Office?",        "expected_scope": "in"},
    {"question": "How do I create a backup of important files?",               "expected_scope": "in"},
    {"question": "How do I reset a jammed printer?",                           "expected_scope": "in"},
    {"question": "How do I set up a conference call on Cisco Webex?",          "expected_scope": "in"},
    {"question": "How do I configure email on an Android device?",             "expected_scope": "in"},

    # Out-of-scope — should return "I don't know" style response
    {"question": "What is the weather like in Mexico City today?",              "expected_scope": "out"},
    {"question": "Who won the last FIFA World Cup?",                           "expected_scope": "out"},
    {"question": "What is the best recipe for chocolate cake?",                "expected_scope": "out"},
    {"question": "Tell me about the history of the Roman Empire.",             "expected_scope": "out"},
]


def run_tests():
    print("\n" + "=" * 65)
    print("  Exercise 03 -- RAG Evaluation Test")
    print("=" * 65)

    index, items, embed_model = get_or_build_index()
    results = []

    for i, tc in enumerate(TEST_QUESTIONS, 1):
        q = tc["question"]
        scope = tc["expected_scope"]
        tag = "[IN-SCOPE] " if scope == "in" else "[OUT-SCOPE]"

        print(f"\n[{i:02d}] {tag}")
        print(f"  Q: {q}")

        answer = ask_rag(q, index, items, embed_model, verbose=False)
        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        results.append({
            "id":             i,
            "expected_scope": scope,
            "question":       q,
            "answer":         answer,
        })

    # Save results
    out_path = Path(__file__).parent / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"  [OK] {len(results)} test cases processed")
    print(f"  [SAVED] Results saved to: test_results.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_tests()
