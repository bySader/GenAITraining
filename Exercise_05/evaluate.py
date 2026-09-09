"""
Exercise 05 -- Automated Evaluation Test

Tests tool selection accuracy, structured output, and no-tool cases.
"""

import json
from typing import Any, Callable, cast
from agent import run_agent as _run_agent, TOOL_LOG as _TOOL_LOG

TOOL_LOG: list[dict[str, Any]] = cast(list[dict[str, Any]], _TOOL_LOG)

run_agent: Callable[[str, bool], dict[str, Any]] = cast(
    Callable[[str, bool], dict[str, Any]], _run_agent
)

TEST_CASES = [
    # ── Tool-required cases ───────────────────────────────────────────────
    {"question": "What is the status of ticket INC-12345?",
     "expected_tool": "get_ticket_status",        "requires_tool": True},

    # ── Challenge 2: Multi-tool ──────────────────────────────────────────────────
    {"question": "Get info for employee 1001 and show all their open tickets",
     "expected_tool": "multi",                    "requires_tool": True},

    # ── No-tool cases (direct LLM knowledge) ────────────────────────────────────
    {"question": "What is artificial intelligence?",
     "expected_tool": None,                       "requires_tool": False},
]


def run_tests():
    print("\n" + "=" * 70)
    print("  Exercise 05 -- Tool Calling Evaluation")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    correct_tool = 0
    correct_no_tool = 0
    total_tool_cases   = sum(1 for t in TEST_CASES if t["requires_tool"])
    total_no_tool_cases = sum(1 for t in TEST_CASES if not t["requires_tool"])

    for i, tc in enumerate(TEST_CASES, 1):
        q: str       = cast(str, tc["question"])
        req_tool     = tc["requires_tool"]
        expected_tool = tc["expected_tool"]
        tag = "[TOOL]    " if req_tool else "[NO-TOOL]"

        print(f"\n[{i:02d}] {tag}")
        print(f"  Q: {q}")

        result = run_agent(q, False)
        answer = result["answer"]
        tool_used = result["tool_used"]
        tools_count = result["tools_count"]

        # Evaluate correctness
        if req_tool:
            if expected_tool == "multi":
                correct = tools_count >= 2
            else:
                if isinstance(tool_used, list):
                    correct = expected_tool in tool_used
                else:
                    correct = tool_used == expected_tool
            if correct:
                correct_tool += 1
        else:
            correct = (tools_count == 0)
            if correct:
                correct_no_tool += 1

        status = "[PASS]" if correct else "[FAIL]"
        print(f"  Tool: {tool_used}  {status}")
        print(f"  A: {answer[:180]}{'...' if len(answer) > 180 else ''}")

        results.append({
            "id":            i,
            "requires_tool": req_tool,
            "expected_tool": expected_tool,
            "tool_used":     tool_used,
            "tools_count":   tools_count,
            "correct":       correct,
            "question":      q,
            "answer":        answer,
        })

    # Summary
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print(f"  Tool cases correct:    {correct_tool}/{total_tool_cases}")
    print(f"  No-tool cases correct: {correct_no_tool}/{total_no_tool_cases}")
    print(f"  Total:                 {correct_tool + correct_no_tool}/{len(TEST_CASES)}")
    print(f"  Tool calls logged:     {len(TOOL_LOG)}")

    # Save results
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": {"total": len(TEST_CASES),
                               "tool_cases_correct": correct_tool,
                               "no_tool_cases_correct": correct_no_tool},
                   "tool_log": TOOL_LOG,
                   "results": results}, f, ensure_ascii=False, indent=2)

    print("  Results saved to test_results.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_tests()
