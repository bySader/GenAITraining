# Exercise 05 — Key Code Concepts for Presentation
## Tool Calling / Function Calling with LLMs

---

## 1. What Problem Are We Solving?

LLMs are great at language but have a critical limitation:

| Limitation | Example |
|-----------|---------|
| **No real-time data** | Doesn't know today's ticket status |
| **No private data access** | Can't look up employee 1001 |
| **Hallucinations** | May invent ticket numbers or names |

**Tool Calling** gives the LLM "hands" — it can now call external functions and base its answers on real, returned data.

---

## 2. Architecture: The Agentic Loop

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│  LLM receives:                   │
│  - System prompt                 │
│  - User question                 │
│  - Tool schemas (JSON)           │
└──────────────┬──────────────────┘
               │
       ┌───────┴────────┐
       │                │
   Tool call?        Direct answer
       │
       ▼
┌──────────────────┐
│  Tool Executor    │  ← Python function runs
│  + Usage Log      │  ← Timestamp, input, output recorded
└──────┬───────────┘
       │  JSON result
       ▼
┌──────────────────┐
│  LLM receives     │  ← Tool result appended to conversation
│  tool result      │
└──────┬───────────┘
       │
       ▼
  Final Answer
  + Structured JSON output
```

This loop can repeat **multiple times** for complex multi-tool queries.

---

## 3. Tool Definition: The Schema Contract

Every tool is defined in two places:

### 3a. Python function
```python
def get_employee_information(employee_id: str) -> dict:
    """Return employee record by ID."""
    result = EMPLOYEES.get(str(employee_id))
    if result:
        return result
    return {"error": f"Employee ID {employee_id} not found."}
```

### 3b. JSON Schema (tells the LLM what the tool does)
```python
{
    "type": "function",
    "function": {
        "name": "get_employee_information",
        "description": "Retrieve full employee profile by their numeric employee ID.
                        Use when the user asks about an employee, their department,
                        role, email, or status.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The numeric employee ID, e.g. '1001'"
                }
            },
            "required": ["employee_id"]
        }
    }
}
```

**Key insight**: The **description** is what the LLM reads. It must be clear and specific enough that the LLM chooses the right tool automatically.

---

## 4. Tool Registry Pattern

Instead of a chain of if/elif statements, all tools are stored in a simple dictionary:

```python
TOOL_FUNCTIONS: dict[str, callable] = {
    "get_employee_information":  get_employee_information,
    "get_ticket_status":         get_ticket_status,
    "search_knowledge_base":     search_knowledge_base,
    "get_weather":               get_weather,
    "get_tickets_by_employee":   get_tickets_by_employee,
    "calculate":                 calculate,
    "get_department_headcount":  get_department_headcount,
}
```

Dispatching any tool call:
```python
def execute_tool(tool_name: str, tool_args: dict) -> str:
    fn = TOOL_FUNCTIONS.get(tool_name)   # look up by name
    result = fn(**tool_args)             # call with unpacked kwargs
    return json.dumps(result)            # return as JSON string
```

**Why this pattern?**
- Adding a new tool = add one function + one schema entry. No other code changes.
- Fully extensible and maintainable.

---

## 5. The Agentic Loop Code

```python
def run_agent(user_question: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_question},
    ]

    for round_num in range(5):   # max 5 rounds to prevent infinite loops
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",    # ← LLM decides: call tool or answer directly
            temperature=0.1,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── Case 1: LLM answers directly (no tool needed) ──
        if finish_reason == "stop" or not msg.tool_calls:
            return {"tool_used": None, "answer": msg.content}

        # ── Case 2: LLM wants to call tool(s) ──
        messages.append(msg)    # append assistant message with tool_calls

        for tool_call in msg.tool_calls:   # supports multi-tool (Challenge 2)
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            tool_result = execute_tool(fn_name, fn_args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,   # must match the request
                "content":      tool_result,
            })
        # Loop continues → LLM sees tool results → generates final answer
```

**Key design decisions**:
- `tool_choice="auto"`: LLM decides when to call a tool. `"required"` forces a call, `"none"` prevents it.
- `temperature=0.1`: Deterministic tool selection — we want precision, not creativity.
- `max_rounds=5`: Safety guard against runaway loops.

---

## 6. LLM vs Tool Calling: The Critical Difference

| | Standard LLM Response | Tool Calling |
|--|----------------------|-------------|
| **Data source** | Training data (static, may be outdated) | Live function execution |
| **Accuracy** | May hallucinate | Grounded in real data |
| **When to use** | General knowledge questions | Lookups, calculations, external APIs |
| **Example** | "What is AI?" | "What is the status of INC-12345?" |

```
Q: "What is artificial intelligence?"
→ LLM answers directly from training knowledge. No tool needed. ✅

Q: "What is the status of ticket INC-12345?"
→ LLM calls get_ticket_status("INC-12345") → gets real data → answers. ✅
```

---

## 7. Multi-Tool Support (Challenge 2)

The loop naturally handles multiple tool calls in one turn:

```
User: "Get info for employee 1001 and show all their open tickets"

Round 1: LLM decides to call TWO tools:
  → get_employee_information("1001")
  → get_tickets_by_employee("1001", status_filter="open")

Round 2: LLM receives both results → generates combined answer
```

The Groq API returns `msg.tool_calls` as a **list** — the loop iterates all of them.

---

## 8. Tool Usage Log (Challenge 3)

Every tool execution is automatically recorded:

```python
TOOL_LOG: list[dict] = []   # global in-memory log

def execute_tool(tool_name, tool_args):
    result = TOOL_FUNCTIONS[tool_name](**tool_args)

    TOOL_LOG.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "tool":      tool_name,
        "input":     tool_args,
        "output":    result,
    })
    return json.dumps(result)
```

Sample log entry:
```json
{
  "timestamp": "2026-08-04T09:55:12.345678",
  "tool": "get_ticket_status",
  "input": {"ticket_id": "INC-12345"},
  "output": {
    "ticket_id": "INC-12345",
    "title": "VPN connection failure",
    "status": "In Progress",
    "priority": "High",
    "assigned_to": "1001"
  }
}
```

**Real-world uses**: billing, auditing, debugging, compliance.

---

## 9. Structured JSON Output (Step 5)

Every response from `run_agent()` is a structured dictionary:

```python
return {
    "tool_used":   "get_ticket_status",             # or list for multi-tool
    "tools_count": 1,
    "answer":      "Ticket INC-12345 is In Progress with High priority.",
}
```

**Why structured output?**
- Downstream systems can parse `tool_used` to route or log calls
- Easy to integrate with APIs, dashboards, or audit systems
- Enables programmatic evaluation of tool selection accuracy

---

## 10. The System Prompt: Guiding Tool Behavior

```python
SYSTEM_PROMPT = """
You have access to 7 tools for employee, ticket, weather, and KB lookups.

RULES:
1. ONLY call a tool when the user's question genuinely requires real data.
2. For general knowledge questions, answer directly WITHOUT calling any tool.
3. After calling a tool, base your answer EXCLUSIVELY on the data returned.
4. Never invent employee names, ticket numbers, or any other data.
"""
```

**Critical rules**:
- Rule 1 prevents wasteful tool calls on general questions
- Rule 3 prevents hallucination after a tool call
- Rule 4 enforces grounding

---

## 11. The 7 Tools at a Glance

| Tool | Input | Returns | Challenge |
|------|-------|---------|-----------|
| `get_employee_information` | `employee_id` | Full profile | ─ |
| `get_ticket_status` | `ticket_id` | Status + priority | ─ |
| `search_knowledge_base` | `keyword` | Top 3 articles | ─ |
| `get_weather` | `city` | Temp + conditions | ─ |
| `get_tickets_by_employee` | `employee_id`, `status_filter` | Ticket list | ─ |
| `calculate` | `expression` | Numeric result | Challenge 1 |
| `get_department_headcount` | `department` | Count + members | ─ |

---

## 12. Key Takeaways

1. **Tool Calling = LLM + Execution** — the LLM orchestrates but doesn't invent data
2. **Schema descriptions** are critical — bad descriptions = wrong tool selection
3. **Registry pattern** makes tools pluggable — add a function + schema, done
4. **`tool_choice="auto"`** lets the LLM decide when tools are truly needed
5. **Multi-tool** is free — the API already returns a list of tool calls
6. **Logging every call** is essential for auditability and debugging
7. **Structured output** bridges the gap between LLM text and downstream systems
8. **Low temperature** (0.1) is important for reliable, deterministic tool selection

---

## File Structure

```
Exercise_05/
├── agent.py           ← Complete agent: tools, schemas, agentic loop, CLI
├── evaluate.py        ← 15-question automated test suite
├── README.md
├── requirements.txt   ← groq, python-dotenv (no heavy deps needed)
├── .env.example
└── test_results.json  ← Generated after running evaluate.py
```
