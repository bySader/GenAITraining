"""
GEN AI Upskilling Training -- Exercise 05
Tool Calling / Function Calling with Groq

Architecture:
  1. Mock tools defined as Python functions with rich in-memory data
  2. Tool schemas registered in Groq's native function-calling format
  3. Agentic loop: LLM decides → tool runs → result fed back → final answer
  4. Structured JSON output for every response
  5. Tool usage log for every execution (Challenge 3)
  6. Multi-tool support: LLM may call several tools per turn (Challenge 2)
  7. Calculator tool (Challenge 1)
  8. Graceful fallback: direct answer when no tool is needed
"""

import os
import json
import math
import datetime
from typing import Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"   # supports Groq native tool calling

# ─────────────────────────────────────────────────────────────
# MOCK DATA  (in-memory dictionaries -- no external DB needed)
# ─────────────────────────────────────────────────────────────

EMPLOYEES = {
    "1001": {"employee_id": "1001", "name": "Alice Martinez",  "department": "IT",        "role": "Senior Developer",    "email": "alice@company.com",  "status": "Active"},
    "1002": {"employee_id": "1002", "name": "Bob Johnson",     "department": "HR",        "role": "HR Specialist",       "email": "bob@company.com",    "status": "Active"},
    "1003": {"employee_id": "1003", "name": "Carol Williams",  "department": "Finance",   "role": "Financial Analyst",   "email": "carol@company.com",  "status": "Active"},
    "1004": {"employee_id": "1004", "name": "David Chen",      "department": "IT",        "role": "DevOps Engineer",     "email": "david@company.com",  "status": "On Leave"},
    "1005": {"employee_id": "1005", "name": "Emma Rodriguez",  "department": "Marketing", "role": "Marketing Manager",   "email": "emma@company.com",   "status": "Active"},
}

TICKETS = {
    "INC-12345": {"ticket_id": "INC-12345", "title": "VPN connection failure",          "status": "In Progress",  "priority": "High",   "assigned_to": "1001", "created": "2026-07-28", "category": "Network"},
    "INC-12346": {"ticket_id": "INC-12346", "title": "Outlook not syncing emails",      "status": "Open",         "priority": "Medium", "assigned_to": "1004", "created": "2026-07-30", "category": "Email"},
    "INC-12347": {"ticket_id": "INC-12347", "title": "Laptop screen flickering",        "status": "Resolved",     "priority": "Low",    "assigned_to": "1001", "created": "2026-07-25", "category": "Hardware"},
    "INC-12348": {"ticket_id": "INC-12348", "title": "Cannot access shared drive",      "status": "Open",         "priority": "High",   "assigned_to": "1001", "created": "2026-08-01", "category": "Storage"},
    "INC-12349": {"ticket_id": "INC-12349", "title": "Printer not found on network",    "status": "In Progress",  "priority": "Medium", "assigned_to": "1004", "created": "2026-08-02", "category": "Hardware"},
    "INC-12350": {"ticket_id": "INC-12350", "title": "Password reset request",           "status": "Resolved",     "priority": "Low",    "assigned_to": "1002", "created": "2026-07-29", "category": "Access"},
}

KNOWLEDGE_BASE = [
    {"id": "KB-001", "title": "Password Reset Procedure",    "category": "Access Management",
     "content": "To reset your password: 1) Go to the IT portal. 2) Click 'Forgot Password'. 3) Enter your employee email. 4) Follow the link sent to your registered email. 5) Set a new password of at least 12 characters including upper, lower, number, and symbol."},
    {"id": "KB-002", "title": "VPN Setup Guide",             "category": "Network",
     "content": "To configure VPN: 1) Download the VPN client from the IT portal. 2) Install and launch. 3) Enter server address vpn.company.com. 4) Use your company credentials. 5) Enable 2FA when prompted."},
    {"id": "KB-003", "title": "Onboarding Checklist",        "category": "HR",
     "content": "New employee onboarding: Complete I-9 form, set up company email, install required software (see IT portal), attend orientation session, meet with your manager to review 90-day plan."},
    {"id": "KB-004", "title": "Expense Reimbursement Policy","category": "Finance",
     "content": "Submit expenses within 30 days of purchase. Receipts are required for amounts over $25. Use the expense portal at finance.company.com. Manager approval required for expenses over $500."},
    {"id": "KB-005", "title": "Remote Work Policy",          "category": "HR",
     "content": "Employees may work remotely up to 3 days per week with manager approval. Core hours are 10am-3pm local time. VPN must be used for all company system access. Equipment policy: company laptop required."},
    {"id": "KB-006", "title": "Software Installation Guide", "category": "IT",
     "content": "Approved software is available via the IT self-service portal. Submit a request for unlisted software. Approval may take 2-3 business days. Never install unlicensed software on company devices."},
    {"id": "KB-007", "title": "Data Backup Procedures",      "category": "IT",
     "content": "All company data must be stored on OneDrive or SharePoint. Local backups are not sufficient. Automated backups run nightly. To restore files, contact IT helpdesk or use OneDrive version history."},
]

WEATHER_DATA = {
    "monterrey":   {"city": "Monterrey",   "temp_c": 35, "temp_f": 95,  "condition": "Sunny",        "humidity": "45%"},
    "mexico city": {"city": "Mexico City", "temp_c": 18, "temp_f": 64,  "condition": "Partly Cloudy","humidity": "70%"},
    "guadalajara": {"city": "Guadalajara", "temp_c": 22, "temp_f": 72,  "condition": "Clear",        "humidity": "55%"},
    "new york":    {"city": "New York",    "temp_c": 28, "temp_f": 82,  "condition": "Humid",        "humidity": "80%"},
    "london":      {"city": "London",      "temp_c": 15, "temp_f": 59,  "condition": "Rainy",        "humidity": "85%"},
    "tokyo":       {"city": "Tokyo",       "temp_c": 30, "temp_f": 86,  "condition": "Clear",        "humidity": "60%"},
    "paris":       {"city": "Paris",       "temp_c": 20, "temp_f": 68,  "condition": "Cloudy",       "humidity": "65%"},
}

# Tool usage log (Challenge 3)
TOOL_LOG: list[dict] = []


# ─────────────────────────────────────────────────────────────
# TOOL 1  get_employee_information
# ─────────────────────────────────────────────────────────────
def get_employee_information(employee_id: str) -> dict:
    """Return employee record by ID."""
    result = EMPLOYEES.get(str(employee_id))
    if result:
        return result
    return {"error": f"Employee ID {employee_id} not found.", "available_ids": list(EMPLOYEES.keys())}


# ─────────────────────────────────────────────────────────────
# TOOL 2  get_ticket_status
# ─────────────────────────────────────────────────────────────
def get_ticket_status(ticket_id: str) -> dict:
    """Return IT support ticket details by ticket ID."""
    result = TICKETS.get(ticket_id.upper())
    if result:
        return result
    return {"error": f"Ticket {ticket_id} not found.", "available_tickets": list(TICKETS.keys())}


# ─────────────────────────────────────────────────────────────
# TOOL 3  search_knowledge_base
# ─────────────────────────────────────────────────────────────
def search_knowledge_base(keyword: str) -> dict:
    """Search the IT/HR knowledge base by keyword. Returns best matches."""
    kw = keyword.lower()
    matches = []
    for article in KNOWLEDGE_BASE:
        score = 0
        if kw in article["title"].lower():
            score += 3
        if kw in article["content"].lower():
            score += 1
        if kw in article["category"].lower():
            score += 2
        if score > 0:
            matches.append({"score": score, **article})

    matches.sort(key=lambda x: x["score"], reverse=True)
    if matches:
        top = matches[:3]
        for m in top:
            del m["score"]
        return {"keyword": keyword, "results_found": len(matches), "top_results": top}
    return {"keyword": keyword, "results_found": 0, "message": "No articles matched. Try different keywords."}


# ─────────────────────────────────────────────────────────────
# TOOL 4  get_weather
# ─────────────────────────────────────────────────────────────
def get_weather(city: str) -> dict:
    """Return current weather information for a city."""
    result = WEATHER_DATA.get(city.lower())
    if result:
        return {**result, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    available = list({v["city"] for v in WEATHER_DATA.values()})
    return {"error": f"Weather data for '{city}' not available.", "available_cities": available}


# ─────────────────────────────────────────────────────────────
# TOOL 5  get_tickets_by_employee
# ─────────────────────────────────────────────────────────────
def get_tickets_by_employee(employee_id: str, status_filter: str = "all") -> dict:
    """Return all IT tickets assigned to a specific employee. Optional status filter: open, resolved, in_progress, all."""
    assigned = [t for t in TICKETS.values() if t["assigned_to"] == str(employee_id)]
    if status_filter.lower() != "all":
        # normalize filter
        status_map = {"open": "Open", "resolved": "Resolved", "in_progress": "In Progress", "in progress": "In Progress"}
        status_target = status_map.get(status_filter.lower(), status_filter)
        assigned = [t for t in assigned if t["status"] == status_target]

    if not assigned:
        return {"employee_id": employee_id, "status_filter": status_filter, "tickets": [], "count": 0}
    return {"employee_id": employee_id, "status_filter": status_filter, "tickets": assigned, "count": len(assigned)}


# ─────────────────────────────────────────────────────────────
# TOOL 6  calculate  (Challenge 1)
# ─────────────────────────────────────────────────────────────
def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt(), abs(), round(), floor(), ceil().
    """
    allowed = {
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


# ─────────────────────────────────────────────────────────────
# TOOL 7  get_department_headcount
# ─────────────────────────────────────────────────────────────
def get_department_headcount(department: str) -> dict:
    """Return the number of employees in a specific department."""
    dept = department.strip()
    members = [e for e in EMPLOYEES.values() if e["department"].lower() == dept.lower()]
    return {
        "department": dept,
        "headcount": len(members),
        "employees": [{"id": e["employee_id"], "name": e["name"], "role": e["role"], "status": e["status"]}
                      for e in members]
    }


# ─────────────────────────────────────────────────────────────
# TOOL REGISTRY  -- maps name → function
# ─────────────────────────────────────────────────────────────
TOOL_FUNCTIONS: dict[str, callable] = {
    "get_employee_information":  get_employee_information,
    "get_ticket_status":         get_ticket_status,
    "search_knowledge_base":     search_knowledge_base,
    "get_weather":               get_weather,
    "get_tickets_by_employee":   get_tickets_by_employee,
    "calculate":                 calculate,
    "get_department_headcount":  get_department_headcount,
}

# ─────────────────────────────────────────────────────────────
# TOOL SCHEMAS  -- Groq native function-calling format (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_information",
            "description": "Retrieve full employee profile by their numeric employee ID. Use when the user asks about an employee, their department, role, email, or status.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Retrieve the current status, priority, and details of an IT support ticket by its ID. Ticket IDs follow the format INC-XXXXX.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID, e.g. 'INC-12345'"
                    }
                },
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the company IT/HR knowledge base articles by keyword. Use when the user asks for documentation, procedures, policies, or how-to guides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Search keyword or phrase, e.g. 'password reset', 'VPN', 'expense reimbursement'"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions and temperature for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Monterrey', 'London', 'Tokyo'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tickets_by_employee",
            "description": "Retrieve all IT support tickets assigned to a specific employee. Optionally filter by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "The numeric employee ID"
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["all", "open", "resolved", "in_progress"],
                        "description": "Filter tickets by status. Defaults to 'all'."
                    }
                },
                "required": ["employee_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt(), abs(), round(), floor(), ceil(), pi, e. Use when the user asks to compute or calculate something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid mathematical expression string, e.g. 'sqrt(144)', '(15 * 8) / 3', '2**10'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_headcount",
            "description": "Return the number of employees and their names in a specific department. Use when the user asks how many people work in a department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Department name, e.g. 'IT', 'HR', 'Finance', 'Marketing'"
                    }
                },
                "required": ["department"]
            }
        }
    },
]

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful AI assistant for a company's internal support system.

You have access to the following tools:
- get_employee_information: look up employee profiles
- get_ticket_status: check IT support ticket status
- search_knowledge_base: find documentation and procedures
- get_weather: get current weather for a city
- get_tickets_by_employee: get all tickets assigned to an employee
- calculate: perform mathematical calculations
- get_department_headcount: get the number of employees in a department

RULES:
1. ONLY call a tool when the user's question genuinely requires real data from that tool.
2. For general knowledge questions (e.g., "What is AI?", "Explain databases"), answer directly WITHOUT calling any tool.
3. After calling a tool, base your final answer EXCLUSIVELY on the data returned by the tool.
4. Never invent employee names, ticket numbers, or any other data.
5. Be concise and professional.
""".strip()


# ─────────────────────────────────────────────────────────────
# CORE: Execute a single tool call and log it
# ─────────────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Run the tool function, log the execution, and return result as JSON string."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        result = {"error": f"Unknown tool: {tool_name}"}
    else:
        result = fn(**tool_args)

    # Challenge 3: Tool Usage Log
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool":      tool_name,
        "input":     tool_args,
        "output":    result,
    }
    TOOL_LOG.append(log_entry)

    return json.dumps(result, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# AGENTIC LOOP
# ─────────────────────────────────────────────────────────────
def run_agent(user_question: str, verbose: bool = False) -> dict:
    """
    Full agentic loop:
      Round 1: LLM receives user question + tool schemas
              → may respond with tool_calls OR directly with text
      Round N: For each tool call, execute the tool, append result as
              a tool message, send back to LLM
              → LLM eventually produces a final text answer
      Output: structured JSON with tool_used, answer, and sources.

    Supports multi-tool: LLM may request multiple tools in one turn (Challenge 2).
    """
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_question},
    ]

    tools_used = []
    max_rounds = 5   # prevent infinite loops

    for round_num in range(max_rounds):
        if verbose:
            print(f"\n  [Round {round_num + 1}] Calling LLM...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",   # LLM decides: call a tool or answer directly
            temperature=0.1,
            max_tokens=1024,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── LLM returned a final text answer (no tool call needed) ──
        if finish_reason == "stop" or not msg.tool_calls:
            final_answer = msg.content or "I was unable to generate an answer."
            break

        # ── LLM wants to call one or more tools ──
        # Append assistant message (with tool_calls) to conversation
        messages.append(msg)

        # Execute all requested tools (Challenge 2: multi-tool)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  [Tool] {fn_name}({fn_args})")

            tool_result = execute_tool(fn_name, fn_args)
            tools_used.append(fn_name)

            if verbose:
                print(f"  [Result] {tool_result[:200]}...")

            # Append tool result as a tool message
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      tool_result,
            })
    else:
        final_answer = "Max tool call rounds reached without a final answer."

    # ── Step 5: Structured JSON output ──
    return {
        "tool_used":   tools_used[0] if len(tools_used) == 1 else tools_used if tools_used else None,
        "tools_count": len(tools_used),
        "answer":      final_answer,
    }


# ─────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  GEN AI Upskilling -- Exercise 05: Tool Calling Agent")
    print("=" * 65)
    print("  Ask anything about employees, tickets, weather, or docs.")
    print("  Commands: 'log' (show tool log) | 'verbose on/off' | 'quit'")
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
        elif question.lower() == "log":
            if not TOOL_LOG:
                print("  No tool calls logged yet.\n")
            else:
                print(json.dumps(TOOL_LOG[-5:], indent=2, ensure_ascii=False))
            continue
        elif question.lower() == "verbose on":
            verbose = True
            print("  Verbose ON.\n")
            continue
        elif question.lower() == "verbose off":
            verbose = False
            print("  Verbose OFF.\n")
            continue

        print("\n[Agent is thinking...]\n")
        result = run_agent(question, verbose=verbose)

        tool_info = f"Tool(s) used: {result['tool_used']}" if result["tool_used"] else "No tool needed"
        print(f"[{tool_info}]")
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Structured JSON:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        print("\n" + "-" * 65)


if __name__ == "__main__":
    main()
