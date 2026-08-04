# Exercise 05 — Tool Calling / Function Calling Agent

## Overview
An AI assistant powered by **Groq's native function-calling API** that can invoke real Python functions when needed, then ground its answers in the returned data.

## Tools Implemented (7 total)

| # | Tool | Description |
|---|------|-------------|
| 1 | `get_employee_information` | Fetch employee profile by ID |
| 2 | `get_ticket_status` | Check IT support ticket status |
| 3 | `search_knowledge_base` | Search IT/HR documentation |
| 4 | `get_weather` | Get weather for a city |
| 5 | `get_tickets_by_employee` | List all tickets for an employee |
| 6 | `calculate` | Safe math expression evaluator |
| 7 | `get_department_headcount` | Count employees per department |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
copy .env.example .env
# Edit .env -> GROQ_API_KEY=your_key

# 3. Run the interactive agent
python agent.py

# 4. Run the automated evaluation
python evaluate.py
```

## Usage Examples

```
You: What is the status of ticket INC-12345?
[Tool: get_ticket_status]
Answer: Ticket INC-12345 "VPN connection failure" is currently In Progress with High priority.

You: Get info for employee 1001 and show all their open tickets
[Tools: get_employee_information + get_tickets_by_employee]
Answer: Alice Martinez (Senior Developer, IT) has 1 open ticket: INC-12348 "Cannot access shared drive" (High priority).

You: What is artificial intelligence?
[No tool needed]
Answer: Artificial intelligence is the simulation of human intelligence...
```

## Commands
| Command | Effect |
|---------|--------|
| `log` | Show last 5 tool call log entries |
| `verbose on/off` | Show/hide internal tool call details |
| `quit` | Exit |

## Challenges Implemented
- ✅ **Challenge 1**: `calculate` tool
- ✅ **Challenge 2**: Multi-tool support (LLM chains multiple tool calls per turn)
- ✅ **Challenge 3**: Tool usage log (timestamp, tool, input, output)
- ✅ **Challenge 4**: Tool execution order driven by the LLM
