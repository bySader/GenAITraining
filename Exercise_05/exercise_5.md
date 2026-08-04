# GEN AI Upskilling Training

This document describes a hands-on exercise designed to help participants get familiar with Tool Calling / Function Calling patterns in modern LLM applications.

This exercise focuses on enabling an LLM to decide when external functions should be executed, how to invoke them, how to consume their results, and how to generate grounded responses based on tool outputs.

## Objective

The main objective of this exercise is to:

Get familiar with concepts such as:

- Tool Calling
- Function Calling
- Structured Outputs
- LLM Orchestration
- External Systems Integration
- Grounded Responses

By the end of this exercise, participants will understand how an LLM can interact with external functions and use real data instead of generating responses entirely from its own knowledge.

---

## Exercise Description

The goal of this exercise is to create an AI Assistant capable of invoking external tools when necessary.

The assistant must be able to:

- Understand a user's request
- Decide whether a tool is required
- Select the appropriate tool
- Execute the tool
- Use the returned information to answer the user

### Step 1

Create a set of mock functions that simulate real business operations.

For example:

```python
get_employee_information(employee_id)
get_ticket_status(ticket_id)
search_knowledge_base(keyword)
get_temperature(city)
```

Example outputs:

```json
{
  "employee_id": "1001",
  "name": "John Doe",
  "department": "IT"
  
}
```

```json
{
  "ticket_id": "INC-12345",
  "status": "In Progress",
  "priority": "High"
}
```

```json
{
  "article": "Password Reset Procedure"
}
```

```json
{
  "temp_mty": "30° C"
}
```

You may store the information in:

- JSON files
- Dictionaries
- CSV files
- Any equivalent lightweight data source

For training purposes, using Python dictionaries is sufficient.

### Step 2

Register the functions as tools available to the LLM.

Research how your selected framework supports tool/function calling.

Examples:

- OpenAI SDK Tools
- Azure OpenAI Function Calling
- LangChain Tools
- LangGraph Tools
- Semantic Kernel Plugins

The LLM must know:

- Tool name
- Tool description
- Input parameters
- Expected output

### Step 3

Create a chat interface that receives questions from users.

Examples:

```text
What is the status of ticket INC-12345?
```

```text
Show information about employee 1001
```

```text
Search for password reset documentation
```

The LLM should:

1. Analyze the user request
2. Determine which tool should be used
3. Generate the tool call
4. Execute the tool
5. Receive the result
6. Generate the final answer

### Step 4

Implement response generation using tool outputs.

The final response must be generated only from the information returned by the selected tool.

### Step 5

Implement structured output.

Regardless of the user question, the final response must also be available in JSON format.

Example:

```json
{
  "tool_used": "get_ticket_status",
  "answer": "Ticket INC-12345 is currently In Progress and has High priority."
}
```

---

## Important Notes

The LLM should not invoke tools unnecessarily.

### Tool Required

```text
What is the status of ticket INC-12345?
```

```text
Search employee 1001
```

### Tool Not Required

```text
What is artificial intelligence?
```

```text
Explain what a database is.
```

For questions that do not require a tool, the assistant may answer directly.

---

## Pre-requisites

- Basic programming knowledge (Python, .NET, or similar)
- An active API key or access to at least one LLM provider
- A local development environment set up (VS Code)
- Basic understanding of prompts and structured outputs
- Completion of previous exercises is recommended

---

## Acceptance Criteria

✅ At least five tools/functions have been implemented

✅ The LLM can automatically determine which tool to invoke

✅ The selected tool is executed successfully

✅ The assistant uses the tool output to generate the response

✅ The response is available in valid JSON format

✅ The solution avoids unnecessary tool calls

✅ The trainee can explain the difference between an LLM response and a Tool Calling workflow

---

## Stack to Use

### LLM Providers

- Azure OpenAI
- OpenAI
- Groq

### Libraries / Frameworks

- OpenAI SDK
- LangChain
- LangGraph
- Semantic Kernel
- Any equivalent GenAI SDK

### Languages

- Python

---

## Evaluation

The solution will be evaluated based on:

- Correct implementation of Tool Calling / Function Calling
- Tool selection accuracy
- Proper handling of input parameters
- Response grounding using tool outputs
- Structured JSON output generation
- Explanation of the overall execution flow
- Code quality and maintainability

---

## Challenges

### Challenge 1

Add a Calculator Tool:

```python
calculate(expression)
```

### Challenge 2

Add Multiple Tool Support

```text
Get information for employee 1001 and show all open tickets assigned to that employee.
```

### Challenge 3

Create a Tool Usage Log

```json
{
  "timestamp": "",
  "tool": "",
  "input": "",
  "output": ""
}
```

### Challenge 4

Tool Execution in different order based on user's request
