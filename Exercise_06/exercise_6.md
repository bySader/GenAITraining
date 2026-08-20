# GEN AI Upskilling Training

This document describes a hands-on exercise designed to help participants get familiar with Tool Calling, Langgraph workflow, Langgraph Agent.

This exercise focuses on creating a first simple Langgraph Workflow 

## Objective

The main objective of this exercise is to:

Get familiar with concepts such as:

- Tool Calling
- LangGraph
- LangGraph Workflow
- LangGraph State

By the end of this exercise, participants will understand how to build a simple LangGraph Workflow.

---

## Exercise Description

The goal of this exercise is to create a Langgrap Workflow that can answer questions based on the responses from a tool. 

Participants must use a LangGraph State to pass information between nodes.

Example:
```
class AgentState(TypedDict):
  question: str
  sql_query: str
  query_result: str
  final_response: str
```

The workflow must be able to:

- Understand a user's request
- Decide whether a tool is required
- Select the appropriate tool
- Route to the right tool
- Construct a final response that makes sense according to user's query and tool response.

### Step 1

* Create/Deploy a database (Anything works, mongo, cosmos, AzureSQL)
* Create 5 new tables, 
TABLE TICKETS

  - ticket_id
  - priority 
  - status 
  - owner 
  - owner_id
  - created_date

TABLE EMPLOYEE

  - employee_id
  - name
  - department
  - hire_date
  - project
  - skills

TABLE PROJECTS

  - project_id
  - name
  - duration
  - team (List employees)
  - termination_date
  - has_open_tickets
  - open_tickets

TABLE CLIENTS

  - client_id
  - client_name
  - industry
  - country
  - account_manager
  - active_projects

TABLE SKILL_CERTIFICATIONS

  - certification_id
  - employee_id
  - certification_name
  - provider
  - issue_date
  - expiration_date

* Fill the tables with mock data 10 or 15 rows per table is fine.

### Step 2

* Research about LangGraph Workflow
* Create a simple workflow with the following nodes:
  - An entry node that decides if query needs tools or not
  - A ToolNode that calls the tool and returns a response
  - A format response node that applies a pretty format to the final response and return it to the user.


### Step 3

The ToolNode must do the following:
Use LLM to Analyze the query from user, for example: 'What ticket is more urgent?'
Convert the user query to SQL query i.e. 'Select ticket, ticket_number from table where max(date)'
,run the query in the database, retrieve the answer and send it back to the next node. 


## Important Notes

The Workflow must not invoke tools unnecessarily.
Workflow's development is up to you, the exercise's core is: an LLM that can connect to a database, run a query and retrieve a valid response. Then use that response in the workflow to create a natural language response to the user.

## Pre-requisites

- Basic programming knowledge (Python, .NET, or similar)
- An active API key or access to at least one LLM provider
- A local development environment set up (VS Code)
- Basic understanding of prompts and structured outputs
- Completion of previous exercises is recommended
- An active database and the connection through connection String, Key, username and pwd, etc.,

---

## Acceptance Criteria

✅ A LangGraph Workflow implementation

✅ Tool to connect to a database

✅ LLM sql query generation

✅ Workflow routing and tool calling

✅ Valid responses / no Hallucinations

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

- Workflow creation
- LLM sql query generation
- LLM responses
- Workflow State handling
- Workflow can answer questions like these:

  What is the most urgent ticket?
  How many open tickets does Project Phoenix have?
  Who is assigned to the highest priority ticket?
  Which employee has the most open tickets?
  Show all critical tickets created this month.

---

## Challenges

### Challenge 1

Add a second Node that performs RAG with the vector index you have already.

### Challenge 2

Forbid the LLM to create SQL statements that can alter the db such as CREATE, DELETE, UPDATE, MERGE, INSERT etc.,