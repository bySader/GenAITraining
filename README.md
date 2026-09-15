# 🚀 GenAI Training

**A hands-on, project-based path to mastering modern Generative AI development with Python.**

From your first prompt to a full LangGraph-powered agent, this repository walks you through the core building blocks of production-grade GenAI applications — one practical exercise at a time.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LLM-Groq-orange?style=for-the-badge&logo=lightning&logoColor=white" alt="Groq">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/License-Internal%20Use-lightgrey?style=for-the-badge" alt="License">
</p>

---

## 📖 About This Repository

This repository provides a structured, progressive learning journey into building real Generative AI applications — not just reading about the theory behind them.

Each exercise is **self-contained**, focused on one core concept, and designed to be completed in an afternoon. You'll go from writing your first structured prompt to shipping a natural-language-to-SQL agent built on LangGraph.

By the end of the path, you'll have hands-on experience with:

- ✅ Prompt Engineering & Structured LLM Outputs
- ✅ Text Classification
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Vector Databases (FAISS) & Embeddings
- ✅ Metadata Filtering & Re-ranking Strategies
- ✅ Tool Calling & Agentic Workflows
- ✅ LangGraph State Machines
- ✅ Natural Language–to–SQL Applications

---

## 🎯 Learning Roadmap

Exercises are ordered from foundational concepts to advanced GenAI architectures — complete them in sequence for the smoothest learning curve.

| # | Exercise | Topic | Key Concepts |
|---|----------|-------|---------------|
| 01 | [Text Summarization](./Exercise_01/) | Prompting & Structured Output | Prompting, JSON output, structured responses |
| 02 | [Text Classification](./Exercise_02/) | Prompt Engineering | Categorization, classification prompts |
| 03 | [RAG Fundamentals](./Exercise_03/) | Retrieval-Augmented Generation | Embeddings, vector search, grounded answers |
| 04 | [Advanced RAG](./Exercise_04/) | Multi-format Retrieval | Metadata filtering, re-ranking, multi-format documents |
| 05 | [Tool Calling Agent](./Exercise_05/) | Agentic Workflows | Function calling, multi-tool reasoning, agent loops |
| 06 | [LangGraph NL-to-SQL](./Exercise_06/) | State Machines | Routing, state management, SQL agents, RAG nodes |

```
Prompt Engineering
        │
        ▼
  Text Processing
        │
        ▼
 Structured Outputs
        │
        ▼
  RAG Fundamentals
        │
        ▼
 Advanced Retrieval
        │
        ▼
   Tool Calling
        │
        ▼
    AI Agents
        │
        ▼
LangGraph Workflows
        │
        ▼
Production-Ready GenAI Applications
```

---

## 🧪 Exercises Overview

| # | Exercise | Description | Technologies |
|---|----------|-------------|---------------|
| 01 | [Exercise_01](./Exercise_01/) | Text summarization with structured JSON output | Python · Groq · LLaMA 3 |
| 02 | [Exercise_02](./Exercise_02/) | Text classification with prompt engineering | Python · Groq · LLaMA 3 |
| 03 | [Exercise_03](./Exercise_03/) | Retrieval-Augmented Generation from scratch | Python · Groq · FAISS · Sentence Transformers |
| 04 | [Exercise_04](./Exercise_04/) | Advanced RAG with multi-format documents, metadata filtering, and re-ranking | Python · BM25 · PDF/DOCX processing |
| 05 | [Exercise_05](./Exercise_05/) | AI agent with tool calling and multi-tool execution | Python · Groq · Function calling |
| 06 | [Exercise_06](./Exercise_06/) | Natural-language-to-SQL workflow | LangGraph · LangChain · Groq · SQLite |

---

## ⚙️ Prerequisites

Before you start, make sure you have:

- 🐍 **Python 3.10+**
- 🔑 A **Groq API key** ([get one free here](https://console.groq.com))
- 💻 **Visual Studio Code** (recommended, not required)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/GenAITraining.git
cd GenAITraining
```

### 2. Navigate to an exercise

```bash
cd Exercise_01
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your environment variables

```bash
cp .env.example .env
```

Then open `.env` and add your API key:

```env
GROQ_API_KEY=your_api_key_here
```

### 5. Run the exercise

```bash
python main.py
```

> 💡 **Tip:** Each exercise folder has its own `README.md` with detailed, exercise-specific instructions — start there once you're set up.

---

## 🖥️ Interactive Dashboard

Prefer a visual workflow over the terminal? The repository ships with a **web-based dashboard** to browse, review, and run every exercise from your browser.

**Features:**

- 📂 Browse all available exercises
- 📝 View exercise descriptions and documentation inline
- ▶️ Execute individual scripts with one click
- 📡 Real-time execution output
- ⌨️ Interactive prompt support
- 📱 Responsive design (desktop & mobile)

**Launch it:**

```bash
pip install -r requirements.txt
python app.py
```

Then open your browser at:

```
http://localhost:5000
```

> The root `requirements.txt` includes every dependency the dashboard needs to run all exercises without module conflicts.

---

## 🗂 Repository Structure

```
GenAITraining/
│
├── Exercise_01/
├── Exercise_02/
├── Exercise_03/
├── Exercise_04/
├── Exercise_05/
├── Exercise_06/
│
├── app.py               # Dashboard entry point
├── requirements.txt      # Root-level dependencies
└── README.md
```

Every exercise follows the same standard layout, so once you've done one, you know your way around all of them:

```
Exercise_XX/
│
├── README.md          # Exercise explanation and instructions
├── requirements.txt    # Required Python dependencies
├── .env.example         # Environment variable template
├── dataset/             # Sample data (when applicable)
└── *.py                 # Exercise implementation
```

---

## 🎓 Learning Objectives

By completing all exercises, you will be able to:

- Build and evaluate LLM prompts
- Generate structured outputs from an LLM
- Implement text classification pipelines
- Create Retrieval-Augmented Generation systems
- Design advanced retrieval strategies (filtering, re-ranking)
- Build AI agents with tool-calling capabilities
- Develop workflow-based AI applications with LangGraph
- Apply GenAI design patterns to real-world scenarios

---

## 🔒 Security Best Practices

- 🚫 Never commit `.env` files.
- 🚫 Never expose API keys in source code.
- ✅ Always use `.env.example` as your project template.
- 🔁 Rotate credentials immediately if accidental exposure occurs.
- 👀 Review generated outputs before using them in production environments.

---

## 🤝 Adding a New Exercise

Want to extend the repository? Follow these steps:

1. Create a new folder using the naming convention `Exercise_XX/`.
2. Follow the [standard exercise structure](#🗂-repository-structure) above.
3. Include:
   - `README.md`
   - `requirements.txt`
   - `.env.example`
   - Sample datasets (if required)
4. Add the exercise to the **Learning Roadmap** and **Exercises Overview** tables above.
5. Make sure the script can be launched individually from the dashboard.

---

## 📄 License

This repository is intended for **internal training and educational purposes**.
Unauthorized distribution or commercial use may be restricted according to your organization's policies.
