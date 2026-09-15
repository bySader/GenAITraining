# 🚀 GenAI Training Repository

> A hands-on training repository designed to help developers learn and practice **Generative AI fundamentals** through real-world Python exercises.
>
> This learning path covers Prompt Engineering, Retrieval-Augmented Generation (RAG), Tool Calling, AI Agents, and LangGraph workflows using modern LLM technologies.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Groq](https://img.shields.io/badge/LLM-Groq-oranges.io/badge/Framework-LangChain-green)
![LangChain](https://img.shields.io/badge/langchain-%231C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)

---

# 📖 About This Repository

This repository provides a structured and progressive learning journey into Generative AI application development.

Each exercise is self-contained and focuses on a specific GenAI concept, allowing participants to learn by building practical solutions rather than studying theory alone.

By the end of the training path, learners will have experience with:

- Prompt Engineering
- Structured LLM Outputs
- Text Classification
- Retrieval-Augmented Generation (RAG)
- Vector Databases (FAISS)
- Embeddings
- Metadata Filtering
- Re-ranking Strategies
- Tool Calling
- Agentic Workflows
- LangGraph State Machines
- Natural Language to SQL Applications

---

# 🎯 Learning Roadmap

The exercises are organized from foundational concepts to advanced GenAI architectures.

| Exercise | Topic | Key Concepts |
|-----------|--------|--------------|
| 01 | Text Summarization | Prompting, JSON Output, Structured Responses |
| 02 | Text Classification | Prompt Engineering, Categorization |
| 03 | RAG Fundamentals | Embeddings, Vector Search, Grounded Answers |
| 04 | Advanced RAG | Metadata Filtering, Re-ranking, Multi-format Documents |
| 05 | Tool Calling Agent | Function Calling, Multi-tool Reasoning, Agent Loops |
| 06 | LangGraph NL-to-SQL | Routing, State Management, SQL Agents, RAG Nodes |

---

# 🗂 Repository Structure

```text
GenAITraining/
│
├── Exercise_01/
├── Exercise_02/
├── Exercise_03/
├── Exercise_04/
├── Exercise_05/
├── Exercise_06/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🧪 Exercises Overview

| # | Exercise | Description | Technologies |
|---|-----------|-------------|-------------|
| 01 | [Exercise_01](./Exercise_01/) | Text Summarizationd JSON Output | Python · Groq · LLaMA3 |
| 02 | [Exercise_02](./Exercise_02/) | Clasification with Prompt Engineering | Python · Groq · LLaMA3 |
| 03 | [Exercise_03](./Exercise_03/) | Retrieval-Augmented Generationem | Python · Groq · FAISS · Sentence Transformers |
| 04 | [Exercise_04](./Exercise_04/) | Advanced RAG with Multi-Format Documents, Metadata Filtering, and Re-ranking | Python · BM25 · PDF/DOCX Processing |
| 05 | [Exercise_05](./Exercise_05/) | AI Agent with Tool Calling and Multi-Tool Executionq · Function Calling |
| 06 | [Exercise_06](./Exercise_06/) | NL-to-SQL Workflow | LangGraph · LangChain · Groq · SQLite |

---

# ⚙️ Prerequisites

Before starting, ensure the following tools are installed:

- Python 3.10 or higher
- Groq API Key
- Visual Studio Code (recommended)

Create a free API key at:

👉 https://console.groq.com

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/GenAITraining.git
cd GenAITraining
```

## 2. Navigate to an Exercise

```bash
cd Exercise_01
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Copy the environment template:

```bash
cp .env.example .env
```

Add your API key:

```env
GROQ_API_KEY=your_api_key_here
```

## 5. Run the Exercise

```bash
python main.py
```

> Refer to the specific README inside each exercise folder for detailed instructions.

---

# 📂 Standard Exercise Layout

All exercises follow a common structure to ensure consistency and ease of navigation.

```text
Exercise_XX/
│
├── README.md
├── requirements.txt
├── .env.example
├── dataset/
└── *.py
```

### Contents

| File*/ Folder | Purpose |
|*--------------|----------|
| READM**md | Exercise explanation and inst*uctions |
| requirements.txt | Req*ired Python dependencies |
| .env.*xample | Environment variable temp*ate |
| dataset/ | Sample data (wh*n applicable) |
| *.py | Exercise *mplementation |

---

# 🖥 Interac*ive Dashboard

The repository incl*des a web-based dashboard that all*ws users to browse, review, and ex*cute exercises directly from a bro*ser.

## Features

✅ Browse all av*ilable exercises

✅ View exercise *escriptions

✅ Access README docum*ntation

✅ Execute individual scri*ts

✅ Real-time execution output

* Interactive prompt support

✅ Res*onsive design (Desktop & Mobile)

*--

## Launch the Dashboard

Insta*l the root dependencies:

```bash
*ip install -r requirements.txt
```*
Start the application:

```bash
p*thon app.py
```

Open the dashboar* in your browser:

```text*http://localhost:5000
```

> The r*ot installation includes the core *ependencies required by the dashbo*rd to launch exercises without mod*le dependency issues.

---

# 📈 T*aining Progression

```text
Prompt*Engineering
        │
        ▼
Te*t Processing
        │
        ▼
S*ructured Outputs
        │
       *▼
RAG Fundamentals
        │
     *  ▼
Advanced Retrieval
        │
 *      ▼
Tool Calling
        │
   *    ▼
AI Agents
        │
        *
LangGraph Workflows
        │
   *    ▼
Production-Ready GenAI Appli*ations
```

---

# 🔒 Security Bes* Practices

- Never commit `.env` *iles.
- Never expose API keys in s*urce code.
- Always use `.env.exam*le` as the project template.
- Rot*te credentials if accidental expos*re occurs.
- Review generated outp*ts before using them in production*environments.

---

# 🤝 Adding a *ew Exercise

To extend the reposit*ry:

1. Create a new folder using *he naming convention:

```text
Exe*cise_XX/
```

2. Follow the standa*d exercise structure.

3* Include:
   - README.md
   - requ*rements.txt
   - .env.example
   -*Sample datasets (if required)

4. *dd the exercise to the **Learning *oadmap** and **Exercises Overview** sections.

5. Ensure all scripts *an*be launched individually from the *ashboard.

---

# 🎓 Learning Obje*tives

By completing all exercises* participants should be able to:

* Build and evaluate LLM prompts
- *enerate structured outputs
- Imple*ent text classification pipelines
* Create Retrieval-Augmented Genera*ion systems
- Design advanced retr*eval strategies
- Build AI agents *ith tool calling capabilities
- De*elop workflow-based AI application* with LangGraph
- Apply GenAI desi*n patterns in real-world scenarios*
---

# 📄 License

This repositor* is intended for **internal traini*g and educational purposes**.

Una*thorized distribution or commercia* use may be restricted according t* your organization's policies.
```*