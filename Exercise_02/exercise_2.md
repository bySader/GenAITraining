## GEN AI Upskilling Training
This document describes a hands-on exercise designed to help participants get familiar with the fundamentals of Generative AI by interacting directly with a Large Language Model (LLM).

This exercise focuses on applying and practicing prompting techniques and finding ways to force a structured JSON response.


### Objective
The main objective of this exercise is to:

Apply and Practice basic prompt engineering techniques.
- Zero shot
- Few shot
- Chain of thought

Learn how to enforce structured outputs (JSON) from an LLM.

By the end of this exercise, participants will know how to effectively apply prompting techniques.
### Exercise Description
You will create a function that classifies text.

Allowed Categories & Sub-Categories:
- Reviews
  
  Sub-categories:
    - Positive
    - Negative
    - Neutral

- Marketing & Promotion
  
  Sub-categories:
    - Product Promotion
    - Service Promotion
    - Discount - Offer
    - Brand awareness

- Academic
  
  Sub-categories:
    - History
    - Ethics
    - Business
    - Geography
  
- Sports/Leagues
  
  Sub-categories:
    - American Football / NFL
    - Basketball / MLB
    - Soccer / MLS
    - Hockey / NHL
    - Women Basketball / WNBA

- Finance & Expenses
  
  Sub-categories:
    - Personal expenses
    - Business expenses
    - Travel expenses
    - Budget planning

The exercise consists of the following steps:

1. Create a Dataset (at least 30 short documents/ texts) that cover different
samples of the categories mentioned above.
You may store them as .txt or PDFs or just strings in a notebook.

**Tip** 
If you don't want to read the content of documents, you can ask copilot or any LLM to generate pieces of text that fit the categories above. (For training purposes only)

2. Load documents 

Use any available library to extract/read the content of the files.PyPdf, .txt standard file read.,

3. Prompt Design (classification + JSON enforcement)

You may use either:
* **Option A (1-call)**: classify category and sub-category in a single prompt
* **Option B (2-call chain)**: first choose category, then choose sub-category constrained by category

Option B tends to reduce Hallucionation risk.

**Notes**
* If the text does not fit in any of the category, map the text to the more appropiate category
* Other categories outside the list above are not allowed.

Output 
* Return strictly valid json object like this:
    {
      "document_name": "",
      "category": "",
      "sub-category: ""
    }

**Examples**
1. {"document_name":"example.txt","category":"Reviews","sub_category":"Positive"}
2. {"document_name":"example.txt","category":"Finance & Expenses","sub_category":"Budget planning"}

**Notes**
* Carefull with the token consumption. Check the max tokens allowed by your model per call.

#### Pre-requisites
Before starting the exercise, you should have:

  * Basic programming knowledge (Python, .NET, or similar)
  * An active API key or access to at least one LLM provider
  * A local development environment set up (vs code)
  * PyPdf package can read pdf documents pretty easy
  

### Acceptance Criteria
The exercise will be considered complete when:

✅ A full list of text/documents/pieces of text that contains many examples of the categories.

✅ The text can be passed to the LLM to be classified

✅ The LLM gets the category and sub-category of each text without hallucinations.

✅ The response is always returned in valid JSON format


### Stack to use
You are free to choose the tools and libraries you prefer. Examples include (but are not limited to):

* LLM Providers
    * Azure OpenAI
    * Groq

* Libraries / Frameworks

    * LangChain
    * OpenAI SDK
    * Any equivalent GenAI SDK

* Languages

    * Python

### Evaluation
The solution will be evaluated based on:

- Prompt clarity and effectiveness
- Implementation of prompting techines (few-zero shot, chain of thought etc.,)
- Accuracy of the LLM to classify text
- I'll have a test cases (sample texts) ready for the next session, the plan is to pass the 
texts through the prompt and validate the output (category & sub-category)
- Enforcement of structured (JSON) output