## GEN AI Upskilling Training
This document describes a hands-on exercise designed to help participants get familiar with Retrieval Augmented Generation (RAG) patterns. 

This exercise focuses on applying and practicing prompting techniques, augmenting the prompt by injecting context, document selection, index creation, embeddings, similarity and vector search. 


### Objective
The main objective of this exercise is to:

Get familiar with concepts such as:
- RAG
- Hallucinations
- Vector search index
- Embeddings
- Chunks
- Grounded answers

By the end of this exercise, participants will know how to effectively implement a simple RAG system, use embeddings and vector search.

### Exercise Description
The goal of this exercise is to create a function that can answer questions based on context. Avoid hallucinations and integrate RAG with an LLM.

**Step 1**

Extract the provided document (synthetic_knowledge_items.csv). Create a new index and upload the content of the file into.

 *Note*: ChromaDB, Faiss (recommended for simplicity)

**Step 2**

Create a python function to query your index and retrieve the context from it.

*Function must receive a string or embeddings vector or both as input and return a string.*

*Function must implement similarity search or vector search or hybrid*

**Step 3**

Use the retrieved information from your index in your prompt. 
Create a smart prompt for the LLM to use; inject the context there. The LLM must respond every question based on the context, if context does not have enough information, just respond "I don't Know". 


**Note**
Prompt must be set to only answer questions based on the provided file. If the question if out of scope, LLM must say it in the answer.

- For example: Q: "How is the weather in Mexico?", A: "I don't know, that is no part of my context"

#### Pre-requisites
Before starting the exercise, you should have:

  * Basic programming knowledge (Python, .NET, or similar)
  * An active API key or access to at least one LLM provider
  * A local development environment set up (vs code)
  * Research about RAG, indexes, embeddings, similarity and vector search.
  

### Acceptance Criteria
The exercise will be considered complete when:

✅ Trainee explain what is RAG and why is necessary.

✅ Trainee explain difference between similarity and vector search. 

✅ LLM answer questions based ONLY in provided data. No Hallucinations

✅ LLM return accurate responses


### Stack to use
You are free to choose the tools and libraries you prefer. Examples include (but are not limited to):

* LLM Providers
    * Azure OpenAI
    * Groq

* Libraries / Frameworks

    * LangChain
    * Faiss
    * ChromaDB
    * Azure AI Search service
    * similar...

* Languages

    * Python

* Dataset
  * Download the csv file from here: https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items
### Evaluation
The solution will be evaluated based on:

- Prompt clarity and effectiveness
- Index creation and text retrieval
- LLM Answer's accuracy (according to information in the file)
- I'll have a list of Q&A and we send that questions to the function and evaluate response.

### References

- https://huggingface.co/blog/ngxson/make-your-own-rag
- https://docs.langchain.com/oss/python/langchain/retrieval
- https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
- https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- https://www.riis.com/blog/introduction-to-rag-with-langchain-and-openai