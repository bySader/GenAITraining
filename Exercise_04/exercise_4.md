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
- Ranking & Re-ranking

By the end of this exercise, participants will know how to effectively implement a simple RAG system, use embeddings and vector search.

### Exercise Description
The goal of this exercise is to create a function that can answer questions based on context. Avoid hallucinations and integrate RAG with an LLM.

This exercise will be similar to previous exercise, now instead of having just one file in the index we will have multiple files with differente extensions (.pdf, .docx, .txt, .json, .md)

**Step 1**

Find/create dummy documents in format PDF, docx, json, txt and markdowns. The documents can contain information about any topic. This is just for learning purposes.

Extract content from documents and upload the content and the document metadata to the index. 
Research how to create a custom index with customized fields. 

For example: 
{
  "Document_name": "",
  "number_of_pages": ,
  "content": "",
  "author": ""
}

 *Note*: Use short documents, no more than 20 pages per document. Include at least 15 different documents from 4 or 5 different topics. The idea is that the function can get more than one document per query. Not that every document mentions a different topic.

**Step 2**

Create a python function to query your index.

*Function must:*

- receive a string/embeddings vector or both as input and return a string.
- retrieve the best document 
- retrieve the best chunk of the selected document to answer the question.
- implement similarity search or vector search or hybrid

IMPORTANT:
If the query is not related to the database/index information, do not search.
Return a polite message saying that the information is out of scope. 

**Step 3**

Use the retrieved document and content from your index in your prompt. 
Create a smart prompt for the LLM to use; inject the context there. The LLM must respond every question based on the context, if context does not have enough information, just respond "I don't Know". 


**Note**
Prompt must be set to only answer questions based on the provided file. If the question if out of scope, LLM must say it in the answer.

- For example: Q: "How is the weather in Mexico?", A: "I don't know, that is no part of my context"

#### Pre-requisites
Before starting the exercise, you should have:

  * Basic programming knowledge (Python, .NET, or similar)
  * An active API key or access to at least one LLM provider
  * A local development environment set up (vs code)
  * Research about RAG, indexes, embeddings, similarity and vector search ranking
  * Download/create dummy documents for the exercise.
  

### Acceptance Criteria
The exercise will be considered complete when:

✅ Show ranking concept and implementation

✅ Show the retrieving documents functionality

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

### Evaluation
The solution will be evaluated based on:

- Prompt clarity and effectiveness
- Index creation, document and chunk retrieval
- LLM Answer's accuracy (according to information in the file)
- Effective prompt that does not search for content that is not part of its knowledge.

### References

- https://huggingface.co/blog/ngxson/make-your-own-rag
- https://docs.langchain.com/oss/python/langchain/retrieval
- https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
- https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- https://www.riis.com/blog/introduction-to-rag-with-langchain-and-openai
- https://towardsdatascience.com/rag-explained-reranking-for-better-answers