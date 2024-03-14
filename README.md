# Question Answering using Retrieval Augmented Generation

This repository contains the code for Retrieval Augmented Generation using LangChain and FAISS.

How to run the code:
```
# Run with no RAG
python main.py --questions questions.csv --output predictions_no_rag.csv

# Run with RAG (with langchain embeddings)
python main.py --questions questions.csv --rag --langchain --passages passages.csv --output predictions_rag_langchain.csv

# Run with RAG (with custom embeddings)
python main.py --questions questions.csv --rag --passages passages.csv --output predictions_rag.csv

```
