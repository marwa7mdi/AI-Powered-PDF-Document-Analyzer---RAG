# 🤖 AI-Powered Document Intelligence: Google ESG Report Analyzer (RAG)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)
![Google Gemini](https://img.shields.io/badge/LLM-Gemini--Flash-orange.svg)
![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-red.svg)

## 📋 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to analyze and query complex PDF documents (using Google's Environmental, Social, and Governance report as a case study). The system allows users to ask natural language questions and receive accurate, context-aware answers extracted directly from the document.

### ✨ Key Features
- **Semantic Search:** Uses `gemini-embedding-001` to understand the meaning behind queries, not just keywords.
- **Context-Strict Guardrails:** Implemented a custom prompt template to ensure the AI only answers based on the provided text, eliminating hallucinations.
- **Resource Optimization:** Optimized for **Google AI Studio Free Tier** by using smart chunking and specific page indexing to manage API rate limits.
- **Persistence:** Uses **ChromaDB** for efficient vector storage and retrieval.

---

## 🏗️ Architecture
1. **Ingestion:** Loading PDF data using `PyPDFLoader`.
2. **Chunking:** Splitting text into manageable pieces using `RecursiveCharacterTextSplitter` (1000 chars with 150 overlap).
3. **Embedding:** Converting text chunks into high-dimensional vectors via Google Generative AI.
4. **Vector Store:** Indexing embeddings in `ChromaDB`.
5. **Retrieval & Generation:** Fetching relevant context and passing it to `gemini-flash-latest` for final answer synthesis.

---

## 🛠️ Tech Stack
* **Orchestration:** LangChain
* **LLM:** Google Gemini Flash Latest
* **Embeddings:** Google Generative AI Embeddings
* **Vector Database:** ChromaDB
* **Environment:** Google Colab / Python

---

## 🚀 Getting Started

### Prerequisites
- A Google AI Studio API Key.
- The target PDF file (e.g., `google_report.pdf`).

### Installation
```bash
pip install langchain-google-genai langchain-community chromadb pypdf langchain
