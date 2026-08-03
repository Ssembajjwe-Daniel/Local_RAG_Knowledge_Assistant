# Local RAG Knowledge Assistant

## Overview

This project is a local Retrieval-Augmented Generation (RAG) prototype for querying information from documents.

The application processes documents, creates embeddings, retrieves relevant information, and generates answers using a locally hosted language model.

The system is designed to keep document processing local by running the application in Docker and connecting it to a local Ollama instance.

## Features

- Document processing for RAG pipelines
- Text chunking and embedding generation
- Vector similarity search
- Question answering with a local LLM
- Streamlit-based user interface
- Docker-based deployment

## Technologies

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- Docker

## Architecture Overview

The application runs inside a Docker container.

Ollama runs locally on the host machine and provides the language model through its API.

## Running the Application

### Prerequisites

Install and start:

- Docker
- Ollama

Ensure that Ollama is running locally before starting the container.

### Start the application

Run:

```bash
docker run -p 8501:8501 \
--add-host=host.docker.internal:host-gateway \
-e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
ai-document-assistant

### Then open
http://localhost:8501

## Project Structure
.
├── interface.py        # Streamlit interface and RAG application logic
├── Dockerfile          # Docker image configuration
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── documents/          # Local documents (not included)


