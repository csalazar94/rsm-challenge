# GenAI Engineer Take-Home Assignment: Python RAG Microservice with Langufuse Observability

---

## Objective

Design and implement a Python-based Retrieval-Augmented Generation (RAG) microservice that ingests a provided document, answers queries using an LLM, and includes full observability via Langufuse.

---

## Provided Material

**Document:**

- [Think Python](https://allendowney.github.io/ThinkPython/index.html)
- [PEP 8](https://peps.python.org/pep-0008/)

---

## Requirements

### 1. Indexing

- On application startup, load the PDF and split it into logical chunks.
- Generate embeddings for each chunk.
- Store embeddings in a vector index of your choice.

### 2. API Endpoints (FastAPI)

- `GET /health` → returns 200 OK.
- `POST /ingest` → reloads and reindexes the PDF.
- `POST /query` → accepts:

  ```json
  { "question": "<text>" }
  ```

  and returns:

  ```json
  {
    "answer": "<generated answer>",
    "sources": [
      { "page": <number>, "text": "<passage text>" },
      …
    ]
  }
  ```

### 3. LLM Integration with langrapgh/langchain

- Use a configurable LLM for answer synthesis (e.g. OpenAI API).
- Store configuration (API keys, endpoints) in environment variables.

### 4. Langufuse, Langsmith, etc. Observability

- Instrument spans for these operations:
  - PDF ingestion and chunking
  - Embedding computation
  - Similarity search
  - LLM inference calls
- Emit structured logs for incoming requests, errors, and significant events.
- Record and expose metrics (e.g. request counts, latency histograms, error rates).

---

## Deliverables

Provide a public GitHub repository containing:

- **Service Code:** Python implementation with type hints and modular structure.
- **Git History:** Commits showing incremental development.
- **Dockerfile:** Builds the microservice image.
- **docker-compose.yml:** Orchestrates the API, vector-store service(s), and any additional services you deem necessary.
- **README.md:** Instructions for building, running, and testing, including:
  - Required environment variables, etc
- **Extra credit:** Unit tests for ingestion and query logic.

---

## Evaluation Criteria

- **Functionality:** Answers are accurate, context-grounded, and sources are correctly returned.
- **Code Quality:** Readability, modular design, error handling, and use of type hints.
- **Observability:** Comprehensive Langufuse spans, structured logging, and meaningful metrics.
- **Deployment:** Docker and Compose setup runs successfully with a single command.
- **Documentation:** Clear and complete README enabling immediate evaluation.

---

## Timeline & Submission

- **Timeframe:** 2 days
- **Submission:** Public GitHub repository link with all required materials.
