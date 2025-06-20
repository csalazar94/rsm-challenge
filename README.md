# RAG Microservice with FastAPI and PostgreSQL

A Python-based Retrieval-Augmented Generation (RAG) microservice that ingests multiple document formats, generates embeddings, and answers queries using OpenAI's language models with full observability.

## Overview

This microservice provides:

- PDF and RST document ingestion and chunking
- Batch processing of multiple documents
- Vector embeddings storage in PostgreSQL with pgvector
- Multi-query retrieval for improved document search
- Question answering with source attribution
- Comprehensive logging and error handling
- Docker containerization

## Architecture

- **FastAPI**: REST API framework
- **PostgreSQL + pgvector**: Vector database for embeddings storage
- **LangChain**: Document processing and LLM integration
- **OpenAI**: Embeddings and chat completions
- **Unstructured**: Advanced PDF processing
- **Pandoc**: Document format conversion support
- **Sentry**: Error monitoring (optional)
- **LangSmith**: Observability (optional)

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- LangSmith API key (optional)
- Python 3.12+ (for local development)

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/csalazar94/rsm-challenge
   cd rsm-challenge
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Add your PDF document**

   ```bash
   # Create a files directory and place your documents there
   mkdir -p files
   cp your-document.pdf files/
   cp your-guide.rst files/
   ```

4. **Start the services**

   ```bash
   docker-compose up -d
   ```

5. **Ingest your documents**

   ```bash
   curl -X POST http://localhost:3000/ingest
   ```

6. **Query the system**
   ```bash
   curl -X POST http://localhost:3000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
   ```

## API Endpoints

### Health Check

```http
GET /health
```

Returns `200 OK` if the service is healthy.

### Document Ingestion

```http
POST /ingest
```

Processes all documents in the `files` directory and stores embeddings in the vector database.

**Response:**

```json
null
```

### Query Documents

```http
POST /query
```

**Request Body:**

```json
{
  "question": "Your question here"
}
```

**Response:**

```json
{
  "answer": "Generated answer based on document content",
  "sources": [
    {
      "filename": "document.pdf",
      "page": 1,
      "text": "Relevant passage from document"
    }
  ]
}
```

## Features

### Advanced Document Processing

- **Multi-format Support**: PDF and reStructuredText (RST) files
- **Batch Processing**: Automatically processes all files in the `files` directory
- High-resolution text extraction for PDFs
- Table structure inference for PDFs
- Metadata filtering
- Configurable chunking strategies

### Multi-Query Retrieval

- Automatic question translation to English
- Query diversification for better retrieval
- Similarity score thresholding
- Duplicate document filtering

### Comprehensive Logging

- Structured JSON logging
- Configurable log levels
- Error monitoring with Sentry

### Production Ready

- Database connection pooling
- Async operations throughout
- Docker containerization
- Environment-based configuration

## Configuration

### Document Processing

- **Chunk Size**: Default 1500 characters
- **Chunk Overlap**: Default 0 characters (no overlap)
- **PDF Processing Strategy**: High-resolution with table inference
- **RST Processing**: Clean whitespace with pandoc support

### Retrieval

- **Score Threshold**: 0.3 (configurable)
- **Max Documents**: 10 per query
- **Multi-query**: Enabled for better retrieval

### LLM Settings

- **Temperature**: 0.2 for consistent responses
- **Model**: gpt-4.1-mini (configurable)
- **Embeddings**: text-embedding-3-large

## Monitoring and Observability

The application includes comprehensive logging:

- Request/response logging
- LangSmith integration for observability (optional)
- Sentry for error tracking (optional)

## Troubleshooting

### Common Issues

1. **Document Processing Fails**

   - Ensure documents exist in the `files/` directory
   - Supported formats: PDF (.pdf) and reStructuredText (.rst)
   - Check file size (max 50MB per file)
   - Verify files are not corrupted
   - Ensure pandoc is installed for RST processing

2. **Database Connection Issues**

   - Verify PostgreSQL is running with pgvector extension
   - Check database connection parameters
   - Ensure network connectivity between containers

3. **OpenAI API Errors**

   - Verify API key is valid and has sufficient credits
   - Check rate limits
   - Ensure model names are correct

4. **No Relevant Documents Found**
   - Lower the similarity score threshold
   - Check if documents were properly ingested
   - Verify embedding model consistency

### Debug Endpoints

```http
GET /debug_error
```

Triggers a test error for Sentry integration testing.
