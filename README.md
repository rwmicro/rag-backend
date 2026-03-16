# RAG Backend

Retrieval-Augmented Generation service for document-based search and answer generation.

## Features

- **Ingestion**: PDF, Markdown, URL, local folder
- **Async ingestion**: non-blocking upload with job polling
- **Deduplication**: SHA-256 hash check, skips unchanged documents
- **Embeddings**: E5, BGE, Multilingual (20+ models)
- **Vector stores**: FAISS, LanceDB, ChromaDB, SQLite-VSS
- **Retrieval**: Hybrid (vector + BM25), Graph RAG, HyDE, Multi-query, Multi-hop, Contrastive
- **Reranking**: BGE cross-encoder
- **LLM**: Ollama (default), any OpenAI-compatible API
- **Multilingual**: 20+ languages in retrieval, 100+ in detection
- **Caching**: semantic cache, embedding cache
- **Per-request LLM override**: model, provider, URL, timeout

## Requirements

- Docker + NVIDIA Container Toolkit (GPU mode)
- [Ollama](https://ollama.com) running locally

## Start

```bash
# GPU
docker compose up -d rag-backend-gpu

# CPU only
docker compose up -d rag-backend-cpu
```

### NVIDIA Container Toolkit (first-time setup, Fedora/RHEL)

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Configuration

Copy `.env.example` to `.env`:

```bash
VECTOR_STORE_TYPE=faiss
EMBEDDING_MODEL=intfloat/multilingual-e5-large
LLM_MODEL=llama3.2
USE_RERANKING=true
USE_HYBRID_SEARCH=true
```

## API

### Health
```
GET /health
```

### Collections
```
GET    /collections
GET    /collections/{id}
PATCH  /collections/{id}?llm_model=llama3.2
DELETE /collections/{id}
DELETE /collections/{id}/documents/{filename}
```

### Ingestion

**Single file (blocking)**
```bash
curl -X POST http://localhost:8001/ingest/file \
  -F "file=@document.pdf" \
  -F "collection_title=my_docs" \
  -F "chunk_size=1000" \
  -F "chunking_strategy=semantic"
```

**Single file (async — recommended for large files)**
```bash
# Submit
curl -X POST http://localhost:8001/ingest/file/async \
  -F "file=@document.pdf" \
  -F "collection_title=my_docs"
# → { "job_id": "abc-123", "status": "queued" }

# Poll status
curl http://localhost:8001/ingest/jobs/abc-123
# → { "status": "completed", "progress": 1.0, "result": { "num_chunks": 42 } }
```

**Local folder (async)**
```bash
curl -X POST http://localhost:8001/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "./data/corpus/my_docs",
    "collection_title": "my_docs",
    "recursive": true
  }'
```

**URL**
```bash
curl -X POST http://localhost:8001/ingest/url \
  -F "url=https://example.com/page" \
  -F "collection_title=my_docs"
```

### Query

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "collection_id": "my_docs",
    "top_k": 5,
    "use_hybrid_search": true,
    "use_reranking": true,
    "stream": false
  }'
```

**Streaming (SSE)**
```
POST /query/stream
```

**Per-request LLM override**
```json
{
  "query": "...",
  "collection_id": "my_docs",
  "llm_model_override": "llama3.2",
  "llm_provider": "ollama",
  "llm_base_url_override": "http://localhost:11434",
  "llm_timeout": 30
}
```

**Advanced retrieval options**
```json
{
  "use_hyde": true,
  "use_graph_rag": true,
  "use_multi_query": true,
  "enable_multi_hop": true,
  "enable_multilingual": true
}
```

### Stats & cache
```
GET    /stats
DELETE /cache
DELETE /cache/graph
```

## Pipeline

```
Query
  → Semantic cache check
  → Conversation contextualization
  → Query routing (optional)
  → Retrieval: HyDE / Multi-query / Hybrid (vector + BM25)
  → Graph RAG enhancement (optional)
  → Reranking + deduplication
  → MMR diversity (optional)
  → Context compression (optional)
  → LLM generation (streaming or blocking)
  → Answer verification (optional)
```

## Project structure

```
rag-backend/
├── rag/
│   ├── main.py              # FastAPI app + all endpoints
│   ├── collections_db.py    # SQLite collections store
│   ├── job_store.py         # Async job tracking
│   ├── doc_registry.py      # Document deduplication
│   ├── ingest.py            # PDF/Markdown parsing
│   ├── chunking.py          # Chunking strategies
│   ├── embeddings.py        # Embedding models
│   ├── vectordb.py          # Vector store abstraction
│   ├── retrieval.py         # Hybrid retrieval + BM25
│   ├── graph_rag.py         # Knowledge graph RAG
│   ├── hyde.py              # Hypothetical Document Embeddings
│   ├── generation.py        # LLM generation + streaming
│   ├── llm_provider.py      # Ollama / OpenAI-compatible
│   └── ...
├── config/
│   └── settings.py
├── data/                    # Runtime data (auto-created)
│   ├── faiss/
│   ├── graph_cache/
│   ├── collections.db
│   ├── jobs.db
│   └── doc_registry.db
├── tests/
│   ├── test_integration.py
│   └── ...
└── docker-compose.yml
```

## Troubleshooting

**Ollama not reachable from container**
```bash
ollama serve
# The container uses host.docker.internal:11434
```

**GPU out of memory**
```bash
# In .env
EMBEDDING_BATCH_SIZE=64
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Port already in use**
```bash
docker ps --filter "publish=8001"
docker stop <container_id>
```

## Tests

```bash
# Unit tests
pytest tests/ -m "not integration"

# Integration tests (requires running server)
pytest tests/test_integration.py -v -m integration
```
