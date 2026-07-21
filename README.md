# RAG Backend

Retrieval-Augmented Generation service for document-based search and answer generation.

## Features

- **Ingestion**: PDF, Markdown, URL, local folder
- **Async ingestion**: non-blocking upload with job polling
- **Deduplication**: SHA-256 hash check, skips unchanged documents
- **Embeddings**: E5, BGE, Multilingual (20+ models)
- **Vector stores**: FAISS, LanceDB, ChromaDB, SQLite-VSS
- **Retrieval**: Hybrid (vector + BM25), Graph RAG, HyDE, Multi-query, Multi-hop, Contrastive
- **Corrective RAG**: grades the retrieved set and retries with a rewritten query
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

Deleting a document rebuilds the FAISS index from the embeddings that remain.
If the stored embeddings file is missing or out of sync the rebuild cannot be
done without discarding everything, so the request is refused with **409** and
nothing is modified — re-ingest the collection to regenerate it, then retry.

### Ingestion

**Single file (blocking)**
```bash
curl -X POST http://localhost:8001/ingest/file \
  -F "file=@document.pdf" \
  -F "collection_title=my_docs" \
  -F "chunk_size=1000" \
  -F "chunking_strategy=semantic"
```

`chunk_size` and `chunk_overlap` are counted in **tokens**, not characters.
Fragments shorter than `min_chunk_size` (default `max(50, chunk_size // 2)`
tokens) are dropped, so a small `chunk_size` on short documents can legitimately
yield no chunks.

Strategies are `semantic`, `recursive` and `markdown`. `smart` — which picks one
from the file type and whether the document has headers — is only resolved by
this endpoint; the async, folder, URL and directory routes pass the value
straight through and reject it.

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
  → Corrective loop: grade + retry with a rewritten query (optional)
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
│   ├── main.py              # FastAPI app, lifespan, query/ingest/collection endpoints
│   ├── routers/             # Health, jobs, cache, config, stats, models, evaluate
│   ├── app_state.py         # Lazy singletons for the SQLite stores
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
│   ├── faiss/               # index.faiss + .metadata.pkl + .embeddings.npy
│   ├── graph_cache/
│   ├── collections.db
│   ├── jobs.db
│   ├── doc_registry.db
│   ├── metadata.db
│   └── feedback.db
├── tests/
│   ├── conftest.py          # Redirects the stores to a temp dir per test
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

**Do not raise the uvicorn worker count**

The FAISS index and its metadata are in-process state flushed to `./data/faiss`.
A second worker holds its own copy of the same collection and overwrites the
first's writes on flush, silently losing chunks. The Dockerfile pins
`--workers 1` for this reason. Concurrency comes from threads instead: requests
that do model work run in the threadpool, and the stores are lock-guarded.

## Tests

The pinned dependencies (`numpy<2.0`, `torch<2.10`, `sentence-transformers<3.0`,
`spacy<4`) have no wheels for recent Python versions, so run the suite inside the
project image rather than on the host:

```bash
docker build --target runtime-cpu -t rag-test:cpu .

docker run --rm -v "$(pwd)":/app -w /app rag-test:cpu \
  sh -c "pip install -q -r requirements-dev.txt; pytest tests/ -q"
```

Mounting the working tree means you test current code without rebuilding.

```bash
pytest tests/ -m "not integration"   # 80 tests
pytest tests/ -m integration         # 14 tests, end-to-end through the app
```

Integration tests do **not** need a running server — they drive the app in-process
with `TestClient`, which also runs the lifespan startup. They do download real
models on first run (e5-large, ~2.2 GB), as do the Graph RAG tests, so mount a
cache to keep reruns cheap:

```bash
-v /tmp/hfcache:/tmp/hfcache -e HF_HOME=/tmp/hfcache -e EMBEDDING_DEVICE=cpu
```

Tests redirect every store to a temp directory (`tests/conftest.py`), so a run
never touches `./data/`. A native shutdown crash in torch/faiss can swallow
pytest's final tally — use `--junitxml` if you need reliable counts.
