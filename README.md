# RAG Backend

Retrieval-Augmented Generation service for document-based search and answer generation.

## Features

- Document Ingestion: PDF, Markdown
- Embeddings: E5, BGE, Multilingual
- Vector Stores: FAISS, LanceDB, ChromaDB, SQLite-VSS
- Retrieval: Hybrid Search, Multi-query, Graph RAG, HyDE, Multi-hop, Contrastive
- Reranking: BGE reranker
- LLM: Ollama
- Multilingual: 20+ languages
- Caching: Semantic cache, embedding cache

## Installation

### Python Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Docker

The GPU service requires the **NVIDIA Container Toolkit**. Without it, Docker cannot access the graphics card.

```bash
# Install NVIDIA Container Toolkit (Fedora/RHEL)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

```bash
# With GPU
docker-compose up rag-backend-gpu

# Without GPU
docker-compose up rag-backend-cpu
```

## Configuration

Copy `.env.example` to `.env` and adjust the values:

```bash
PORT=8001
VECTOR_STORE_TYPE=faiss
EMBEDDING_MODEL=intfloat/e5-large-v2
LLM_MODEL=qwen3:8b
USE_RERANKING=true
USE_HYBRID_SEARCH=true
```

## Start

```bash
# Development
uvicorn rag.main:app --reload --host 0.0.0.0 --port 8001

# Production
gunicorn rag.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001
```

## API

### Health
```bash
GET /health
```

### Collections
```bash
GET    /collections
POST   /collections
DELETE /collections/{id}
```

### Ingestion
```bash
POST /ingest
Content-Type: multipart/form-data

collection_id: my_collection
files: <document.pdf>
```

### Query
```bash
POST /query
{
  "collection_id": "my_collection",
  "query": "What is the main topic?",
  "top_k": 10,
  "use_reranking": true
}
```

### Streaming
```bash
POST /query/stream
```

## Structure

```
rag-backend/
├── rag/
│   ├── main.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── pipeline_service.py
│   └── ...
├── config/
│   └── settings.py
├── utils/
│   └── gpu_queue_manager.py
├── data/
│   ├── corpus/
│   ├── faiss/
│   └── cache/
├── scripts/
│   ├── index_corpus.py
│   └── test_query.py
└── tests/
```

## Pipeline

```
Query → Language Detection → Query Classification
  ↓
HyDE / Multi-query
  ↓
Embedding Generation
  ↓
Vector Search + BM25
  ↓
Hybrid Fusion
  ↓
Reranking
  ↓
Context Compression
  ↓
LLM Generation
  ↓
Answer Verification
```

## Retrieval Strategies

- Basic Vector Search: Cosine similarity
- Hybrid Search: Vector + BM25 with RRF fusion
- Graph RAG: Knowledge graph enrichment
- HyDE: Hypothetical documents
- Multi-hop: Complex query decomposition
- Contrastive: Negation handling

## Indexing

### Via API
```bash
curl -X POST http://localhost:8001/ingest \
  -F "collection_id=my_docs" \
  -F "files=@document.pdf"
```

### Via Script
```bash
python scripts/index_corpus.py --collection my_docs --corpus-dir ./data/corpus/my_docs
```

## Troubleshooting

### Ollama
```bash
ollama serve
ollama pull qwen3:8b
```

### FAISS index
```bash
python scripts/index_corpus.py --collection <collection_id>
```

### GPU memory
- Reduce `EMBEDDING_BATCH_SIZE`
- Reduce `GPU_MEMORY_FRACTION`
- Enable `ENABLE_GPU_QUEUE=true`

## Performance

- Embedding: ~500 chunks/sec (GPU, batch 256)
- Retrieval: <100ms for 10k documents
- Reranking: ~50ms for top-10
- Generation: ~30 tokens/sec (qwen3:8b)

## Tests

```bash
python scripts/run_tests.py
pytest tests/
```
