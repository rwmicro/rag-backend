# RAG Backend

Service de Retrieval-Augmented Generation pour la recherche et la génération de réponses basées sur des documents.

## Fonctionnalités

- Document Ingestion : PDF, Markdown
- Embeddings : E5, BGE, Multilingual
- Vector Stores : FAISS, LanceDB, ChromaDB, SQLite-VSS
- Retrieval : Hybrid Search, Multi-query, Graph RAG, HyDE, Multi-hop, Contrastive
- Reranking : BGE reranker
- LLM : Ollama
- Multilingue : 20+ langues
- Caching : Semantic cache, embedding cache

## Installation

### Environnement Python

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Docker

```bash
# Avec GPU
docker-compose up rag-backend-gpu

# Sans GPU
docker-compose up rag-backend-cpu
```

## Configuration

Fichier `.env` principal :

```bash
PORT=8000
VECTOR_STORE_TYPE=faiss
EMBEDDING_MODEL=intfloat/e5-large-v2
LLM_MODEL=qwen3:8b
USE_RERANKING=true
USE_HYBRID_SEARCH=true
```

## Démarrage

```bash
# Développement
uvicorn rag.main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn rag.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
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

## Stratégies de Retrieval

- Basic Vector Search : Similarité cosine
- Hybrid Search : Vector + BM25 avec fusion RRF
- Graph RAG : Enrichissement via graphe de connaissances
- HyDE : Documents hypothétiques
- Multi-hop : Décomposition de requêtes complexes
- Contrastive : Gestion des négations

## Indexation

### Via API
```bash
curl -X POST http://localhost:8000/ingest \
  -F "collection_id=my_docs" \
  -F "files=@document.pdf"
```

### Via Script
```bash
python scripts/index_corpus.py --collection my_docs --corpus-dir ./data/corpus/my_docs
```

## Dépannage

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
- Réduire `EMBEDDING_BATCH_SIZE`
- Réduire `GPU_MEMORY_FRACTION`
- Activer `ENABLE_GPU_QUEUE=true`

## Performance

- Embedding : ~500 chunks/sec (GPU, batch 256)
- Retrieval : <100ms pour 10k documents
- Reranking : ~50ms pour top-10
- Generation : ~30 tokens/sec (qwen3:8b)

## Tests

```bash
python scripts/run_tests.py
pytest tests/
```
