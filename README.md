# RAG Backend

Service de Retrieval-Augmented Generation (RAG) pour la recherche et la génération de réponses basées sur des documents.

## 📚 Fonctionnalités

- **Document Ingestion** : PDF, Markdown avec préservation de la structure
- **Embeddings** : Multiple modèles (E5, BGE, Multilingual)
- **Vector Stores** : FAISS, LanceDB, ChromaDB, SQLite-VSS
- **Retrieval Avancé** :
  - Hybrid Search (Vector + BM25)
  - Multi-query retrieval
  - Graph RAG
  - HyDE (Hypothetical Document Embeddings)
  - Multi-hop retrieval
  - Contrastive retrieval
- **Reranking** : Cross-encoder BGE reranker
- **LLM Integration** : Ollama pour génération
- **Multilingue** : Support de 20+ langues
- **Caching** : Semantic cache, embedding cache

## 🚀 Démarrage Rapide

### Installation

```bash
cd /home/micro/rag-backend

# Créer environnement virtuel
python3.12 -m venv venv
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt

# Télécharger modèles spaCy
python -m spacy download en_core_web_sm
```

### Configuration

Le fichier `.env` est déjà configuré. Variables principales :

- `PORT=8001` - Port du service
- `VECTOR_STORE_TYPE=faiss` - Type de vector store
- `EMBEDDING_MODEL=intfloat/e5-large-v2` - Modèle d'embeddings
- `LLM_MODEL=qwen3:8b` - Modèle LLM (Ollama)
- `USE_RERANKING=true` - Activer reranking
- `USE_HYBRID_SEARCH=true` - Recherche hybride

### Démarrage

**Mode développement :**
```bash
./start.sh
```

**Mode production :**
```bash
./start-production.sh
```

## 📡 API Endpoints

### Health Check
```bash
GET http://localhost:8001/
GET http://localhost:8001/api/rag/health
```

### Collections
```bash
GET http://localhost:8001/collections
POST http://localhost:8001/collections
DELETE http://localhost:8001/collections/{collection_id}
```

### Document Ingestion
```bash
POST http://localhost:8001/ingest
Content-Type: multipart/form-data

collection_id: my_collection
files: <document.pdf>
```

### Query
```bash
POST http://localhost:8001/query
Content-Type: application/json

{
  "collection_id": "my_collection",
  "query": "What is the main topic?",
  "top_k": 10,
  "use_reranking": true,
  "use_hyde": false
}
```

### Streaming Query
```bash
POST http://localhost:8001/query/stream
```

## 🔧 Structure du Projet

```
rag-backend/
├── rag/                    # Modules RAG
│   ├── main.py             # FastAPI entry point
│   ├── embeddings.py       # Modèles d'embeddings
│   ├── vectordb.py         # Vector stores
│   ├── retrieval.py        # Stratégies de retrieval
│   ├── generation.py       # LLM generation
│   ├── reranking.py        # Reranking
│   ├── chunking.py         # Chunking strategies
│   ├── graph_rag.py        # Graph RAG
│   ├── hyde.py             # HyDE
│   ├── multi_hop_retrieval.py
│   └── [30+ autres modules]
├── config/                 # Configuration
│   └── settings.py
├── utils/                  # Utilitaires
│   └── gpu_queue_manager.py
├── data/                   # Données
│   ├── corpus/             # Documents sources
│   ├── faiss/              # Index FAISS
│   ├── cache/              # Caches
│   ├── feedback.db         # Feedback database
│   └── metadata.db         # Metadata database
├── scripts/                # Scripts utilitaires
│   ├── index_corpus.py
│   └── test_query.py
├── logs/                   # Logs
├── .env                    # Configuration
├── requirements.txt        # Dépendances
└── start-production.sh     # Script production
```

## 📊 Pipeline RAG

```
Query
  ↓
Language Detection
  ↓
Query Classification
  ↓
[Optional] HyDE / Multi-query
  ↓
Embedding Generation
  ↓
Vector Search (FAISS/LanceDB)
  ↓
[Optional] BM25 Search
  ↓
Hybrid Fusion
  ↓
Reranking (BGE reranker)
  ↓
[Optional] Graph Expansion
  ↓
Context Compression
  ↓
LLM Generation (Ollama)
  ↓
Answer Verification
  ↓
Response + Citations
```

## 🎯 Stratégies de Retrieval

### Basic Vector Search
Recherche par similarité cosine simple.

### Hybrid Search (Recommandé)
Combine vector search + BM25 avec fusion RRF.

### Graph RAG
Enrichit le contexte avec des relations extraites du graphe de connaissances.

### HyDE
Génère des documents hypothétiques pour améliorer la recherche.

### Multi-hop Retrieval
Décompose les requêtes complexes en sous-requêtes.

### Contrastive Retrieval
Gère les négations ("sans", "pas", "sauf").

## 🤝 Coordination avec Voice Backend

Si exécuté sur le même GPU que Voice Backend :

```bash
# Dans .env
ENABLE_GPU_QUEUE=true
VOICE_API_URL=http://localhost:8002
```

Le RAG Backend demandera à Voice de libérer le GPU si nécessaire.

## 🔍 Indexation de Documents

### Via API
```bash
curl -X POST http://localhost:8001/ingest \
  -F "collection_id=my_docs" \
  -F "files=@document.pdf"
```

### Via Script
```bash
python scripts/index_corpus.py \
  --collection my_docs \
  --corpus-dir ./data/corpus/my_docs
```

## 🐛 Dépannage

### Ollama not running
```bash
# Démarrer Ollama
ollama serve

# Vérifier modèles disponibles
ollama list

# Télécharger modèle si nécessaire
ollama pull qwen3:8b
```

### FAISS index not found
```bash
# Réindexer les documents
python scripts/index_corpus.py --collection <collection_id>
```

### GPU out of memory
- Réduire `EMBEDDING_BATCH_SIZE` dans `.env`
- Réduire `GPU_MEMORY_FRACTION`
- Activer `ENABLE_GPU_QUEUE=true`

### Import errors
```bash
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
python -m spacy download en_core_web_sm
```

## 📊 Performance

- **Embedding** : ~500 chunks/sec (GPU, batch 256)
- **Retrieval** : <100ms pour 10k documents
- **Reranking** : ~50ms pour top-10
- **Generation** : Variable selon LLM (qwen3:8b ~30 tokens/sec)

## 🔬 Évaluation

```bash
# Exécuter tests d'évaluation
python scripts/evaluate.py --collection test_collection
```

Métriques :
- Recall@K, Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Generation quality (BERTScore, ROUGE)

## 📝 License

Propriétaire
