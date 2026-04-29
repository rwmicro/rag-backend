"""Pydantic request/response schemas for the RAG API.

Extracted from rag/main.py to keep the app module focused on wiring
endpoints to the pipeline. Kept as a single file because the schemas
are small and tightly coupled to the public HTTP surface.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Message in conversation history"""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""

    query: str = Field(..., description="Search query")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list, description="Previous conversation for context"
    )
    collection_id: Optional[str] = Field(
        None, description="Collection ID to query (uses collection's LLM model)"
    )
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    auto_route: bool = Field(
        False, description="Automatically select optimal retrieval strategy"
    )
    use_hybrid_search: bool = Field(
        True, description="Use hybrid search (vector + BM25)"
    )
    use_multi_query: bool = Field(
        False, description="Use multi-query retrieval (generate query variations)"
    )
    num_query_variations: int = Field(
        2, description="Number of query variations for multi-query", ge=1, le=5
    )
    use_reranking: bool = Field(True, description="Apply reranking")
    use_compression: bool = Field(False, description="Apply context compression")
    use_graph_rag: bool = Field(
        False, description="Use Graph RAG for enhanced retrieval"
    )
    graph_expansion_depth: int = Field(
        1, description="Graph expansion depth for Graph RAG", ge=1, le=3
    )
    graph_alpha: float = Field(
        0.7, description="Weight for vector vs graph scores (0-1)", ge=0.0, le=1.0
    )
    use_hyde: bool = Field(
        False, description="Use HyDE (Hypothetical Document Embeddings)"
    )
    hyde_fusion: str = Field(
        "rrf", description="HyDE fusion method: 'average', 'max', or 'rrf'"
    )
    num_hypothetical_docs: int = Field(
        3, description="Number of hypothetical docs for HyDE", ge=1, le=5
    )
    use_adaptive_fusion: bool = Field(
        False, description="Use adaptive embedding fusion"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filters (e.g., {'file_type': 'pdf', 'date': {'$gte': '2024-01-01'}})",
    )
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    stream: bool = Field(True, description="Stream the response")

    # Advanced RAG features
    enable_query_classification: bool = Field(
        False, description="Enable query classification and routing"
    )
    enable_adaptive_alpha: bool = Field(
        False, description="Enable adaptive hybrid search alpha"
    )
    enable_mmr: bool = Field(False, description="Enable MMR diversity enforcement")
    mmr_lambda: Optional[float] = Field(
        None, description="MMR lambda parameter (0-1), uses settings default if None"
    )
    enable_contrastive: bool = Field(
        False, description="Enable contrastive retrieval for negation handling"
    )
    enable_multi_hop: bool = Field(
        False, description="Enable multi-hop retrieval for complex queries"
    )
    max_hops: int = Field(
        3, description="Maximum reasoning hops for multi-hop retrieval", ge=1, le=5
    )
    enable_answer_verification: bool = Field(
        False, description="Enable answer verification before presenting"
    )
    verification_threshold: Optional[float] = Field(
        None, description="Minimum verification score (0-1)"
    )
    enable_confidence_evaluation: bool = Field(
        True,
        description=(
            "Grade retrieval confidence. When the backend has corrective RAG "
            "enabled, low confidence triggers a query rewrite + retry."
        ),
    )
    enable_feedback_logging: bool = Field(
        True, description="Enable feedback and performance logging"
    )

    # Multilingual features
    enable_multilingual: bool = Field(
        False, description="Enable full multilingual pipeline"
    )
    query_language: Optional[str] = Field(
        None,
        description="Query language code (auto-detected if None, e.g., 'en', 'fr', 'es')",
    )
    use_multilingual_embeddings: bool = Field(
        False,
        description="Use multilingual-e5-large embeddings for cross-lingual retrieval",
    )
    use_multilingual_bm25: bool = Field(
        False, description="Use language-specific BM25 tokenization"
    )
    use_multilingual_hyde: bool = Field(
        False, description="Generate hypothetical documents in query's language"
    )
    use_multilingual_classifier: bool = Field(
        False, description="Use multilingual query classification patterns"
    )
    detect_language: bool = Field(
        True, description="Automatically detect query language"
    )

    # LLM override per request
    llm_provider: Optional[str] = Field(
        None,
        description="Override LLM provider: 'ollama' or 'openai_compatible'",
    )
    llm_model_override: Optional[str] = Field(
        None,
        description="Override LLM model for this request (e.g., 'llama3.2', 'gpt-4o')",
    )
    llm_base_url_override: Optional[str] = Field(
        None,
        description="Override LLM base URL for this request",
    )
    llm_timeout: Optional[int] = Field(
        None,
        description="LLM call timeout in seconds (5-300)",
        ge=5,
        le=300,
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint"""

    answer: str
    sources: List[Dict[str, Any]]
    query: str
    metadata: Dict[str, Any] = {}
    llm_model: Optional[str] = None
    collection_id: Optional[str] = None


class IngestRequest(BaseModel):
    """Request model for /ingest endpoint"""

    recursive: bool = Field(True, description="Process subdirectories recursively")
    chunk_size: int = Field(1000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    chunking_strategy: str = Field("semantic", description="Chunking strategy")


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint"""

    success: bool
    message: str
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """Response model for /stats endpoint"""

    total_chunks: int
    total_files: int
    embedding_model: str
    llm_model: Optional[str] = None
    vector_store_type: str
    cache_stats: Dict[str, Any]


class EvaluationRequest(BaseModel):
    """Request model for /evaluate endpoint"""

    test_dataset: List[Dict[str, Any]] = Field(
        ..., description="List of evaluation samples"
    )
    collection_id: Optional[str] = Field(None, description="Collection to evaluate on")
    k_values: List[int] = Field([1, 3, 5, 10], description="K values for @k metrics")
    evaluate_generation: bool = Field(
        False, description="Also evaluate generation quality"
    )


class EvaluationResponse(BaseModel):
    """Response model for /evaluate endpoint"""

    retrieval_metrics: Dict[str, Any]
    generation_metrics: Optional[Dict[str, Any]] = None
    sample_count: int
    timestamp: str


class AsyncIngestResponse(BaseModel):
    """Response model for async ingest endpoint"""

    job_id: str
    status: str = "queued"
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status endpoint"""

    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class IngestFolderRequest(BaseModel):
    """Request model for folder ingestion"""

    folder_path: str = Field(
        ..., description="Absolute or relative path to folder on server"
    )
    collection_title: str = Field(..., description="Collection name/ID")
    llm_model: Optional[str] = Field(None, description="LLM model to use")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    chunk_size: int = Field(1000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    chunking_strategy: str = Field("semantic", description="Chunking strategy")
    embedding_model_name: Optional[str] = Field(None, description="Embedding model to use")
