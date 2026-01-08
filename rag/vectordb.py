"""
Vector Store Module
Supports LanceDB, FAISS, and ChromaDB
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import pickle
from loguru import logger

# Import vector stores
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
import faiss
from chromadb import Client as ChromaClient
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from .chunking import Chunk
from .metadata_filter import MetadataFilter, build_chroma_filter


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            metadata_filter: Optional metadata filters (e.g., {"file_type": "pdf", "date": {"$gte": "2024-01-01"}})

        Returns:
            List of (chunk, score) tuples, sorted by score descending
        """
        pass

    @abstractmethod
    def delete_by_filename(self, filename: str) -> None:
        """Delete all chunks from a specific file"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass


class LanceDBStore(VectorStore):
    """
    LanceDB vector store implementation
    Modern, fast, and disk-based
    """

    def __init__(
        self,
        uri: str = "./data/lancedb",
        table_name: str = "documents",
        dimension: int = 1024,
    ):
        self.uri = uri
        self.table_name = table_name
        self.dimension = dimension

        # Connect to LanceDB
        self.db = lancedb.connect(uri)

        # Create or get table
        try:
            self.table = self.db.open_table(table_name)
            logger.info(f"Opened existing LanceDB table: {table_name}")
        except Exception:
            # Table doesn't exist, will be created on first add
            self.table = None
            logger.info(f"Will create new LanceDB table: {table_name}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to LanceDB"""
        if not chunks:
            return

        import json

        # Prepare data
        data = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")

            # Serialize metadata to JSON string to avoid schema conflicts
            # LanceDB can have issues with varying metadata structures
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"

            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": metadata_json,
                    "vector": chunk.embedding,
                }
            )

        # Create or append to table
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=data)
            logger.info(f"Created LanceDB table with {len(chunks)} chunks")
        else:
            self.table.add(data)
            logger.info(f"Added {len(chunks)} chunks to LanceDB")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search LanceDB with optional metadata filtering"""
        if self.table is None:
            return []

        import json

        # Convert to list if numpy
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Start search query
        search_query = self.table.search(query_embedding).limit(
            top_k * 3 if metadata_filter else top_k
        )

        # Apply metadata filter if provided (post-processing since metadata is JSON string)
        results = search_query.to_list()

        # Convert to Chunk objects and filter
        chunks_with_scores = []
        for result in results:
            # Deserialize metadata from JSON string
            metadata = result.get("metadata", "{}")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Apply metadata filter
            if metadata_filter and not self._matches_filter(metadata, metadata_filter):
                continue

            chunk = Chunk(
                chunk_id=result["chunk_id"],
                content=result["content"],
                metadata=metadata,
                embedding=result.get("vector"),
            )
            # LanceDB returns _distance (L2 distance)
            # Convert to similarity score (1 / (1 + distance))
            distance = result.get("_distance", 0)
            score = 1.0 / (1.0 + distance)

            chunks_with_scores.append((chunk, score))

            # Stop when we have enough results
            if len(chunks_with_scores) >= top_k:
                break

        return chunks_with_scores

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            # Handle operators like $gte, $lte, $gt, $lt, $ne
            if isinstance(value, dict):
                metadata_value = metadata[key]
                for op, op_value in value.items():
                    if op == "$gte" and metadata_value < op_value:
                        return False
                    elif op == "$lte" and metadata_value > op_value:
                        return False
                    elif op == "$gt" and metadata_value <= op_value:
                        return False
                    elif op == "$lt" and metadata_value >= op_value:
                        return False
                    elif op == "$ne" and metadata_value == op_value:
                        return False
                    elif op == "$in" and metadata_value not in op_value:
                        return False
            else:
                # Direct equality check
                if metadata[key] != value:
                    return False

        return True

    def delete_by_filename(self, filename: str) -> None:
        """Delete chunks by filename"""
        if self.table is None:
            return

        import json

        # Since metadata is stored as JSON string, we need to filter differently
        # Get all data and filter in Python
        try:
            df = self.table.to_pandas()

            # Find chunk_ids to delete
            chunk_ids_to_delete = []
            for idx, row in df.iterrows():
                metadata_json = row.get("metadata", "{}")
                try:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json

                    if metadata.get("filename") == filename:
                        chunk_ids_to_delete.append(row["chunk_id"])
                except (json.JSONDecodeError, TypeError):
                    continue

            # Delete by chunk_id
            if chunk_ids_to_delete:
                for chunk_id in chunk_ids_to_delete:
                    # Escape single quotes in chunk_id
                    escaped_id = chunk_id.replace("'", "''")
                    self.table.delete(f"chunk_id = '{escaped_id}'")
                logger.info(
                    f"Deleted {len(chunk_ids_to_delete)} chunks for file: {filename}"
                )
            else:
                logger.warning(f"No chunks found for file: {filename}")
        except Exception as e:
            logger.error(f"Error deleting chunks for {filename}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if self.table is None:
            return {"total_chunks": 0, "total_files": 0}

        import json

        count = self.table.count_rows()

        # Get unique filenames
        try:
            all_data = self.table.to_pandas()
            unique_files = set()
            for metadata_json in all_data["metadata"]:
                try:
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                        if "filename" in metadata:
                            unique_files.add(metadata["filename"])
                    elif isinstance(metadata_json, dict):
                        if "filename" in metadata_json:
                            unique_files.add(metadata_json["filename"])
                except (json.JSONDecodeError, TypeError):
                    continue
        except Exception as e:
            logger.warning(f"Error getting unique files: {e}")
            unique_files = set()

        return {
            "total_chunks": count,
            "total_files": len(unique_files),
        }


class FAISSStore(VectorStore):
    """
    FAISS vector store implementation
    High-performance similarity search
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_path: Optional[str] = None,
        use_gpu: bool = False,
        metadata_store=None,  # Optional MetadataStore for pre-filtering
    ):
        self.dimension = dimension
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.use_gpu = use_gpu
        self.metadata_store = metadata_store

        # Create or load index
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index from {self.index_path}")
        else:
            # Create HNSW index (fast and accurate)
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            logger.info(f"Created new FAISS HNSW index (dim={dimension})")

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("Moved FAISS index to GPU")

        # Store chunks metadata separately
        self.chunks_metadata = []
        # Also maintain chunk_id to index mapping for efficient lookup
        self.chunk_id_to_index = {}
        # Store embeddings separately for visualization (HNSW doesn't support reconstruction)
        self.stored_embeddings = None

        metadata_path = Path(self.index_path).with_suffix(".metadata.pkl")
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.chunks_metadata = pickle.load(f)
            # Build chunk_id to index mapping
            for idx, entry in enumerate(self.chunks_metadata):
                self.chunk_id_to_index[entry["chunk_id"]] = idx
            logger.info(f"Loaded {len(self.chunks_metadata)} chunk metadata entries")

        # Load stored embeddings if available
        embeddings_path = Path(self.index_path).with_suffix(".embeddings.npy")
        if embeddings_path.exists():
            try:
                self.stored_embeddings = np.load(embeddings_path)
                logger.info(
                    f"Loaded {len(self.stored_embeddings)} stored embeddings for visualization"
                )
            except Exception as e:
                logger.warning(f"Failed to load stored embeddings: {e}")
                self.stored_embeddings = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to FAISS"""
        if not chunks:
            return

        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Get current index size (for building chunk_id mapping)
        start_idx = self.index.ntotal

        # Add to index
        self.index.add(embeddings)

        # Store metadata and update mapping
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks_metadata.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
            )
            self.chunk_id_to_index[chunk.chunk_id] = idx

            # Also add to metadata store if available
            if self.metadata_store and chunk.metadata:
                self.metadata_store.add_chunk_metadata(chunk.chunk_id, chunk.metadata)

        logger.info(f"Added {len(chunks)} chunks to FAISS")

        # Save index, metadata, AND embeddings (for visualization)
        self._save(embeddings=embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search FAISS with optional metadata filtering"""
        if self.index.ntotal == 0:
            return []

        # Prepare query
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Search with MANY more results if filtering (Hakka content may rank low semantically)
        search_k = min(500, self.index.ntotal) if metadata_filter else top_k
        scores, indices = self.index.search(query, search_k)

        # Convert to Chunk objects and apply filter
        chunks_with_scores = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue

            metadata_entry = self.chunks_metadata[idx]

            # Apply metadata filter
            if metadata_filter and not self._matches_filter(
                metadata_entry["metadata"], metadata_filter
            ):
                continue

            chunk = Chunk(
                chunk_id=metadata_entry["chunk_id"],
                content=metadata_entry["content"],
                metadata=metadata_entry["metadata"],
            )

            # FAISS returns inner product (for normalized vectors, this is cosine similarity)
            chunks_with_scores.append((chunk, float(score)))

            # Stop when we have enough results
            if len(chunks_with_scores) >= top_k:
                break

        return chunks_with_scores

    def search_with_metadata_store(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filters: Optional[List] = None,  # List of MetadataFilter objects
    ) -> List[Tuple[Chunk, float]]:
        """
        Search FAISS with metadata pre-filtering using MetadataStore.
        More efficient than post-filtering when metadata_store is available.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            metadata_filters: List of MetadataFilter objects from metadata_store module

        Returns:
            List of (chunk, score) tuples
        """
        if self.index.ntotal == 0:
            return []

        # If no metadata store or no filters, fall back to regular search
        if not self.metadata_store or not metadata_filters:
            return self.search(query_embedding, top_k)

        # Get eligible chunk IDs from metadata store
        eligible_chunk_ids = self.metadata_store.filter(metadata_filters)

        if not eligible_chunk_ids:
            logger.warning("No chunks match metadata filters")
            return []

        # Convert chunk IDs to FAISS indices
        eligible_indices = []
        for chunk_id in eligible_chunk_ids:
            if chunk_id in self.chunk_id_to_index:
                eligible_indices.append(self.chunk_id_to_index[chunk_id])

        if not eligible_indices:
            logger.warning("No eligible chunks found in FAISS index")
            return []

        logger.info(
            f"Pre-filtered to {len(eligible_indices)} eligible chunks from {len(metadata_filters)} filters"
        )

        # Create ID selector for FAISS
        eligible_indices_np = np.array(eligible_indices, dtype=np.int64)
        id_selector = faiss.IDSelectorArray(
            len(eligible_indices_np), faiss.swig_ptr(eligible_indices_np)
        )

        # Prepare query
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Create search parameters with selector
        params = faiss.SearchParametersHNSW()
        params.sel = id_selector

        # Search only within eligible IDs
        search_k = min(top_k * 2, len(eligible_indices))  # Over-fetch slightly
        scores, indices = self.index.search(query, search_k, params=params)

        # Convert to Chunk objects
        chunks_with_scores = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue

            metadata_entry = self.chunks_metadata[idx]

            chunk = Chunk(
                chunk_id=metadata_entry["chunk_id"],
                content=metadata_entry["content"],
                metadata=metadata_entry["metadata"],
            )

            chunks_with_scores.append((chunk, float(score)))

            if len(chunks_with_scores) >= top_k:
                break

        logger.debug(
            f"Metadata pre-filtering returned {len(chunks_with_scores)} results"
        )
        return chunks_with_scores

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria"""
        return MetadataFilter.matches(metadata, filter_dict)

    def delete_by_filename(self, filename: str) -> None:
        """Delete chunks by filename (requires rebuilding index)"""
        # FAISS doesn't support deletion, so we rebuild
        new_chunks_metadata = [
            m for m in self.chunks_metadata if m["metadata"].get("filename") != filename
        ]

        if len(new_chunks_metadata) == len(self.chunks_metadata):
            return  # Nothing to delete

        # Rebuild index
        self.chunks_metadata = new_chunks_metadata
        self._rebuild_index()

        logger.info(f"Deleted chunks for file: {filename}")

    def _rebuild_index(self):
        """Rebuild FAISS index from metadata"""
        # Create new index
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        # Note: This requires embeddings to be stored in metadata
        # For now, we'll just clear the index
        # In production, you'd want to store embeddings and rebuild

        logger.warning("FAISS index cleared (rebuild requires stored embeddings)")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        unique_files = set(m["metadata"].get("filename") for m in self.chunks_metadata)

        return {
            "total_chunks": self.index.ntotal,
            "total_files": len(unique_files),
        }

    def _save(self, embeddings: Optional[np.ndarray] = None):
        """Save index, metadata, and optionally embeddings"""
        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        metadata_path = Path(self.index_path).with_suffix(".metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks_metadata, f)

        # Save embeddings for visualization (append to existing)
        if embeddings is not None:
            embeddings_path = Path(self.index_path).with_suffix(".embeddings.npy")
            if self.stored_embeddings is not None:
                # Append to existing embeddings
                self.stored_embeddings = np.vstack([self.stored_embeddings, embeddings])
            else:
                self.stored_embeddings = embeddings

            np.save(embeddings_path, self.stored_embeddings)
            logger.info(
                f"Saved {len(self.stored_embeddings)} embeddings for visualization"
            )


class ChromaDBStore(VectorStore):
    """ChromaDB vector store (compatible with existing TypeScript implementation)"""

    def __init__(
        self,
        collection_name: str = "ragcoon_documents",
        host: str = "localhost",
        port: int = 8000,
    ):
        self.collection_name = collection_name

        # Connect to ChromaDB
        self.client = ChromaClient(
            Settings=ChromaSettings(
                chroma_api_impl="rest",
                chroma_server_host=host,
                chroma_server_http_port=port,
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Connected to ChromaDB collection: {collection_name}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to ChromaDB"""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search ChromaDB with optional metadata filtering"""
        # ChromaDB has native metadata filtering support
        where_clause = None
        if metadata_filter:
            where_clause = self._build_chroma_filter(metadata_filter)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
        )

        chunks_with_scores = []
        for i in range(len(results["ids"][0])):
            chunk = Chunk(
                chunk_id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )

            # ChromaDB returns distance, convert to similarity
            distance = results["distances"][0][i]
            score = 1.0 - distance

            chunks_with_scores.append((chunk, score))

        return chunks_with_scores

    def _build_chroma_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filter dict to ChromaDB where clause format"""
        return build_chroma_filter(filter_dict)

    def delete_by_filename(self, filename: str) -> None:
        """Delete chunks by filename"""
        self.collection.delete(where={"filename": filename})
        logger.info(f"Deleted chunks for file: {filename}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        count = self.collection.count()
        return {"total_chunks": count, "total_files": 0}


class SQLiteVectorStore(VectorStore):
    """
    SQLite vector store with sqlite-vss extension
    Single-file portable vector database
    """

    def __init__(
        self,
        db_path: str = "./data/rag.db",
        dimension: int = 1024,
    ):
        self.db_path = db_path
        self.dimension = dimension

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Import sqlite-vss
        try:
            try:
                import pysqlite3 as sqlite3
            except ImportError:
                import sqlite3
            import sqlite_vss

            self.sqlite3 = sqlite3
            self.sqlite_vss = sqlite_vss
        except ImportError:
            logger.error("sqlite-vss not installed. Run: pip install sqlite-vss")
            raise

        # Connect to database
        self.conn = self.sqlite3.connect(db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        self.sqlite_vss.load(self.conn)
        self.conn.enable_load_extension(False)

        # Create tables
        self._init_tables()

        logger.info(f"Connected to SQLite vector store: {db_path}")

    def _init_tables(self):
        """Initialize database tables and vector index"""
        cursor = self.conn.cursor()

        # Create main chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create vectors table for embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            )
        """)

        # Create virtual table for vector search using sqlite-vss
        try:
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vss0(
                    embedding({self.dimension})
                )
            """)
        except Exception as e:
            logger.warning(f"Could not create VSS table (may already exist): {e}")

        # Create index on metadata for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata ON chunks(metadata)
        """)

        self.conn.commit()
        logger.info("SQLite tables initialized")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to SQLite"""
        if not chunks:
            return

        import json

        cursor = self.conn.cursor()

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")

            # Serialize metadata to JSON
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"

            # Insert into chunks table
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, content, metadata)
                VALUES (?, ?, ?)
            """,
                (chunk.chunk_id, chunk.content, metadata_json),
            )

            # Convert embedding to bytes
            embedding_bytes = np.array(chunk.embedding, dtype=np.float32).tobytes()

            # Insert into vectors table
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding)
                VALUES (?, ?)
            """,
                (chunk.chunk_id, embedding_bytes),
            )

            # Insert into VSS table
            try:
                # Get rowid for the chunk
                cursor.execute(
                    "SELECT rowid FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
                )
                rowid = cursor.fetchone()
                if rowid:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO vss_chunks (rowid, embedding)
                        VALUES (?, ?)
                    """,
                        (rowid[0], embedding_bytes),
                    )
            except Exception as e:
                logger.warning(f"Could not insert into VSS table: {e}")

        self.conn.commit()
        logger.info(f"Added {len(chunks)} chunks to SQLite")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Search SQLite using vector similarity with optional metadata filtering"""
        import json

        cursor = self.conn.cursor()

        # Convert query embedding to bytes
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        # Try VSS search first
        try:
            # Search with more results if filtering
            search_k = top_k * 3 if metadata_filter else top_k

            # sqlite-vss uses vss_search_params() to pass query and limit
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.content,
                    c.metadata,
                    v.distance
                FROM vss_chunks v
                JOIN chunks c ON c.rowid = v.rowid
                WHERE vss_search(v.embedding, vss_search_params(?, {search_k}))
            """,
                (query_bytes,),
            )

            results = cursor.fetchall()

            chunks_with_scores = []
            for chunk_id, content, metadata_json, distance in results:
                # Deserialize metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}

                # Apply metadata filter
                if metadata_filter and not self._matches_filter(
                    metadata, metadata_filter
                ):
                    continue

                chunk = Chunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                )

                # Convert distance to similarity score
                score = 1.0 / (1.0 + distance)
                chunks_with_scores.append((chunk, score))

                # Stop when we have enough results
                if len(chunks_with_scores) >= top_k:
                    break

            return chunks_with_scores

        except Exception as e:
            logger.warning(f"VSS search failed, falling back to brute force: {e}")
            # Fallback to brute force search
            return self._brute_force_search(query_embedding, top_k, metadata_filter)

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria"""
        return MetadataFilter.matches(metadata, filter_dict)

    def _brute_force_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Fallback brute force vector search with optional filtering"""
        import json

        cursor = self.conn.cursor()

        # Get all chunks and embeddings
        cursor.execute("""
            SELECT c.chunk_id, c.content, c.metadata, cv.embedding
            FROM chunks c
            JOIN chunk_vectors cv ON c.chunk_id = cv.chunk_id
        """)

        results = cursor.fetchall()

        if not results:
            return []

        # Calculate cosine similarity for all
        scores = []
        for chunk_id, content, metadata_json, embedding_bytes in results:
            # Deserialize metadata
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError:
                metadata = {}

            # Apply metadata filter
            if metadata_filter and not self._matches_filter(metadata, metadata_filter):
                continue

            # Deserialize embedding
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )

            chunk = Chunk(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
            )

            scores.append((chunk, float(similarity)))

        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete_by_filename(self, filename: str) -> None:
        """Delete chunks by filename"""
        import json

        cursor = self.conn.cursor()

        # Find chunks with matching filename in metadata
        cursor.execute("SELECT chunk_id, metadata FROM chunks")
        rows = cursor.fetchall()

        chunk_ids_to_delete = []
        for chunk_id, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                if metadata.get("filename") == filename:
                    chunk_ids_to_delete.append(chunk_id)
            except json.JSONDecodeError:
                continue

        # Delete chunks
        if chunk_ids_to_delete:
            placeholders = ",".join("?" * len(chunk_ids_to_delete))
            cursor.execute(
                f"""
                DELETE FROM chunks WHERE chunk_id IN ({placeholders})
            """,
                chunk_ids_to_delete,
            )

            # VSS table will be cleaned up via trigger or manually
            try:
                cursor.execute(f"""
                    DELETE FROM vss_chunks
                    WHERE rowid NOT IN (SELECT rowid FROM chunks)
                """)
            except Exception:
                pass

            self.conn.commit()
            logger.info(
                f"Deleted {len(chunk_ids_to_delete)} chunks for file: {filename}"
            )
        else:
            logger.warning(f"No chunks found for file: {filename}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        import json

        cursor = self.conn.cursor()

        # Get total chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        # Get unique filenames
        cursor.execute("SELECT metadata FROM chunks")
        rows = cursor.fetchall()

        unique_files = set()
        for (metadata_json,) in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                if "filename" in metadata:
                    unique_files.add(metadata["filename"])
            except json.JSONDecodeError:
                continue

        return {
            "total_chunks": total_chunks,
            "total_files": len(unique_files),
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed SQLite connection")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


def create_vector_store(
    store_type: Optional[str] = None, collection_id: Optional[str] = None, **kwargs
) -> VectorStore:
    """
    Factory function to create vector store

    Args:
        store_type: Type of store ("lancedb", "faiss", "chroma", "sqlite")
        collection_id: Collection ID (for per-collection stores like SQLite)
        **kwargs: Store-specific arguments

    Returns:
        VectorStore instance
    """
    store_type = store_type or settings.VECTOR_STORE_TYPE

    if store_type == "lancedb":
        # Extract uri from kwargs or use settings
        uri = kwargs.pop("uri", settings.LANCEDB_URI)
        table_name = kwargs.pop("table_name", "documents")
        dimension = kwargs.pop("dimension", settings.EMBEDDING_DIMENSION)

        return LanceDBStore(
            uri=uri, table_name=table_name, dimension=dimension, **kwargs
        )
    elif store_type == "faiss":
        # Extract dimension from kwargs or use settings default
        dimension = kwargs.pop("dimension", settings.EMBEDDING_DIMENSION)
        index_path = kwargs.pop("index_path", settings.FAISS_INDEX_PATH)

        return FAISSStore(dimension=dimension, index_path=index_path, **kwargs)
    elif store_type == "chroma":
        return ChromaDBStore(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, **kwargs
        )
    elif store_type == "sqlite":
        # Generate per-collection database path if collection_id is provided
        if collection_id and "db_path" not in kwargs:
            base_dir = Path(
                getattr(settings, "SQLITE_VECTOR_DB_PATH", "./data/rag.db")
            ).parent
            db_path = str(base_dir / f"rag-{collection_id}.db")
        else:
            db_path = kwargs.pop(
                "db_path", getattr(settings, "SQLITE_VECTOR_DB_PATH", "./data/rag.db")
            )

        dimension = kwargs.pop("dimension", settings.EMBEDDING_DIMENSION)

        return SQLiteVectorStore(db_path=db_path, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
