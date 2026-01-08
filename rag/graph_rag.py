"""
Graph-based RAG Implementation
Builds semantic knowledge graphs from documents for enhanced retrieval
Based on Microsoft GraphRAG (2025)
"""

from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from loguru import logger
import pickle
import os
from pathlib import Path

from .chunking import Chunk
from .embeddings import EmbeddingModel

# spaCy for NER
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load spaCy model (will be lazy-loaded on first use)
    _nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, falling back to simple entity extraction")


@dataclass
class Entity:
    """Represents an entity extracted from text"""
    name: str
    entity_type: str  # PERSON, ORG, CONCEPT, etc.
    mentions: List[str]  # Different ways it's mentioned
    chunk_ids: Set[str]  # Chunks where it appears
    embedding: Optional[np.ndarray] = None


@dataclass
class Relation:
    """Represents a relation between entities"""
    source: str  # Entity name
    target: str  # Entity name
    relation_type: str  # RELATED_TO, PART_OF, MENTIONS, etc.
    weight: float = 1.0
    context: Optional[str] = None  # Context where relation was found


class MultilingualNER:
    """
    Multilingual Named Entity Recognition

    Supports 12+ languages with spaCy models:
    - English: en_core_web_sm
    - Spanish: es_core_news_sm
    - French: fr_core_news_sm
    - German: de_core_news_sm
    - Italian: it_core_news_sm
    - Portuguese: pt_core_news_sm
    - Dutch: nl_core_news_sm
    - Russian: ru_core_news_sm
    - Chinese: zh_core_web_sm
    - Japanese: ja_core_news_sm
    - Polish: pl_core_news_sm
    - Greek: el_core_news_sm
    - Multilingual (fallback): xx_ent_wiki_sm

    Usage:
        ner = MultilingualNER()
        entities = ner.extract_entities("Le président français Emmanuel Macron", language="fr")
    """

    # Language code to spaCy model mapping
    SPACY_MODELS = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm",
        "nl": "nl_core_news_sm",
        "ru": "ru_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
        "pl": "pl_core_news_sm",
        "el": "el_core_news_sm",
        # Fallback multilingual model
        "xx": "xx_ent_wiki_sm",
    }

    def __init__(self, default_language: str = "en"):
        """
        Initialize multilingual NER

        Args:
            default_language: Default language code (ISO 639-1)
        """
        self.default_language = default_language
        self.loaded_models = {}  # Cache loaded models
        self.language_detector = None

        # Try to import language detector
        try:
            from .language_detection import get_language_detector
            self.language_detector = get_language_detector()
            logger.info("Language detector loaded for multilingual NER")
        except ImportError:
            logger.warning("Language detector not available, using default language for NER")

        logger.info(f"MultilingualNER initialized (default={default_language})")

    def _get_model(self, language: str) -> Optional[Any]:
        """
        Get spaCy model for language (lazy loading)

        Args:
            language: ISO 639-1 language code

        Returns:
            Loaded spaCy model or None if not available
        """
        if not SPACY_AVAILABLE:
            return None

        # Use cached model if available
        if language in self.loaded_models:
            return self.loaded_models[language]

        # Get model name for language (or fallback to multilingual)
        model_name = self.SPACY_MODELS.get(language, self.SPACY_MODELS.get("xx"))

        if not model_name:
            logger.debug(f"No spaCy model for language '{language}', using fallback")
            model_name = self.SPACY_MODELS["xx"]

        # Try to load model
        try:
            nlp = spacy.load(model_name)
            self.loaded_models[language] = nlp
            logger.info(f"Loaded spaCy model {model_name} for language {language}")
            return nlp
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Download it with: python -m spacy download {model_name}"
            )

            # Try fallback to multilingual model
            if model_name != self.SPACY_MODELS["xx"]:
                try:
                    fallback_model = self.SPACY_MODELS["xx"]
                    nlp = spacy.load(fallback_model)
                    self.loaded_models[language] = nlp
                    logger.info(f"Using fallback multilingual model: {fallback_model}")
                    return nlp
                except OSError:
                    logger.warning(f"Fallback model not available. Install with: python -m spacy download {fallback_model}")

            return None
        except Exception as e:
            logger.error(f"Error loading spaCy model {model_name}: {e}")
            return None

    def extract_entities(
        self,
        text: str,
        language: Optional[str] = None,
        chunk_id: Optional[str] = None,
        max_entities: int = 10
    ) -> List[Entity]:
        """
        Extract named entities from text

        Args:
            text: Text to extract entities from
            language: Language code (auto-detected if None)
            chunk_id: Optional chunk ID to associate entities with
            max_entities: Maximum entities to extract

        Returns:
            List of Entity objects
        """
        if not text:
            return []

        # Detect language if not provided
        if language is None:
            if self.language_detector:
                language = self.language_detector.detect_with_fallback(
                    text,
                    default=self.default_language,
                    min_confidence=0.5
                )
            else:
                language = self.default_language

        # Get spaCy model for language
        nlp = self._get_model(language)

        if nlp is None:
            logger.debug(f"spaCy model not available for {language}, using fallback extraction")
            return self._fallback_entity_extraction(text, chunk_id, max_entities)

        # Process text with spaCy
        try:
            doc = nlp(text)

            entities = []
            for ent in doc.ents:
                # Filter by entity type - focus on important ones
                if ent.label_ in [
                    'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT',
                    'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP', 'FAC',
                    'LOC',  # Location
                    'DATE',  # Date
                    'MONEY',  # Monetary value
                ]:
                    chunk_ids = {chunk_id} if chunk_id else set()

                    entities.append(Entity(
                        name=ent.text,
                        entity_type=ent.label_,
                        mentions=[ent.text],
                        chunk_ids=chunk_ids
                    ))

            # Limit entities
            return entities[:max_entities]

        except Exception as e:
            logger.error(f"spaCy NER failed: {e}, using fallback")
            return self._fallback_entity_extraction(text, chunk_id, max_entities)

    def _fallback_entity_extraction(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        max_entities: int = 10
    ) -> List[Entity]:
        """
        Fallback entity extraction using simple heuristics

        Extracts capitalized multi-word phrases as potential entities.

        Args:
            text: Text to extract from
            chunk_id: Optional chunk ID
            max_entities: Maximum entities to extract

        Returns:
            List of Entity objects
        """
        entities = []
        words = text.split()
        current_entity = []

        for i, word in enumerate(words):
            # Check if word is capitalized and not at start of sentence
            is_capitalized = word[0].isupper() if word else False
            is_start_of_sentence = i == 0 or words[i-1].endswith('.')

            if is_capitalized and not is_start_of_sentence and len(word) > 2:
                current_entity.append(word)
            else:
                if len(current_entity) >= 2:  # Multi-word entity
                    entity_name = ' '.join(current_entity)
                    chunk_ids = {chunk_id} if chunk_id else set()

                    entities.append(Entity(
                        name=entity_name,
                        entity_type='UNKNOWN',
                        mentions=[entity_name],
                        chunk_ids=chunk_ids
                    ))

                current_entity = []

        # Don't forget last entity
        if len(current_entity) >= 2:
            entity_name = ' '.join(current_entity)
            chunk_ids = {chunk_id} if chunk_id else set()

            entities.append(Entity(
                name=entity_name,
                entity_type='UNKNOWN',
                mentions=[entity_name],
                chunk_ids=chunk_ids
            ))

        return entities[:max_entities]


class GraphRAG:
    """
    Graph-based RAG implementation

    Builds a knowledge graph from documents:
    - Nodes: Entities (concepts, persons, orgs) + Document chunks
    - Edges: Relations between entities + chunk-entity links

    Retrieval combines:
    - Vector similarity (embeddings)
    - Graph traversal (related entities)
    - Structural importance (PageRank)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        min_entity_mentions: int = 2,  # Minimum mentions to include entity
        max_entities_per_chunk: int = 10,
        use_pagerank: bool = True,
        cache_dir: Optional[str] = None,
        use_spacy_ner: bool = True,  # Use spaCy NER if available
        spacy_model: str = "en_core_web_sm",  # spaCy model name
    ):
        self.embedding_model = embedding_model
        self.min_entity_mentions = min_entity_mentions
        self.max_entities_per_chunk = max_entities_per_chunk
        self.use_pagerank = use_pagerank
        self.cache_dir = cache_dir or "data/graph_cache"
        self.use_spacy_ner = use_spacy_ner and SPACY_AVAILABLE
        self.spacy_model_name = spacy_model

        # Graph components
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.chunk_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.entity_embeddings: Dict[str, np.ndarray] = {}

        # spaCy NER pipeline (lazy-loaded)
        self.nlp = None

        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"GraphRAG initialized (cache_dir: {self.cache_dir}, use_spacy_ner: {self.use_spacy_ner})")

    def _load_spacy_model(self):
        """Lazy-load spaCy model"""
        if self.nlp is None and self.use_spacy_ner:
            try:
                logger.info(f"Loading spaCy model: {self.spacy_model_name}")
                self.nlp = spacy.load(self.spacy_model_name)
                logger.info(f"✓ spaCy model loaded: {self.spacy_model_name}")
            except OSError:
                logger.warning(
                    f"spaCy model '{self.spacy_model_name}' not found. "
                    f"Download it with: python -m spacy download {self.spacy_model_name}"
                )
                logger.warning("Falling back to simple entity extraction")
                self.use_spacy_ner = False
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                self.use_spacy_ner = False

    def build_graph(self, chunks: List[Chunk], cache_name: str = "default", force_rebuild: bool = False):
        """
        Build knowledge graph from chunks

        Steps:
        1. Try to load from cache (unless force_rebuild=True)
        2. Extract entities from each chunk
        3. Identify relations between entities
        4. Build graph structure
        5. Compute entity embeddings
        6. Calculate PageRank (optional)
        7. Save to cache

        Args:
            chunks: List of document chunks
            cache_name: Name for cache (e.g., collection name)
            force_rebuild: If True, rebuild even if cache exists
        """
        # Try to load from cache first
        if not force_rebuild:
            if self.load_cache(cache_name):
                logger.info(f"Graph loaded from cache, skipping build")
                return

        logger.info(f"Building graph from {len(chunks)} chunks...")

        # Step 1: Extract entities
        for chunk in chunks:
            entities = self._extract_entities(chunk)

            for entity in entities:
                # Add or update entity
                if entity.name in self.entities:
                    self.entities[entity.name].chunk_ids.update(entity.chunk_ids)
                    self.entities[entity.name].mentions.extend(entity.mentions)
                else:
                    self.entities[entity.name] = entity

                # Track chunk-entity mapping
                self.chunk_to_entities[chunk.chunk_id].append(entity.name)

        # Filter low-frequency entities
        self.entities = {
            name: entity
            for name, entity in self.entities.items()
            if len(entity.chunk_ids) >= self.min_entity_mentions
        }

        logger.info(f"Extracted {len(self.entities)} entities")

        # Step 2: Build graph structure
        self._build_graph_structure(chunks)

        # Step 3: Compute entity embeddings
        self._compute_entity_embeddings(chunks)

        # Step 4: Calculate PageRank
        if self.use_pagerank:
            self._calculate_importance()

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")

    def _extract_entities(self, chunk: Chunk) -> List[Entity]:
        """
        Extract entities from chunk using spaCy NER

        Uses spaCy's Named Entity Recognition to extract:
        - PERSON: People, including fictional
        - ORG: Companies, agencies, institutions
        - GPE: Geopolitical entities (countries, cities, states)
        - PRODUCT: Objects, vehicles, foods, etc.
        - EVENT: Named events
        - WORK_OF_ART: Titles of books, songs, etc.
        - LAW: Named documents made into laws
        - LANGUAGE: Any named language

        Also extracts from metadata (headers, titles) as concepts
        """
        entities = []
        text = chunk.content

        # Extract from metadata (headers, titles) - always do this
        if 'title' in chunk.metadata:
            entities.append(Entity(
                name=chunk.metadata['title'],
                entity_type='CONCEPT',
                mentions=[chunk.metadata['title']],
                chunk_ids={chunk.chunk_id}
            ))

        if 'section' in chunk.metadata:
            entities.append(Entity(
                name=chunk.metadata['section'],
                entity_type='CONCEPT',
                mentions=[chunk.metadata['section']],
                chunk_ids={chunk.chunk_id}
            ))

        # Use spaCy NER if available
        if self.use_spacy_ner:
            # Load model if not already loaded
            if self.nlp is None:
                self._load_spacy_model()

            if self.nlp is not None:
                # Process text with spaCy
                doc = self.nlp(text)

                # Extract named entities
                for ent in doc.ents:
                    # Filter by entity type - focus on important ones
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP', 'FAC']:
                        entities.append(Entity(
                            name=ent.text,
                            entity_type=ent.label_,
                            mentions=[ent.text],
                            chunk_ids={chunk.chunk_id}
                        ))

                # Limit entities per chunk
                return entities[:self.max_entities_per_chunk]

        # Fallback: Simple heuristic entity extraction if spaCy not available
        logger.debug("Using fallback entity extraction (spaCy not available)")
        words = text.split()
        current_entity = []

        for i, word in enumerate(words):
            # Check if word is capitalized and not at start of sentence
            is_capitalized = word[0].isupper() if word else False
            is_start_of_sentence = i == 0 or words[i-1].endswith('.')

            if is_capitalized and not is_start_of_sentence and len(word) > 2:
                current_entity.append(word)
            else:
                if len(current_entity) >= 2:  # Multi-word entity
                    entity_name = ' '.join(current_entity)
                    entities.append(Entity(
                        name=entity_name,
                        entity_type='ENTITY',
                        mentions=[entity_name],
                        chunk_ids={chunk.chunk_id}
                    ))
                current_entity = []

        # Limit entities per chunk
        return entities[:self.max_entities_per_chunk]

    def _build_graph_structure(self, chunks: List[Chunk]):
        """Build graph nodes and edges"""

        # Add entity nodes
        for entity_name, entity in self.entities.items():
            self.graph.add_node(
                entity_name,
                type='entity',
                entity_type=entity.entity_type,
                num_mentions=len(entity.chunk_ids)
            )

        # Add chunk nodes
        for chunk in chunks:
            self.graph.add_node(
                chunk.chunk_id,
                type='chunk',
                content=chunk.content[:200]  # Store preview
            )

        # Add edges: chunk -> entity
        for chunk_id, entity_names in self.chunk_to_entities.items():
            for entity_name in entity_names:
                if entity_name in self.entities:
                    self.graph.add_edge(
                        chunk_id,
                        entity_name,
                        type='mentions',
                        weight=1.0
                    )

        # Add edges: entity -> entity (co-occurrence)
        for chunk_id, entity_names in self.chunk_to_entities.items():
            entity_names = [e for e in entity_names if e in self.entities]

            # Connect entities that co-occur in same chunk
            for i, entity1 in enumerate(entity_names):
                for entity2 in entity_names[i+1:]:
                    # Add bidirectional edges
                    if not self.graph.has_edge(entity1, entity2):
                        self.graph.add_edge(
                            entity1,
                            entity2,
                            type='co_occurs',
                            weight=1.0
                        )
                        self.graph.add_edge(
                            entity2,
                            entity1,
                            type='co_occurs',
                            weight=1.0
                        )
                    else:
                        # Increase weight for repeated co-occurrence
                        self.graph[entity1][entity2]['weight'] += 1.0
                        self.graph[entity2][entity1]['weight'] += 1.0

    def _compute_entity_embeddings(self, chunks: List[Chunk]):
        """
        Compute embeddings for entities based on their context

        Entity embedding = average of embeddings of chunks mentioning it
        """
        logger.info("Computing entity embeddings...")

        chunk_embeddings = {
            chunk.chunk_id: np.array(chunk.embedding)
            for chunk in chunks
            if chunk.embedding is not None
        }

        for entity_name, entity in self.entities.items():
            # Get embeddings of chunks mentioning this entity
            related_embeddings = [
                chunk_embeddings[chunk_id]
                for chunk_id in entity.chunk_ids
                if chunk_id in chunk_embeddings
            ]

            if related_embeddings:
                # Average embeddings
                entity_embedding = np.mean(related_embeddings, axis=0)

                # Normalize
                entity_embedding = entity_embedding / (np.linalg.norm(entity_embedding) + 1e-8)

                self.entity_embeddings[entity_name] = entity_embedding
                entity.embedding = entity_embedding

    def _calculate_importance(self):
        """Calculate entity importance using PageRank"""
        logger.info("Calculating entity importance (PageRank)...")

        try:
            # PageRank on entity subgraph
            entity_subgraph = self.graph.subgraph([
                node for node, data in self.graph.nodes(data=True)
                if data.get('type') == 'entity'
            ])

            if entity_subgraph.number_of_nodes() > 0:
                pagerank = nx.pagerank(entity_subgraph, weight='weight')

                # Store in graph
                nx.set_node_attributes(self.graph, pagerank, 'importance')

                logger.info(f"PageRank computed for {len(pagerank)} entities")
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")

    def retrieve_with_graph(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: int = 10,
        expansion_depth: int = 1,
        alpha: float = 0.7,  # Weight for vector similarity vs graph proximity
    ) -> List[Tuple[Chunk, float]]:
        """
        Enhanced retrieval using graph structure

        Algorithm:
        1. Start with initial vector search results
        2. Identify entities in top chunks
        3. Expand via graph traversal (related entities)
        4. Retrieve chunks connected to expanded entities
        5. Combine scores: vector similarity + graph proximity + importance

        Args:
            query: Search query
            chunks_with_scores: Initial retrieval results (from vector search)
            top_k: Final number of results
            expansion_depth: How many hops to expand in graph
            alpha: Weight for vector vs graph scores

        Returns:
            Reranked chunks with combined scores
        """
        if not chunks_with_scores:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        # Step 1: Get entities from top chunks
        initial_chunks = [chunk for chunk, _ in chunks_with_scores[:top_k]]
        seed_entities = set()

        for chunk in initial_chunks:
            if chunk.chunk_id in self.chunk_to_entities:
                seed_entities.update(self.chunk_to_entities[chunk.chunk_id])

        # Filter to known entities
        seed_entities = {e for e in seed_entities if e in self.entities}

        if not seed_entities:
            logger.warning("No entities found in initial results, returning vector results")
            return chunks_with_scores[:top_k]

        logger.info(f"Seed entities: {len(seed_entities)}")

        # Step 2: Expand via graph (find related entities)
        expanded_entities = self._expand_entities(seed_entities, expansion_depth)
        logger.info(f"Expanded to {len(expanded_entities)} entities")

        # Step 3: Get chunks connected to expanded entities
        candidate_chunks = set()
        for entity_name in expanded_entities:
            if entity_name in self.entities:
                candidate_chunks.update(self.entities[entity_name].chunk_ids)

        # Step 4: Score chunks
        chunk_scores = {}
        chunk_map = {chunk.chunk_id: chunk for chunk, _ in chunks_with_scores}

        for chunk_id in candidate_chunks:
            if chunk_id not in chunk_map:
                continue

            chunk = chunk_map[chunk_id]

            # Vector similarity score
            if chunk.embedding is not None:
                chunk_emb = np.array(chunk.embedding)
                vector_score = float(np.dot(query_embedding, chunk_emb))
            else:
                vector_score = 0.0

            # Graph proximity score (average similarity to seed entities)
            chunk_entities = [
                e for e in self.chunk_to_entities.get(chunk_id, [])
                if e in expanded_entities
            ]

            if chunk_entities:
                entity_scores = []
                for entity_name in chunk_entities:
                    if entity_name in self.entity_embeddings:
                        entity_emb = self.entity_embeddings[entity_name]
                        entity_sim = float(np.dot(query_embedding, entity_emb))

                        # Boost by importance if available
                        importance = self.graph.nodes[entity_name].get('importance', 1.0)
                        entity_scores.append(entity_sim * importance)

                graph_score = np.mean(entity_scores) if entity_scores else 0.0
            else:
                graph_score = 0.0

            # Combined score
            combined_score = alpha * vector_score + (1 - alpha) * graph_score
            chunk_scores[chunk_id] = combined_score

        # Step 5: Sort and return top-k
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in sorted_chunks
            if chunk_id in chunk_map
        ]

        logger.info(f"Graph-enhanced retrieval: {len(results)} results")
        return results

    def _expand_entities(
        self,
        seed_entities: Set[str],
        depth: int
    ) -> Set[str]:
        """
        Expand entities via graph traversal

        Performs BFS from seed entities to find related entities
        """
        expanded = set(seed_entities)
        current_layer = seed_entities

        for _ in range(depth):
            next_layer = set()

            for entity in current_layer:
                if entity in self.graph:
                    # Get neighbors (related entities)
                    neighbors = [
                        neighbor
                        for neighbor in self.graph.neighbors(entity)
                        if self.graph.nodes[neighbor].get('type') == 'entity'
                    ]
                    next_layer.update(neighbors)

            expanded.update(next_layer)
            current_layer = next_layer

            if not current_layer:
                break

        return expanded

    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get contextual information about an entity"""
        if entity_name not in self.entities:
            return {}

        entity = self.entities[entity_name]

        # Get related entities
        related = []
        if entity_name in self.graph:
            for neighbor in self.graph.neighbors(entity_name):
                if self.graph.nodes[neighbor].get('type') == 'entity':
                    edge_data = self.graph[entity_name][neighbor]
                    related.append({
                        'entity': neighbor,
                        'relation': edge_data.get('type', 'related'),
                        'weight': edge_data.get('weight', 1.0)
                    })

        return {
            'name': entity.name,
            'type': entity.entity_type,
            'num_mentions': len(entity.chunk_ids),
            'importance': self.graph.nodes[entity_name].get('importance', 0.0),
            'related_entities': sorted(related, key=lambda x: x['weight'], reverse=True)[:10]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            'num_entities': len(self.entities),
            'num_chunks': sum(1 for n, d in self.graph.nodes(data=True) if d.get('type') == 'chunk'),
            'num_edges': self.graph.number_of_edges(),
            'avg_entities_per_chunk': float(np.mean([
                len(entities) for entities in self.chunk_to_entities.values()
            ])) if self.chunk_to_entities else 0.0,
            'top_entities': self._get_top_entities(10)
        }

    def _get_top_entities(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most important entities"""
        entity_data = [
            {
                'name': name,
                'mentions': len(entity.chunk_ids),
                'importance': float(self.graph.nodes[name].get('importance', 0.0))
            }
            for name, entity in self.entities.items()
        ]

        sorted_entities = sorted(
            entity_data,
            key=lambda x: (x['importance'], x['mentions']),
            reverse=True
        )

        return sorted_entities[:n]

    def save_cache(self, cache_name: str = "default"):
        """
        Save graph state to cache

        Args:
            cache_name: Name for this cache (e.g., collection name)
        """
        cache_path = Path(self.cache_dir) / f"{cache_name}.pkl"

        try:
            cache_data = {
                'graph': self.graph,
                'entities': self.entities,
                'relations': self.relations,
                'chunk_to_entities': dict(self.chunk_to_entities),
                'entity_embeddings': self.entity_embeddings,
                'min_entity_mentions': self.min_entity_mentions,
                'max_entities_per_chunk': self.max_entities_per_chunk,
                'use_pagerank': self.use_pagerank,
                'use_spacy_ner': self.use_spacy_ner,
                'spacy_model_name': self.spacy_model_name,
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Graph cache saved to {cache_path} "
                       f"({len(self.entities)} entities, {self.graph.number_of_edges()} edges)")
            return True

        except Exception as e:
            logger.error(f"Failed to save graph cache: {e}")
            return False

    def load_cache(self, cache_name: str = "default") -> bool:
        """
        Load graph state from cache

        Args:
            cache_name: Name of cache to load

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_path = Path(self.cache_dir) / f"{cache_name}.pkl"

        if not cache_path.exists():
            logger.info(f"No cache found at {cache_path}")
            return False

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Restore graph state
            self.graph = cache_data['graph']
            self.entities = cache_data['entities']
            self.relations = cache_data['relations']
            self.chunk_to_entities = defaultdict(list, cache_data['chunk_to_entities'])
            self.entity_embeddings = cache_data['entity_embeddings']
            self.min_entity_mentions = cache_data['min_entity_mentions']
            self.max_entities_per_chunk = cache_data['max_entities_per_chunk']
            self.use_pagerank = cache_data['use_pagerank']

            # Restore spaCy settings (backward compatible)
            cached_use_spacy = cache_data.get('use_spacy_ner', False)
            cached_spacy_model = cache_data.get('spacy_model_name', 'en_core_web_sm')

            # Log if there's a configuration mismatch
            if cached_use_spacy != self.use_spacy_ner:
                logger.warning(
                    f"Cache was built with use_spacy_ner={cached_use_spacy}, "
                    f"but current setting is {self.use_spacy_ner}"
                )
            if cached_spacy_model != self.spacy_model_name:
                logger.warning(
                    f"Cache was built with spacy_model={cached_spacy_model}, "
                    f"but current model is {self.spacy_model_name}"
                )

            logger.info(f"Graph cache loaded from {cache_path} "
                       f"({len(self.entities)} entities, {self.graph.number_of_edges()} edges, "
                       f"spacy_ner={cached_use_spacy})")
            return True

        except Exception as e:
            logger.error(f"Failed to load graph cache: {e}")
            return False

    def clear_cache(self, cache_name: Optional[str] = None):
        """
        Clear graph cache

        Args:
            cache_name: Specific cache to clear, or None to clear all caches
        """
        try:
            if cache_name:
                cache_path = Path(self.cache_dir) / f"{cache_name}.pkl"
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"Cleared cache: {cache_name}")
            else:
                # Clear all caches
                cache_dir = Path(self.cache_dir)
                for cache_file in cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info(f"Cleared all caches in {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
