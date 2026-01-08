"""
Query Router Module
Automatically selects optimal retrieval strategy based on query characteristics
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger

from config.settings import settings


class QueryType(str, Enum):
    """Query type classifications"""
    FACTUAL_SIMPLE = "factual_simple"
    FACTUAL_COMPLEX = "factual_complex"
    MULTI_HOP = "multi_hop"
    VAGUE_OR_SPARSE = "vague_or_sparse"
    EXPLORATORY = "exploratory"
    ENTITY_CENTRIC = "entity_centric"


@dataclass
class RetrievalStrategy:
    """Configuration for retrieval strategy"""
    use_hybrid_search: bool = True
    use_multi_query: bool = False
    use_hyde: bool = False
    use_graph_rag: bool = False
    use_reranking: bool = True
    num_query_variations: int = 2
    num_hypothetical_docs: int = 3
    hyde_fusion: str = "rrf"
    graph_expansion_depth: int = 1
    graph_alpha: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "use_hybrid_search": self.use_hybrid_search,
            "use_multi_query": self.use_multi_query,
            "use_hyde": self.use_hyde,
            "use_graph_rag": self.use_graph_rag,
            "use_reranking": self.use_reranking,
            "num_query_variations": self.num_query_variations,
            "num_hypothetical_docs": self.num_hypothetical_docs,
            "hyde_fusion": self.hyde_fusion,
            "graph_expansion_depth": self.graph_expansion_depth,
            "graph_alpha": self.graph_alpha,
        }


class QueryRouter:
    """
    Routes queries to optimal retrieval strategies based on query characteristics
    Supports both rule-based and LLM-based routing
    """

    def __init__(self, mode: str = "rules", llm_generator = None):
        """
        Initialize query router

        Args:
            mode: Routing mode ("rules" or "llm")
            llm_generator: LLM generator for LLM-based routing
        """
        self.mode = mode
        self.llm_generator = llm_generator

        # Question word patterns
        self.question_words = {
            "what": ["what", "which"],
            "how": ["how"],
            "why": ["why"],
            "when": ["when"],
            "where": ["where"],
            "who": ["who", "whom"],
        }

        # Multi-hop indicators
        self.multi_hop_patterns = [
            r"\band\b.*\b(what|how|why|when|where|who)",
            r"\bafter\b.*\b(what|how|why|when|where|who)",
            r"\bbefore\b.*\b(what|how|why|when|where|who)",
            r"\brelate",
            r"\bconnect",
            r"\bcompare",
            r"\bcontrast",
            r"difference between",
            r"relationship between",
            r"impact of.*on",
        ]

        # Entity indicators (proper nouns, specific names)
        self.entity_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",  # Multi-word proper nouns
            r"\b(?:CEO|CTO|CFO|President|Director)\b",  # Titles
            r"\b(?:Inc\.|Corp\.|LLC|Ltd\.)\b",  # Company suffixes
        ]

        # Exploratory query indicators
        self.exploratory_patterns = [
            r"\bexplore\b",
            r"\binvestigate\b",
            r"\bfind out\b",
            r"\btell me about\b",
            r"\blearn about\b",
            r"\bunderstand\b",
            r"\boverview\b",
            r"\bsummary\b",
            r"\bexplain\b",
        ]

    def route(self, query: str) -> tuple[QueryType, RetrievalStrategy]:
        """
        Route query to optimal retrieval strategy

        Args:
            query: User query

        Returns:
            Tuple of (query_type, retrieval_strategy)
        """
        if self.mode == "llm" and self.llm_generator:
            return self._route_with_llm(query)
        else:
            return self._route_with_rules(query)

    def _route_with_rules(self, query: str) -> tuple[QueryType, RetrievalStrategy]:
        """
        Rule-based query routing using heuristics

        Args:
            query: User query

        Returns:
            Tuple of (query_type, retrieval_strategy)
        """
        query_lower = query.lower()
        query_length = len(query.split())

        # Check for multi-hop reasoning
        if self._is_multi_hop(query, query_lower):
            logger.info(f"Classified as multi_hop query")
            return QueryType.MULTI_HOP, RetrievalStrategy(
                use_hybrid_search=True,
                use_graph_rag=True,
                use_reranking=True,
                graph_expansion_depth=2,
            )

        # Check for entity-centric queries
        if self._is_entity_centric(query):
            logger.info(f"Classified as entity_centric query")
            return QueryType.ENTITY_CENTRIC, RetrievalStrategy(
                use_hybrid_search=True,
                use_graph_rag=True,
                use_reranking=True,
                graph_expansion_depth=1,
            )

        # Check for exploratory queries
        if self._is_exploratory(query_lower):
            logger.info(f"Classified as exploratory query")
            return QueryType.EXPLORATORY, RetrievalStrategy(
                use_hybrid_search=True,
                use_multi_query=True,
                use_reranking=True,
                num_query_variations=3,
            )

        # Check for vague or sparse queries
        if self._is_vague_or_sparse(query, query_lower, query_length):
            logger.info(f"Classified as vague_or_sparse query")
            return QueryType.VAGUE_OR_SPARSE, RetrievalStrategy(
                use_hybrid_search=True,
                use_hyde=True,
                use_reranking=True,
                num_hypothetical_docs=3,
                hyde_fusion="rrf",
            )

        # Check for complex factual queries
        if self._is_factual_complex(query, query_lower, query_length):
            logger.info(f"Classified as factual_complex query")
            return QueryType.FACTUAL_COMPLEX, RetrievalStrategy(
                use_hybrid_search=True,
                use_reranking=True,
            )

        # Default: simple factual query (fast path)
        logger.info(f"Classified as factual_simple query (default)")
        return QueryType.FACTUAL_SIMPLE, RetrievalStrategy(
            use_hybrid_search=True,
            use_reranking=False,  # Fast path - skip reranking
        )

    def _is_multi_hop(self, query: str, query_lower: str) -> bool:
        """Check if query requires multi-hop reasoning"""
        for pattern in self.multi_hop_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _is_entity_centric(self, query: str) -> bool:
        """Check if query is entity-centric"""
        # Count proper nouns and entity patterns
        entity_count = 0
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entity_count += len(matches)

        # If query has multiple entities, it's entity-centric
        return entity_count >= 2

    def _is_exploratory(self, query_lower: str) -> bool:
        """Check if query is exploratory"""
        for pattern in self.exploratory_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _is_vague_or_sparse(self, query: str, query_lower: str, query_length: int) -> bool:
        """Check if query is vague or sparse"""
        # Very short queries (1-3 words) without specific question words
        if query_length <= 3:
            has_question_word = any(
                word in query_lower
                for words in self.question_words.values()
                for word in words
            )
            if not has_question_word:
                return True

        # Queries that are just keywords without structure
        if query_length >= 2 and query_length <= 5:
            # Check if it's mostly nouns/keywords (no verbs or structure)
            # Simple heuristic: no question words, no common verbs
            common_verbs = ["is", "are", "was", "were", "do", "does", "did", "can", "will", "should"]
            has_structure = any(verb in query_lower.split() for verb in common_verbs)
            has_question_word = any(
                word in query_lower
                for words in self.question_words.values()
                for word in words
            )

            if not has_structure and not has_question_word:
                return True

        return False

    def _is_factual_complex(self, query: str, query_lower: str, query_length: int) -> bool:
        """Check if query is a complex factual question"""
        # Long queries (>10 words) with question words
        if query_length > 10:
            has_question_word = any(
                word in query_lower
                for words in self.question_words.values()
                for word in words
            )
            if has_question_word:
                return True

        # Queries with "how" are often complex
        if any(word in query_lower.split() for word in self.question_words["how"]):
            return True

        # Queries with "why" are often complex
        if any(word in query_lower.split() for word in self.question_words["why"]):
            return True

        return False

    def _route_with_llm(self, query: str) -> tuple[QueryType, RetrievalStrategy]:
        """
        LLM-based query routing

        Args:
            query: User query

        Returns:
            Tuple of (query_type, retrieval_strategy)
        """
        prompt = f"""Classify the following query into one of these categories:

1. factual_simple: Simple, direct factual questions (e.g., "What is X?", "When did Y happen?")
2. factual_complex: Complex factual questions requiring detailed answers (e.g., "How does X work?", "Why does Y occur?")
3. multi_hop: Questions requiring multiple reasoning steps or connecting multiple pieces of information
4. vague_or_sparse: Vague, ambiguous, or keyword-based queries lacking clear structure
5. exploratory: Broad, open-ended queries seeking general information or overviews
6. entity_centric: Questions focused on specific entities, people, organizations, or locations

Query: "{query}"

Respond with ONLY the category name (one of the above). No explanation."""

        try:
            response = self.llm_generator.generate(prompt, max_tokens=50, temperature=0.0)
            category = response.strip().lower()

            # Map to QueryType
            if "factual_simple" in category:
                query_type = QueryType.FACTUAL_SIMPLE
            elif "factual_complex" in category:
                query_type = QueryType.FACTUAL_COMPLEX
            elif "multi_hop" in category:
                query_type = QueryType.MULTI_HOP
            elif "vague" in category or "sparse" in category:
                query_type = QueryType.VAGUE_OR_SPARSE
            elif "exploratory" in category:
                query_type = QueryType.EXPLORATORY
            elif "entity" in category:
                query_type = QueryType.ENTITY_CENTRIC
            else:
                logger.warning(f"Unknown LLM classification: {category}, falling back to rules")
                return self._route_with_rules(query)

            # Get strategy for this query type
            strategy = self._get_strategy_for_type(query_type)

            logger.info(f"LLM classified as {query_type.value}")
            return query_type, strategy

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, falling back to rules")
            return self._route_with_rules(query)

    def _get_strategy_for_type(self, query_type: QueryType) -> RetrievalStrategy:
        """Get retrieval strategy for a query type"""
        if query_type == QueryType.FACTUAL_SIMPLE:
            return RetrievalStrategy(
                use_hybrid_search=True,
                use_reranking=False,
            )
        elif query_type == QueryType.FACTUAL_COMPLEX:
            return RetrievalStrategy(
                use_hybrid_search=True,
                use_reranking=True,
            )
        elif query_type == QueryType.MULTI_HOP:
            return RetrievalStrategy(
                use_hybrid_search=True,
                use_graph_rag=True,
                use_reranking=True,
                graph_expansion_depth=2,
            )
        elif query_type == QueryType.VAGUE_OR_SPARSE:
            return RetrievalStrategy(
                use_hybrid_search=True,
                use_hyde=True,
                use_reranking=True,
                num_hypothetical_docs=3,
            )
        elif query_type == QueryType.EXPLORATORY:
            return RetrievalStrategy(
                use_hybrid_search=True,
                use_multi_query=True,
                use_reranking=True,
                num_query_variations=3,
            )
        elif query_type == QueryType.ENTITY_CENTRIC:
            return RetrievalStrategy(
                use_hybrid_search=True,
                use_graph_rag=True,
                use_reranking=True,
                graph_expansion_depth=1,
            )
        else:
            return RetrievalStrategy(use_hybrid_search=True, use_reranking=True)


def create_query_router(llm_generator = None) -> QueryRouter:
    """
    Factory function to create a query router

    Args:
        llm_generator: Optional LLM generator for LLM-based routing

    Returns:
        QueryRouter instance
    """
    return QueryRouter(
        mode=settings.ROUTER_MODE,
        llm_generator=llm_generator,
    )
