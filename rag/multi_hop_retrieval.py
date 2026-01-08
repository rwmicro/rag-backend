"""
Multi-Hop Retrieval System

Handles complex queries requiring multiple reasoning steps by:
1. Decomposing complex queries into sub-queries
2. Retrieving information for each sub-query
3. Synthesizing results across hops
4. Building a reasoning chain
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from loguru import logger

from .chunking import Chunk
from config.settings import settings


class HopType(Enum):
    """Types of reasoning hops"""
    INITIAL = "initial"           # First query
    BRIDGE = "bridge"             # Connects two pieces of information
    COMPARISON = "comparison"     # Compares retrieved information
    AGGREGATION = "aggregation"   # Aggregates multiple results
    VERIFICATION = "verification" # Verifies consistency


@dataclass
class RetrievalHop:
    """Single hop in multi-hop retrieval"""
    hop_id: int
    hop_type: HopType
    query: str
    parent_hop_ids: List[int]  # Dependencies
    results: List[Tuple[Chunk, float]] = field(default_factory=list)
    entities_found: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class MultiHopPlan:
    """Plan for multi-hop retrieval"""
    original_query: str
    hops: List[RetrievalHop]
    reasoning_type: str  # "sequential", "parallel", "hierarchical"
    confidence: float


class QueryDecomposer:
    """
    Decomposes complex queries into sub-queries for multi-hop retrieval

    Handles:
    - Sequential reasoning: "What is X's Y, and then what is Y's Z?"
    - Comparison queries: "Compare X and Y on aspect Z"
    - Aggregation: "Summarize all information about X"
    - Complex conditions: "Find X where condition1 AND condition2"
    """

    def __init__(self, llm_generator=None):
        """
        Initialize query decomposer

        Args:
            llm_generator: Optional LLM for intelligent decomposition
        """
        self.llm_generator = llm_generator

        # Patterns for detecting multi-hop queries
        self.sequential_patterns = [
            r'(\w+)(?:\'s|s\')\s+(\w+).*?(?:and|then).*?(\w+)',  # "X's Y and then Z"
            r'based on.*?(?:and|then)',  # "Based on X, then Y"
            r'after.*?(?:and|then)',  # "After X, then Y"
        ]

        self.comparison_patterns = [
            r'\bcompare\b.*?\band\b',
            r'\bdifference between\b.*?\band\b',
            r'\bversus\b|\bvs\.?\b',
            r'\bboth\b.*?\band\b',
        ]

        self.aggregation_patterns = [
            r'\ball\b.*?\babout\b',
            r'\beverything\b.*?\brelated to\b',
            r'\bsummarize\b',
            r'\blist all\b',
        ]

        logger.info("Initialized QueryDecomposer")

    def decompose(
        self,
        query: str,
        max_hops: int = 3,
    ) -> MultiHopPlan:
        """
        Decompose query into multi-hop plan

        Args:
            query: Complex query
            max_hops: Maximum number of hops

        Returns:
            MultiHopPlan with sub-queries
        """
        query_lower = query.lower()

        # Detect query type
        is_sequential = any(re.search(p, query_lower) for p in self.sequential_patterns)
        is_comparison = any(re.search(p, query_lower) for p in self.comparison_patterns)
        is_aggregation = any(re.search(p, query_lower) for p in self.aggregation_patterns)

        # Determine reasoning type
        if is_sequential:
            reasoning_type = "sequential"
            hops = self._decompose_sequential(query, max_hops)
        elif is_comparison:
            reasoning_type = "parallel"
            hops = self._decompose_comparison(query)
        elif is_aggregation:
            reasoning_type = "hierarchical"
            hops = self._decompose_aggregation(query)
        else:
            # Try LLM-based decomposition
            if self.llm_generator:
                reasoning_type = "llm_based"
                hops = self._decompose_with_llm(query, max_hops)
            else:
                # Fallback: single hop
                reasoning_type = "single"
                hops = [RetrievalHop(
                    hop_id=0,
                    hop_type=HopType.INITIAL,
                    query=query,
                    parent_hop_ids=[],
                    reasoning="Single-hop query (no decomposition needed)"
                )]

        confidence = 0.9 if len(hops) > 1 else 0.5

        logger.info(
            f"Query decomposition: {reasoning_type} with {len(hops)} hops "
            f"(confidence={confidence:.2f})"
        )

        return MultiHopPlan(
            original_query=query,
            hops=hops,
            reasoning_type=reasoning_type,
            confidence=confidence,
        )

    def _decompose_sequential(self, query: str, max_hops: int) -> List[RetrievalHop]:
        """
        Decompose sequential reasoning query

        Example: "What company did the founder of SpaceX start before SpaceX?"
        -> Hop 1: "Who is the founder of SpaceX?"
        -> Hop 2: "What company did [founder] start before SpaceX?"
        """
        hops = []

        # Pattern: "What is X's Y?"
        # Hop 1: Get X
        # Hop 2: Get Y of X

        # Simple heuristic: split on "and then", "and", "then"
        split_patterns = [r'\s+and then\s+', r'\s+then\s+', r'\s+and\s+']

        sub_queries = [query]
        for pattern in split_patterns:
            parts = re.split(pattern, query, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_queries = parts
                break

        # Create hops
        for i, sub_query in enumerate(sub_queries[:max_hops]):
            hop = RetrievalHop(
                hop_id=i,
                hop_type=HopType.BRIDGE if i > 0 else HopType.INITIAL,
                query=sub_query.strip(),
                parent_hop_ids=[i-1] if i > 0 else [],
                reasoning=f"Sequential step {i+1}: {sub_query.strip()}"
            )
            hops.append(hop)

        return hops

    def _decompose_comparison(self, query: str) -> List[RetrievalHop]:
        """
        Decompose comparison query

        Example: "Compare the GDP of USA and China"
        -> Hop 1: "GDP of USA"
        -> Hop 2: "GDP of China"
        -> Hop 3: Compare results
        """
        hops = []

        # Extract entities to compare
        # Simple heuristic: look for "and", "versus", "vs"
        entities_match = re.search(
            r'(\w+(?:\s+\w+)*?)\s+(?:and|versus|vs\.?)\s+(\w+(?:\s+\w+)*)',
            query,
            re.IGNORECASE
        )

        if entities_match:
            entity1 = entities_match.group(1)
            entity2 = entities_match.group(2)

            # Extract the aspect being compared
            aspect = query.split()[0]  # First word often describes what to compare

            # Hop 1: Get info about entity 1
            hops.append(RetrievalHop(
                hop_id=0,
                hop_type=HopType.INITIAL,
                query=f"{aspect} {entity1}",
                parent_hop_ids=[],
                reasoning=f"Retrieve information about {entity1}"
            ))

            # Hop 2: Get info about entity 2
            hops.append(RetrievalHop(
                hop_id=1,
                hop_type=HopType.INITIAL,
                query=f"{aspect} {entity2}",
                parent_hop_ids=[],
                reasoning=f"Retrieve information about {entity2}"
            ))

            # Hop 3: Comparison
            hops.append(RetrievalHop(
                hop_id=2,
                hop_type=HopType.COMPARISON,
                query=query,  # Original query for context
                parent_hop_ids=[0, 1],
                reasoning=f"Compare {entity1} and {entity2}"
            ))
        else:
            # Fallback
            hops.append(RetrievalHop(
                hop_id=0,
                hop_type=HopType.COMPARISON,
                query=query,
                parent_hop_ids=[],
                reasoning="Comparison query (entities not extracted)"
            ))

        return hops

    def _decompose_aggregation(self, query: str) -> List[RetrievalHop]:
        """
        Decompose aggregation query

        Example: "Summarize all research about quantum computing in 2023"
        -> Hop 1: Broad retrieval with filters
        -> Hop 2: Aggregate and synthesize
        """
        hops = []

        # Hop 1: Initial broad retrieval
        hops.append(RetrievalHop(
            hop_id=0,
            hop_type=HopType.INITIAL,
            query=query,
            parent_hop_ids=[],
            reasoning="Initial broad retrieval for aggregation"
        ))

        # Hop 2: Aggregation
        hops.append(RetrievalHop(
            hop_id=1,
            hop_type=HopType.AGGREGATION,
            query=query,
            parent_hop_ids=[0],
            reasoning="Aggregate and synthesize results"
        ))

        return hops

    def _decompose_with_llm(self, query: str, max_hops: int) -> List[RetrievalHop]:
        """
        Use LLM to intelligently decompose query

        Args:
            query: Complex query
            max_hops: Maximum number of hops

        Returns:
            List of reasoning hops
        """
        prompt = f"""Decompose this complex query into a sequence of simpler sub-queries.

Original Query: "{query}"

Instructions:
1. Identify if this query requires multiple steps to answer
2. Break it down into 2-{max_hops} sequential sub-queries
3. Each sub-query should build on previous ones
4. Format: Return a JSON list of objects with "query" and "reasoning" fields

Example:
Query: "What company did the founder of SpaceX start before SpaceX?"
Output:
[
  {{"query": "Who founded SpaceX?", "reasoning": "First identify the founder"}},
  {{"query": "What company did [founder] start before SpaceX?", "reasoning": "Then find their previous company"}}
]

Now decompose: "{query}"
Return only valid JSON, no explanation.
"""

        try:
            response = self.llm_generator.generate(prompt, max_tokens=300, temperature=0.0)

            # Parse response
            import json
            response_clean = response.replace("```json", "").replace("```", "").strip()
            sub_queries_data = json.loads(response_clean)

            hops = []
            for i, item in enumerate(sub_queries_data[:max_hops]):
                hop = RetrievalHop(
                    hop_id=i,
                    hop_type=HopType.BRIDGE if i > 0 else HopType.INITIAL,
                    query=item.get("query", ""),
                    parent_hop_ids=[i-1] if i > 0 else [],
                    reasoning=item.get("reasoning", "")
                )
                hops.append(hop)

            logger.info(f"LLM decomposed query into {len(hops)} hops")
            return hops

        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}. Falling back to single hop.")
            return [RetrievalHop(
                hop_id=0,
                hop_type=HopType.INITIAL,
                query=query,
                parent_hop_ids=[],
                reasoning="LLM decomposition failed, using original query"
            )]


class MultiHopRetriever:
    """
    Multi-hop retrieval system

    Executes multi-hop retrieval plan by:
    1. Processing hops in dependency order
    2. Using results from previous hops to inform next hops
    3. Extracting entities from results for query refinement
    4. Aggregating results across hops
    """

    def __init__(
        self,
        base_retriever,
        query_decomposer: Optional[QueryDecomposer] = None,
        entity_extractor=None,
    ):
        """
        Initialize multi-hop retriever

        Args:
            base_retriever: Base retriever for single-hop retrieval
            query_decomposer: Query decomposer (creates one if None)
            entity_extractor: Optional entity extractor for query refinement
        """
        self.base_retriever = base_retriever
        self.query_decomposer = query_decomposer or QueryDecomposer()
        self.entity_extractor = entity_extractor

        logger.info("Initialized MultiHopRetriever")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_hops: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Tuple[Chunk, float]], MultiHopPlan]:
        """
        Multi-hop retrieval

        Args:
            query: Complex query
            top_k: Results per hop
            max_hops: Maximum reasoning hops
            metadata_filter: Optional metadata filters

        Returns:
            Tuple of (final_results, multi_hop_plan)
        """
        # 1. Decompose query
        plan = self.query_decomposer.decompose(query, max_hops=max_hops)

        logger.info(
            f"Executing multi-hop retrieval: {plan.reasoning_type} "
            f"with {len(plan.hops)} hops"
        )

        # 2. Execute hops in order
        for hop in plan.hops:
            self._execute_hop(hop, plan, top_k, metadata_filter)

        # 3. Aggregate results
        final_results = self._aggregate_results(plan, top_k)

        logger.info(
            f"Multi-hop retrieval complete: {len(final_results)} final results "
            f"from {len(plan.hops)} hops"
        )

        return final_results, plan

    def _execute_hop(
        self,
        hop: RetrievalHop,
        plan: MultiHopPlan,
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]],
    ):
        """
        Execute a single retrieval hop

        Args:
            hop: Hop to execute
            plan: Full multi-hop plan (for accessing previous hops)
            top_k: Number of results
            metadata_filter: Optional filters
        """
        logger.debug(f"Executing hop {hop.hop_id}: {hop.query}")

        # Refine query based on previous hop results
        refined_query = self._refine_query(hop, plan)

        # Execute retrieval
        results = self.base_retriever.retrieve(
            query=refined_query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # Store results in hop
        hop.results = results

        # Extract entities if available
        if self.entity_extractor and results:
            for chunk, _ in results[:3]:  # Top 3 chunks
                entities = self.entity_extractor.extract(chunk.content)
                hop.entities_found.update(entities)

        hop.metadata["refined_query"] = refined_query
        hop.metadata["result_count"] = len(results)

        logger.debug(
            f"Hop {hop.hop_id} complete: {len(results)} results, "
            f"{len(hop.entities_found)} entities"
        )

    def _refine_query(self, hop: RetrievalHop, plan: MultiHopPlan) -> str:
        """
        Refine hop query based on previous hop results

        Args:
            hop: Current hop
            plan: Full plan with previous hops

        Returns:
            Refined query
        """
        if not hop.parent_hop_ids:
            return hop.query

        # Get parent hop results
        parent_entities = set()
        for parent_id in hop.parent_hop_ids:
            if parent_id < len(plan.hops):
                parent_entities.update(plan.hops[parent_id].entities_found)

        # If query has placeholders like [founder], replace them
        refined_query = hop.query

        # Replace entity placeholders
        placeholders = re.findall(r'\[(\w+)\]', refined_query)
        for placeholder in placeholders:
            # Try to find matching entity
            matching_entities = [
                e for e in parent_entities
                if placeholder.lower() in e.lower()
            ]
            if matching_entities:
                # Use first matching entity
                refined_query = refined_query.replace(
                    f"[{placeholder}]",
                    matching_entities[0]
                )

        return refined_query

    def _aggregate_results(
        self,
        plan: MultiHopPlan,
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """
        Aggregate results from all hops

        Args:
            plan: Completed multi-hop plan
            top_k: Final number of results

        Returns:
            Aggregated and ranked results
        """
        all_results = {}  # chunk_id -> (chunk, max_score)

        # Collect results from all hops
        for hop in plan.hops:
            for chunk, score in hop.results:
                chunk_id = chunk.chunk_id

                if chunk_id in all_results:
                    # Keep maximum score
                    existing_score = all_results[chunk_id][1]
                    all_results[chunk_id] = (chunk, max(score, existing_score))
                else:
                    all_results[chunk_id] = (chunk, score)

        # Sort by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:top_k]

    def explain_reasoning(self, plan: MultiHopPlan) -> str:
        """
        Generate explanation of multi-hop reasoning

        Args:
            plan: Executed multi-hop plan

        Returns:
            Explanation string
        """
        explanation = "=== Multi-Hop Retrieval Report ===\n"
        explanation += f"Original Query: \"{plan.original_query}\"\n"
        explanation += f"Reasoning Type: {plan.reasoning_type}\n"
        explanation += f"Number of Hops: {len(plan.hops)}\n"
        explanation += f"Confidence: {plan.confidence:.2f}\n\n"

        for hop in plan.hops:
            explanation += f"--- Hop {hop.hop_id} ({hop.hop_type.value}) ---\n"
            explanation += f"Query: \"{hop.query}\"\n"
            explanation += f"Reasoning: {hop.reasoning}\n"

            if hop.parent_hop_ids:
                explanation += f"Depends on: Hop {', '.join(map(str, hop.parent_hop_ids))}\n"

            if "refined_query" in hop.metadata:
                refined = hop.metadata["refined_query"]
                if refined != hop.query:
                    explanation += f"Refined Query: \"{refined}\"\n"

            explanation += f"Results: {hop.metadata.get('result_count', 0)}\n"

            if hop.entities_found:
                entities_str = ", ".join(list(hop.entities_found)[:5])
                explanation += f"Entities Found: {entities_str}\n"

            explanation += "\n"

        explanation += "=" * 35

        return explanation
