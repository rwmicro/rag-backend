"""
Query Classification and Routing System
Classifies queries and routes them to optimal retrieval strategies
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from loguru import logger


class MultilingualPatterns:
    """
    Multilingual query classification patterns

    Supports 6 major languages:
    - English
    - French
    - German
    - Spanish
    - Chinese
    - Japanese

    Each language has patterns for:
    - Factoid queries
    - Comparative queries
    - Procedural queries
    - Negation queries
    """

    @staticmethod
    def get_patterns(language: str) -> Dict[str, List[str]]:
        """
        Get query classification patterns for a specific language

        Args:
            language: ISO 639-1 language code

        Returns:
            Dictionary mapping query types to regex patterns
        """
        patterns_by_language = {
            "en": MultilingualPatterns._english_patterns(),
            "fr": MultilingualPatterns._french_patterns(),
            "de": MultilingualPatterns._german_patterns(),
            "es": MultilingualPatterns._spanish_patterns(),
            "zh": MultilingualPatterns._chinese_patterns(),
            "ja": MultilingualPatterns._japanese_patterns(),
        }

        return patterns_by_language.get(language, MultilingualPatterns._english_patterns())

    @staticmethod
    def _english_patterns() -> Dict[str, List[str]]:
        """English query patterns"""
        return {
            "factoid": [
                r'\b(what|who|when|where|which)\s+(is|are|was|were)\b',
                r'\bdefine\b',
            ],
            "comparative": [
                r'\b(compare|comparison|versus|vs\.?|differ|difference|between)\b',
                r'\b(better|worse|more|less)\s+than\b',
            ],
            "procedural": [
                r'\bhow\s+(to|do|can)\b',
                r'\b(steps?|process|guide|tutorial)\b',
            ],
            "negation": [
                r'\b(not|no|never|without|excluding?|except)\b',
                r"\bisn'?t\b",
            ],
        }

    @staticmethod
    def _french_patterns() -> Dict[str, List[str]]:
        """French query patterns"""
        return {
            "factoid": [
                r'\b(qu\'?est-ce que|quel|quelle|qui|quand|où|quoi)\b',
                r'\b(définir|définition)\b',
                r'\bc\'?est quoi\b',
            ],
            "comparative": [
                r'\b(comparer|comparaison|différence|entre)\b',
                r'\b(meilleur|pire|plus|moins)\s+que\b',
                r'\bpar rapport à\b',
            ],
            "procedural": [
                r'\bcomment\b',
                r'\b(étapes?|procédure|processus|guide|tutoriel)\b',
                r'\bfaire\b.*\bcomment\b',
            ],
            "negation": [
                r'\b(pas|non|jamais|sans|sauf|excepté)\b',
                r'\bn\'?est\s+pas\b',
                r'\baucun\b',
            ],
        }

    @staticmethod
    def _german_patterns() -> Dict[str, List[str]]:
        """German query patterns"""
        return {
            "factoid": [
                r'\b(was|wer|wann|wo|welche|welcher|welches)\s+(ist|sind|war|waren)\b',
                r'\bdefinier\w*\b',
                r'\bwas bedeutet\b',
            ],
            "comparative": [
                r'\b(vergleich|unterschied|zwischen)\b',
                r'\b(besser|schlechter|mehr|weniger)\s+als\b',
                r'\bgegenüber\b',
            ],
            "procedural": [
                r'\bwie\b',
                r'\b(schritte?|prozess|anleitung|tutorial)\b',
                r'\bmachen\b',
            ],
            "negation": [
                r'\b(nicht|kein|nie|ohne|ausgenommen)\b',
                r'\bist\s+nicht\b',
            ],
        }

    @staticmethod
    def _spanish_patterns() -> Dict[str, List[str]]:
        """Spanish query patterns"""
        return {
            "factoid": [
                r'\b(qué|quién|cuándo|dónde|cuál|cuáles)\s+(es|son|era|eran)\b',
                r'\bdefinir\b',
                r'\bqué significa\b',
            ],
            "comparative": [
                r'\b(comparar|comparación|diferencia|entre)\b',
                r'\b(mejor|peor|más|menos)\s+que\b',
                r'\bfrente a\b',
            ],
            "procedural": [
                r'\bcómo\b',
                r'\b(pasos?|proceso|guía|tutorial)\b',
                r'\bhacer\b',
            ],
            "negation": [
                r'\b(no|nunca|sin|excepto|salvo)\b',
                r'\bno\s+es\b',
                r'\bningún\b',
            ],
        }

    @staticmethod
    def _chinese_patterns() -> Dict[str, List[str]]:
        """Chinese query patterns"""
        return {
            "factoid": [
                r'什么是',
                r'谁是',
                r'什么时候',
                r'哪里',
                r'定义',
            ],
            "comparative": [
                r'比较',
                r'对比',
                r'区别',
                r'差异',
                r'和.*之间',
            ],
            "procedural": [
                r'如何',
                r'怎么',
                r'步骤',
                r'过程',
                r'教程',
            ],
            "negation": [
                r'不是',
                r'没有',
                r'除了',
                r'排除',
            ],
        }

    @staticmethod
    def _japanese_patterns() -> Dict[str, List[str]]:
        """Japanese query patterns"""
        return {
            "factoid": [
                r'とは',
                r'何ですか',
                r'誰ですか',
                r'いつ',
                r'どこ',
                r'定義',
            ],
            "comparative": [
                r'比較',
                r'違い',
                r'差異',
                r'と.*の間',
                r'より.*ほう',
            ],
            "procedural": [
                r'どのように',
                r'どうやって',
                r'方法',
                r'手順',
                r'チュートリアル',
            ],
            "negation": [
                r'ではない',
                r'ない',
                r'除く',
                r'以外',
            ],
        }


class QueryType(Enum):
    """Types of queries with different retrieval needs"""
    FACTOID = "factoid"                # "What is X?", "When did Y happen?"
    ANALYTICAL = "analytical"          # "Why does X cause Y?", "Explain the relationship"
    COMPARATIVE = "comparative"        # "Compare X and Y", "Differences between"
    PROCEDURAL = "procedural"          # "How to do X?", "Steps to accomplish"
    NAVIGATIONAL = "navigational"      # "Find section about", "Show me the part where"
    AGGREGATIVE = "aggregative"        # "Summarize all", "List all instances of"
    EXACT_MATCH = "exact_match"        # Contains IDs, codes, specific numbers
    TEMPORAL = "temporal"              # "Latest", "most recent", "in 2023"
    NEGATION = "negation"              # "What is NOT", "excluding", "except"
    MULTI_HOP = "multi_hop"            # Requires chaining multiple facts
    CONVERSATIONAL = "conversational"  # Follow-up, context-dependent


@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    query_type: QueryType
    confidence: float
    entities: List[str]
    temporal_markers: List[str]
    has_negation: bool
    requires_exact_match: bool
    suggested_strategy: str
    suggested_params: Dict[str, Any]
    reasoning: Optional[str] = None


class QueryClassifier:
    """
    Classifies queries and suggests optimal retrieval strategies.
    Uses rule-based pattern matching with optional LLM fallback.
    """

    def __init__(self, use_llm: bool = False, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self._load_patterns()

    def _load_patterns(self):
        """Load regex patterns for rule-based classification."""
        self.patterns = {
            QueryType.FACTOID: [
                r'^(what|who|when|where|which)\s+(is|are|was|were)\b',
                r'^define\b',
                r'^list\s+the\s+(definition|meaning)',
                r'\?$',  # Ends with question mark (factoid indicator)
            ],
            QueryType.ANALYTICAL: [
                r'\b(why|how\s+does|explain|analyze|analysis)\b',
                r'\b(reason|cause|effect|impact|consequence)\b',
                r'\b(relationship|correlation|connection)\b',
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|comparison|versus|vs\.?|differ|difference|between)\b',
                r'\b(better|worse|more|less)\s+than\b',
                r'\b(similar|different|contrast)\b',
                r'\bx\s+vs\s+y\b',
            ],
            QueryType.PROCEDURAL: [
                r'^how\s+(to|do|can|should|would)\b',
                r'\b(steps?|process|procedure|instructions?|guide)\b',
                r'\b(tutorial|walkthrough|implement)\b',
            ],
            QueryType.NAVIGATIONAL: [
                r'\b(find|show|locate|where.*section|which.*part)\b',
                r'\b(documentation|docs?)\s+(for|on|about)\b',
                r'\b(page|section|chapter)\s+\d+\b',
            ],
            QueryType.AGGREGATIVE: [
                r'\b(all|every|summarize|list|overview)\b',
                r'\b(instances?|occurrences?|mentions?)\s+of\b',
                r'\b(total|count|number)\s+of\b',
            ],
            QueryType.EXACT_MATCH: [
                r'\b[A-Z]{2,}-\d+\b',  # IDs like JIRA-123, REQ-456
                r'#\d+',               # Issue numbers
                r'\b\d{4,}\b',         # Long numbers (product codes, etc.)
                r'["\'"].+["\']',      # Quoted strings (exact match)
                r'\b[A-Z0-9]{8,}\b',   # Serial numbers, hashes
            ],
            QueryType.TEMPORAL: [
                r'\b(latest|recent|newest|current|today|yesterday|tomorrow)\b',
                r'\b(20\d{2}|19\d{2})\b',  # Years
                r'\b(q[1-4]|quarter\s+[1-4])\b',  # Quarters
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b(last|past|previous|next)\s+(year|month|week|day|quarter)\b',
            ],
            QueryType.NEGATION: [
                r'\b(not|no|never|without|excluding?|except)\b',
                r"\bisn'?t\b",
                r"\baren'?t\b",
                r"\bdon'?t\b",
                r"\bdoesn'?t\b",
                r'\bnone\s+of\b',
            ],
            QueryType.MULTI_HOP: [
                r'\b(and\s+then|after\s+that|followed\s+by)\b',
                r'\bof\s+the\s+\w+\s+that\b',  # "revenue of the company that acquired..."
                r'\b(which|who|what)\s+\w+.*\b(which|who|what)\b',  # Multiple wh-questions
            ],
            QueryType.CONVERSATIONAL: [
                r'^(it|this|that|those|these|they)\b',  # Anaphoric references
                r'^(yes|no|maybe|ok|okay)\b',
                r'^(what\s+about|how\s+about)\b',
                r'^(and|but|however|also)\b',  # Continuation markers
            ],
        }

        # Entity patterns (basic NER-like patterns)
        self.entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # John Doe
            'ORG': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Company)\b',
            'DATE': r'\b(?:20\d{2}|19\d{2})[-/]\d{1,2}[-/]\d{1,2}\b',
            'NUMBER': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
        }

    def classify(self, query: str, context: Optional[List[Dict]] = None) -> QueryAnalysis:
        """
        Classify query and return analysis with recommended strategy.

        Args:
            query: User query
            context: Optional conversation history for context-aware classification

        Returns:
            QueryAnalysis with query type and strategy recommendation
        """
        query_lower = query.lower()

        # Run rule-based pattern matching
        type_scores = self._score_query_types(query_lower)

        # Extract entities
        entities = self._extract_entities(query)

        # Detect temporal markers
        temporal_markers = self._extract_temporal_markers(query)

        # Check for negation
        has_negation = self._check_negation(query_lower)

        # Check for exact match indicators
        requires_exact_match = self._check_exact_match(query)

        # Determine query type
        if type_scores:
            # Get type with highest score
            query_type, confidence = max(type_scores.items(), key=lambda x: x[1])
        else:
            # Default to factoid if no pattern matches
            query_type = QueryType.FACTOID
            confidence = 0.5

        # If confidence is low and LLM is available, use LLM for classification
        if confidence < 0.7 and self.use_llm and self.llm_client:
            llm_result = self._classify_with_llm(query, context)
            if llm_result:
                query_type = llm_result["query_type"]
                confidence = llm_result["confidence"]

        # Get retrieval strategy for this query type
        strategy_config = self.get_retrieval_strategy(
            query_type,
            has_negation,
            requires_exact_match,
            temporal_markers,
            entities
        )

        analysis = QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            entities=entities,
            temporal_markers=temporal_markers,
            has_negation=has_negation,
            requires_exact_match=requires_exact_match,
            suggested_strategy=strategy_config["strategy"],
            suggested_params=strategy_config.get("params", {}),
            reasoning=f"Matched {query_type.value} pattern with {confidence:.2f} confidence"
        )

        logger.info(f"Query classified as {query_type.value} (confidence: {confidence:.2f})")
        logger.debug(f"Strategy: {strategy_config['strategy']}, Params: {strategy_config.get('params', {})}")

        return analysis

    def _score_query_types(self, query: str) -> Dict[QueryType, float]:
        """Score query against all patterns."""
        scores = {}

        for query_type, patterns in self.patterns.items():
            type_score = 0.0
            matches = 0

            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matches += 1
                    type_score += 1.0

            if matches > 0:
                # Normalize by number of patterns (more matches = higher confidence)
                confidence = min(1.0, type_score / len(patterns) + 0.5)
                scores[query_type] = confidence

        return scores

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query (simple pattern-based)."""
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            for match in re.finditer(pattern, query):
                entities.append(match.group(0))

        # Remove duplicates
        return list(set(entities))

    def _extract_temporal_markers(self, query: str) -> List[str]:
        """Extract temporal expressions."""
        temporal = []

        temporal_patterns = [
            r'\b(20\d{2}|19\d{2})\b',  # Years
            r'\b(q[1-4]|quarter\s+[1-4])\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(latest|recent|current|yesterday|tomorrow|today)\b',
        ]

        for pattern in temporal_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                temporal.append(match.group(0))

        return temporal

    def _check_negation(self, query: str) -> bool:
        """Check if query contains negation."""
        negation_words = [
            r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bwithout\b',
            r'\bexcluding?\b', r'\bexcept\b', r"\bisn'?t\b",
            r"\baren'?t\b", r"\bdon'?t\b", r"\bdoesn'?t\b"
        ]

        for pattern in negation_words:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def _check_exact_match(self, query: str) -> bool:
        """Check if query requires exact matching (IDs, codes, quoted strings)."""
        exact_match_patterns = [
            r'\b[A-Z]{2,}-\d+\b',  # IDs
            r'#\d+',                # Issue numbers
            r'["\'"].+["\']',      # Quoted strings
            r'\b[A-Z0-9]{8,}\b',   # Serial numbers
        ]

        for pattern in exact_match_patterns:
            if re.search(pattern, query):
                return True

        return False

    def _classify_with_llm(
        self,
        query: str,
        context: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for classification (fallback for ambiguous queries)."""
        if not self.llm_client:
            return None

        # Build context string
        context_str = ""
        if context:
            context_str = "Previous conversation:\n"
            for msg in context[-3:]:  # Last 3 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_str += f"{role}: {content}\n"

        prompt = f"""Classify this query into one of these categories:
- factoid: Simple fact-based questions (What is X? Who is Y?)
- analytical: Explanation/reasoning questions (Why? How does it work?)
- comparative: Comparison questions (X vs Y, differences)
- procedural: How-to questions (How to do X?)
- navigational: Finding specific sections/pages
- aggregative: Summarization/listing (all, every, total)
- exact_match: Queries with IDs, codes, or quoted exact strings
- temporal: Time-based queries (latest, 2023, recent)
- negation: Queries about what is NOT (excluding, except)
- multi_hop: Complex queries needing multiple facts
- conversational: Follow-up questions with pronouns (it, that, this)

{context_str}
Query: "{query}"

Respond with JSON:
{{
    "query_type": "category",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm_client.generate(prompt, temperature=0.1, max_tokens=200)
            import json
            result = json.loads(response)

            # Map string to enum
            type_str = result.get("query_type", "factoid")
            try:
                query_type = QueryType(type_str)
            except ValueError:
                query_type = QueryType.FACTOID

            return {
                "query_type": query_type,
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None

    def get_retrieval_strategy(
        self,
        query_type: QueryType,
        has_negation: bool = False,
        requires_exact_match: bool = False,
        temporal_markers: Optional[List[str]] = None,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Map query analysis to retrieval configuration.

        Returns:
            Dict with strategy name and parameters
        """
        strategy_map = {
            QueryType.FACTOID: {
                "strategy": "hybrid",
                "params": {
                    "alpha": 0.7,  # Balanced
                    "top_k": 20,
                    "use_reranking": True
                }
            },
            QueryType.ANALYTICAL: {
                "strategy": "vector",
                "params": {
                    "top_k": 30,  # More context for explanation
                    "use_reranking": True,
                    "use_mmr": True  # Diverse perspectives
                }
            },
            QueryType.COMPARATIVE: {
                "strategy": "multi_query",
                "params": {
                    "generate_subqueries": True,  # Split into query for X, query for Y
                    "ensure_coverage": entities if entities else [],
                    "top_k_per_query": 15,
                    "use_mmr": True  # Ensure diverse coverage
                }
            },
            QueryType.PROCEDURAL: {
                "strategy": "hybrid",
                "params": {
                    "alpha": 0.6,  # Slightly favor keywords (step-by-step)
                    "top_k": 25,
                    "sequential_ordering": True
                }
            },
            QueryType.NAVIGATIONAL: {
                "strategy": "metadata_filter",
                "params": {
                    "extract_filters": True,
                    "top_k": 10
                }
            },
            QueryType.AGGREGATIVE: {
                "strategy": "cluster_retrieval",
                "params": {
                    "cluster_results": True,
                    "top_k": 100,  # Get many results
                    "sample_per_cluster": 3,
                    "use_mmr": True
                }
            },
            QueryType.EXACT_MATCH: {
                "strategy": "hybrid",
                "params": {
                    "alpha": 0.3,  # Heavily favor BM25 for exact matching
                    "top_k": 30,
                    "fallback": "full_text_search"
                }
            },
            QueryType.TEMPORAL: {
                "strategy": "vector",
                "params": {
                    "top_k": 50,
                    "metadata_filters": {
                        "temporal": temporal_markers if temporal_markers else []
                    },
                    "sort_by_date": True
                }
            },
            QueryType.NEGATION: {
                "strategy": "contrastive",
                "params": {
                    "retrieve_positive": True,
                    "retrieve_negative": True,
                    "filter_negative_from_positive": True
                }
            },
            QueryType.MULTI_HOP: {
                "strategy": "iterative",
                "params": {
                    "max_hops": 3,
                    "use_graph": True,
                    "use_multi_query": True
                }
            },
            QueryType.CONVERSATIONAL: {
                "strategy": "hybrid",
                "params": {
                    "alpha": 0.8,
                    "top_k": 15,
                    "contextualize_query": True  # Rewrite with conversation context
                }
            }
        }

        base_config = strategy_map.get(query_type, {
            "strategy": "hybrid",
            "params": {"alpha": 0.8, "top_k": 20}
        })

        # Apply overrides based on special conditions
        if has_negation and query_type != QueryType.NEGATION:
            base_config["params"]["handle_negation"] = True

        if requires_exact_match and query_type != QueryType.EXACT_MATCH:
            base_config["params"]["boost_exact_match"] = True

        return base_config


class RetrievalRouter:
    """
    Routes queries to appropriate retrieval strategy based on classification.
    """

    def __init__(self, classifier: QueryClassifier):
        self.classifier = classifier

    def analyze_and_route(
        self,
        query: str,
        context: Optional[List[Dict]] = None
    ) -> Tuple[QueryAnalysis, Dict[str, Any]]:
        """
        Analyze query and return routing decision.

        Args:
            query: User query
            context: Conversation history

        Returns:
            Tuple of (QueryAnalysis, routing_config)
        """
        # Classify query
        analysis = self.classifier.classify(query, context)

        # Build routing config
        routing_config = {
            "strategy": analysis.suggested_strategy,
            "params": analysis.suggested_params,
            "query_type": analysis.query_type.value,
            "confidence": analysis.confidence,
            "metadata_filters": None,
            "pre_processing": [],
            "post_processing": []
        }

        # Add pre-processing steps
        if analysis.query_type == QueryType.CONVERSATIONAL:
            routing_config["pre_processing"].append("contextualize_query")

        if analysis.temporal_markers:
            # Extract metadata filters from temporal markers
            from metadata_store import FilterBuilder
            filters = FilterBuilder.from_query_analysis(query, {"temporal_markers": analysis.temporal_markers})
            routing_config["metadata_filters"] = filters

        # Add post-processing steps
        if analysis.query_type in [QueryType.AGGREGATIVE, QueryType.COMPARATIVE]:
            routing_config["post_processing"].append("ensure_diversity")

        if analysis.has_negation:
            routing_config["post_processing"].append("filter_negative_content")

        logger.info(f"Routed query to '{routing_config['strategy']}' strategy")

        return analysis, routing_config


class MultilingualQueryClassifier(QueryClassifier):
    """
    Multilingual Query Classifier

    Extends QueryClassifier with multilingual pattern support.
    Automatically detects query language and applies appropriate patterns.

    Supports:
    - English, French, German, Spanish, Chinese, Japanese
    - Auto language detection
    - Fallback to English patterns for unsupported languages

    Usage:
        classifier = MultilingualQueryClassifier()
        analysis = classifier.classify("Qu'est-ce que le machine learning?")
        # Detects French, applies French patterns
    """

    def __init__(self, use_llm: bool = False, llm_client=None, default_language: str = "en"):
        """
        Initialize multilingual query classifier

        Args:
            use_llm: Whether to use LLM for ambiguous cases
            llm_client: Optional LLM client
            default_language: Default language if detection fails
        """
        super().__init__(use_llm=use_llm, llm_client=llm_client)
        self.default_language = default_language
        self.language_detector = None

        # Try to import language detector
        try:
            from .language_detection import get_language_detector
            self.language_detector = get_language_detector()
            logger.info("Language detector loaded for multilingual query classification")
        except ImportError:
            logger.warning("Language detector not available, using default language")

    def classify(self, query: str, context: Optional[List[Dict]] = None, language: Optional[str] = None) -> QueryAnalysis:
        """
        Classify query with multilingual support

        Args:
            query: User query (in any supported language)
            context: Optional conversation history
            language: Optional language code (auto-detected if None)

        Returns:
            QueryAnalysis with query type and strategy
        """
        # Detect language if not provided
        if language is None:
            if self.language_detector:
                language = self.language_detector.detect_with_fallback(
                    query,
                    default=self.default_language,
                    min_confidence=0.5
                )
            else:
                language = self.default_language

        logger.debug(f"Classifying query in language: {language}")

        # Get patterns for detected language
        multilingual_patterns = MultilingualPatterns.get_patterns(language)

        # Temporarily replace patterns with language-specific ones
        original_patterns = self.patterns
        self._apply_multilingual_patterns(multilingual_patterns, language)

        # Call parent classify method
        try:
            analysis = super().classify(query, context)
            # Add language to analysis
            if analysis.suggested_params:
                analysis.suggested_params["query_language"] = language
            return analysis
        finally:
            # Restore original patterns
            self.patterns = original_patterns

    def _apply_multilingual_patterns(self, multilingual_patterns: Dict[str, List[str]], language: str):
        """
        Apply multilingual patterns to classifier

        Args:
            multilingual_patterns: Language-specific patterns
            language: Language code
        """
        # Map simple pattern keys to QueryType enum
        pattern_mapping = {
            "factoid": QueryType.FACTOID,
            "comparative": QueryType.COMPARATIVE,
            "procedural": QueryType.PROCEDURAL,
            "negation": QueryType.NEGATION,
        }

        # Update patterns for supported types
        for pattern_key, patterns in multilingual_patterns.items():
            query_type = pattern_mapping.get(pattern_key)
            if query_type:
                # Add multilingual patterns to existing patterns
                if query_type in self.patterns:
                    # Prepend multilingual patterns (higher priority)
                    self.patterns[query_type] = patterns + self.patterns[query_type]
                else:
                    self.patterns[query_type] = patterns

        logger.debug(f"Applied multilingual patterns for {language}")

