"""
HyDE - Hypothetical Document Embeddings
Generates hypothetical answer documents for better retrieval

Based on: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
Adapted for production use (2025)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from loguru import logger

from .chunking import Chunk
from .embeddings import EmbeddingModel
from .generation import LLMGenerator


class HyDE:
    """
    HyDE (Hypothetical Document Embeddings) Retriever

    Instead of searching with query embedding directly:
    1. Generate hypothetical answer document using LLM
    2. Embed the hypothetical document
    3. Search with hypothetical document embedding

    Why it works:
    - Query and documents live in different embedding spaces
    - Hypothetical document is closer to actual documents
    - Bridges the semantic gap between question and answer
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_generator: LLMGenerator,
        num_hypothetical_docs: int = 3,
        doc_length: int = 200,
    ):
        """
        Initialize HyDE

        Args:
            embedding_model: Model to generate embeddings
            llm_generator: LLM to generate hypothetical documents
            num_hypothetical_docs: Number of hypothetical docs to generate
            doc_length: Target length of hypothetical docs (tokens)
        """
        self.embedding_model = embedding_model
        self.llm_generator = llm_generator
        self.num_hypothetical_docs = num_hypothetical_docs
        self.doc_length = doc_length

        logger.info(f"HyDE initialized (generating {num_hypothetical_docs} hypothetical docs)")

    def generate_hypothetical_documents(self, query: str) -> List[str]:
        """
        Generate hypothetical answer documents

        Uses LLM to generate plausible answers to the query.
        These answers approximate what real documents might say.

        Args:
            query: User query

        Returns:
            List of hypothetical document texts
        """
        hypothetical_docs = []

        prompt_template = """Please write a detailed passage that would answer this question:

Question: {query}

Write a comprehensive passage (around {length} words) that directly answers this question.
Focus on factual content and include relevant details. Do not mention that this is hypothetical.

Passage:"""

        for i in range(self.num_hypothetical_docs):
            try:
                # Generate variation by adding context to prompt
                if i == 0:
                    variation = ""
                elif i == 1:
                    variation = "\n(Focus on technical details)"
                else:
                    variation = "\n(Focus on practical applications)"

                prompt = prompt_template.format(
                    query=query,
                    length=self.doc_length
                ) + variation

                # Generate hypothetical document
                hyp_doc = self.llm_generator.generate(
                    prompt=prompt,
                    max_tokens=self.doc_length * 2,  # Token limit
                    temperature=0.7 if i > 0 else 0.5  # Vary temperature for diversity
                )

                if hyp_doc and len(hyp_doc.strip()) > 50:
                    hypothetical_docs.append(hyp_doc.strip())
                    logger.debug(f"Generated hypothetical doc {i+1}: {len(hyp_doc)} chars")

            except Exception as e:
                logger.warning(f"Failed to generate hypothetical doc {i+1}: {e}")
                continue

        if not hypothetical_docs:
            logger.warning("No hypothetical documents generated, falling back to query")
            hypothetical_docs = [query]  # Fallback

        logger.info(f"Generated {len(hypothetical_docs)} hypothetical documents")
        return hypothetical_docs

    def retrieve(
        self,
        query: str,
        vector_store,
        top_k: int = 10,
        fusion_method: str = "average",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve using HyDE

        Args:
            query: User query
            vector_store: Vector store to search
            top_k: Number of results
            fusion_method: How to combine multi-doc results
                - "average": Average embeddings of hypothetical docs
                - "max": Take max score across all searches
                - "rrf": Reciprocal Rank Fusion
            metadata_filter: Optional metadata filters

        Returns:
            List of (chunk, score) tuples
        """
        logger.info(f"HyDE retrieval for query: {query[:100]}...")

        # Step 1: Generate hypothetical documents
        hyp_docs = self.generate_hypothetical_documents(query)

        if not hyp_docs:
            logger.error("No hypothetical documents generated")
            return []

        # Step 2: Get embeddings for hypothetical documents
        hyp_embeddings = self.embedding_model.encode(hyp_docs)

        # Step 3: Search with each hypothetical document
        if fusion_method == "average":
            # Average embeddings and search once
            avg_embedding = np.mean(hyp_embeddings, axis=0)
            # Normalize
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

            results = vector_store.search(avg_embedding, top_k, metadata_filter=metadata_filter)

            logger.info(f"HyDE (average fusion): Retrieved {len(results)} chunks")
            return results

        elif fusion_method == "max":
            # Search with each, take max scores
            all_results = {}

            for i, hyp_emb in enumerate(hyp_embeddings):
                results = vector_store.search(hyp_emb, top_k * 2, metadata_filter=metadata_filter)

                for chunk, score in results:
                    if chunk.chunk_id not in all_results:
                        all_results[chunk.chunk_id] = (chunk, score)
                    else:
                        # Take max score
                        existing_score = all_results[chunk.chunk_id][1]
                        if score > existing_score:
                            all_results[chunk.chunk_id] = (chunk, score)

            # Sort and return top-k
            sorted_results = sorted(
                all_results.values(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            logger.info(f"HyDE (max fusion): Retrieved {len(sorted_results)} chunks")
            return sorted_results

        elif fusion_method == "rrf":
            # Reciprocal Rank Fusion
            return self._rrf_fusion(hyp_embeddings, vector_store, top_k, metadata_filter=metadata_filter)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def _rrf_fusion(
        self,
        hyp_embeddings: np.ndarray,
        vector_store,
        top_k: int,
        k: int = 60,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion

        Combines multiple ranked lists using RRF score:
        score(chunk) = sum(1 / (k + rank_i))

        where rank_i is the rank in the i-th search result
        """
        rrf_scores = {}

        for i, hyp_emb in enumerate(hyp_embeddings):
            results = vector_store.search(hyp_emb, top_k * 2, metadata_filter=metadata_filter)

            for rank, (chunk, _) in enumerate(results):
                rrf_score = 1.0 / (k + rank + 1)

                if chunk.chunk_id not in rrf_scores:
                    rrf_scores[chunk.chunk_id] = {
                        'chunk': chunk,
                        'score': rrf_score
                    }
                else:
                    rrf_scores[chunk.chunk_id]['score'] += rrf_score

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        results = [(item['chunk'], item['score']) for item in sorted_results]

        logger.info(f"HyDE (RRF fusion): Retrieved {len(results)} chunks")
        return results

    def retrieve_hybrid(
        self,
        query: str,
        vector_store,
        top_k: int = 10,
        hyde_weight: float = 0.7
    ) -> List[Tuple[Chunk, float]]:
        """
        Hybrid retrieval: HyDE + direct query search

        Combines:
        - HyDE search (with hypothetical docs)
        - Direct query search

        Args:
            query: User query
            vector_store: Vector store
            top_k: Number of results
            hyde_weight: Weight for HyDE vs direct (0-1)

        Returns:
            Fused results
        """
        logger.info(f"Hybrid HyDE+Direct retrieval")

        # HyDE search
        hyde_results = self.retrieve(
            query=query,
            vector_store=vector_store,
            top_k=top_k * 2,
            fusion_method="average"
        )

        # Direct query search
        query_embedding = self.embedding_model.encode_single(query, is_query=True)
        direct_results = vector_store.search(query_embedding, top_k * 2)

        # Combine scores
        combined_scores = {}

        # Add HyDE scores
        for chunk, score in hyde_results:
            combined_scores[chunk.chunk_id] = {
                'chunk': chunk,
                'score': hyde_weight * score
            }

        # Add direct scores
        for chunk, score in direct_results:
            if chunk.chunk_id in combined_scores:
                combined_scores[chunk.chunk_id]['score'] += (1 - hyde_weight) * score
            else:
                combined_scores[chunk.chunk_id] = {
                    'chunk': chunk,
                    'score': (1 - hyde_weight) * score
                }

        # Sort and return
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        results = [(item['chunk'], item['score']) for item in sorted_results]

        logger.info(f"Hybrid HyDE: Retrieved {len(results)} chunks")
        return results


class AdaptiveHyDE(HyDE):
    """
    Adaptive HyDE that adjusts based on query complexity

    - Simple queries: Direct search (no HyDE)
    - Medium queries: HyDE with 1-2 hypothetical docs
    - Complex queries: HyDE with 3+ hypothetical docs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_threshold_low = 5  # words
        self.complexity_threshold_high = 15  # words

    def _assess_query_complexity(self, query: str) -> str:
        """
        Assess query complexity

        Returns: "simple", "medium", or "complex"
        """
        words = query.split()
        num_words = len(words)

        # Check for question words
        question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which'}
        has_question_word = any(w.lower() in question_words for w in words)

        if num_words < self.complexity_threshold_low:
            return "simple"
        elif num_words < self.complexity_threshold_high and has_question_word:
            return "medium"
        else:
            return "complex"

    def retrieve(
        self,
        query: str,
        vector_store,
        top_k: int = 10,
        fusion_method: str = "average",
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Adaptive retrieval based on query complexity"""

        complexity = self._assess_query_complexity(query)
        logger.info(f"Query complexity: {complexity}")

        if complexity == "simple":
            # Direct search, no HyDE
            logger.info("Using direct search (simple query)")
            query_embedding = self.embedding_model.encode_single(query, is_query=True)
            return vector_store.search(query_embedding, top_k, metadata_filter=metadata_filter)

        elif complexity == "medium":
            # HyDE with 1 hypothetical doc
            old_num = self.num_hypothetical_docs
            self.num_hypothetical_docs = 1
            results = super().retrieve(query, vector_store, top_k, fusion_method, metadata_filter=metadata_filter)
            self.num_hypothetical_docs = old_num
            return results

        else:
            # Full HyDE
            return super().retrieve(query, vector_store, top_k, fusion_method, metadata_filter=metadata_filter)


class MultilingualHyDE(HyDE):
    """
    Multilingual HyDE - Hypothetical Document Embeddings for Multiple Languages

    Generates hypothetical documents in the query's detected language.
    Supports 12+ languages with language-specific prompts.

    Languages:
    - English, French, German, Spanish, Italian, Portuguese
    - Dutch, Russian, Chinese, Japanese, Korean, Arabic

    Usage:
        hyde = MultilingualHyDE(embedding_model, llm_generator)
        results = hyde.retrieve("Qu'est-ce que le machine learning?", vector_store)
        # Generates French hypothetical document
    """

    # Multilingual prompt templates
    PROMPTS = {
        "en": """Please write a detailed passage that would answer this question:

Question: {query}

Write a comprehensive passage (around {length} words) that directly answers this question.
Focus on factual content and include relevant details. Do not mention that this is hypothetical.

Passage:""",

        "fr": """Veuillez écrire un passage détaillé qui répondrait à cette question:

Question: {query}

Écrivez un passage complet (environ {length} mots) qui répond directement à cette question.
Concentrez-vous sur le contenu factuel et incluez des détails pertinents. Ne mentionnez pas que cela est hypothétique.

Passage:""",

        "de": """Bitte schreiben Sie einen detaillierten Abschnitt, der diese Frage beantworten würde:

Frage: {query}

Schreiben Sie einen umfassenden Abschnitt (etwa {length} Wörter), der diese Frage direkt beantwortet.
Konzentrieren Sie sich auf sachliche Inhalte und fügen Sie relevante Details hinzu. Erwähnen Sie nicht, dass dies hypothetisch ist.

Abschnitt:""",

        "es": """Por favor, escriba un pasaje detallado que respondería a esta pregunta:

Pregunta: {query}

Escriba un pasaje completo (alrededor de {length} palabras) que responda directamente a esta pregunta.
Céntrese en el contenido factual e incluya detalles relevantes. No mencione que esto es hipotético.

Pasaje:""",

        "it": """Per favore, scrivi un passaggio dettagliato che risponderebbe a questa domanda:

Domanda: {query}

Scrivi un passaggio completo (circa {length} parole) che risponda direttamente a questa domanda.
Concentrati sul contenuto fattuale e includi dettagli rilevanti. Non menzionare che questo è ipotetico.

Passaggio:""",

        "pt": """Por favor, escreva uma passagem detalhada que responderia a esta pergunta:

Pergunta: {query}

Escreva uma passagem abrangente (cerca de {length} palavras) que responda diretamente a esta pergunta.
Concentre-se no conteúdo factual e inclua detalhes relevantes. Não mencione que isto é hipotético.

Passagem:""",

        "nl": """Schrijf alstublieft een gedetailleerde passage die deze vraag zou beantwoorden:

Vraag: {query}

Schrijf een uitgebreide passage (ongeveer {length} woorden) die deze vraag direct beantwoordt.
Focus op feitelijke inhoud en voeg relevante details toe. Vermeld niet dat dit hypothetisch is.

Passage:""",

        "ru": """Пожалуйста, напишите подробный отрывок, который ответил бы на этот вопрос:

Вопрос: {query}

Напишите исчерпывающий отрывок (около {length} слов), который напрямую отвечает на этот вопрос.
Сосредоточьтесь на фактическом содержании и включите соответствующие детали. Не упоминайте, что это гипотетически.

Отрывок:""",

        "zh": """请撰写一段详细的文章来回答这个问题：

问题：{query}

写一篇全面的文章（约{length}字）直接回答这个问题。
专注于事实内容并包含相关细节。不要提及这是假设性的。

文章：""",

        "ja": """この質問に答える詳細な文章を書いてください：

質問：{query}

この質問に直接答える包括的な文章（約{length}語）を書いてください。
事実的な内容に焦点を当て、関連する詳細を含めてください。これが仮説的であることは言及しないでください。

文章：""",

        "ko": """이 질문에 답할 수 있는 상세한 문장을 작성해 주세요:

질문: {query}

이 질문에 직접 답하는 포괄적인 문장(약 {length}단어)을 작성하세요.
사실적인 내용에 집중하고 관련 세부 사항을 포함하세요. 이것이 가상이라고 언급하지 마세요.

문장:""",

        "ar": """من فضلك اكتب مقطعًا مفصلاً يجيب على هذا السؤال:

السؤال: {query}

اكتب مقطعًا شاملاً (حوالي {length} كلمة) يجيب مباشرة على هذا السؤال.
ركز على المحتوى الواقعي وقم بتضمين تفاصيل ذات صلة. لا تذكر أن هذا افتراضي.

المقطع:""",
    }

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_generator: LLMGenerator,
        num_hypothetical_docs: int = 3,
        doc_length: int = 200,
        default_language: str = "en"
    ):
        """
        Initialize multilingual HyDE

        Args:
            embedding_model: Embedding model (should be multilingual)
            llm_generator: LLM generator
            num_hypothetical_docs: Number of hypothetical docs to generate
            doc_length: Target length in words
            default_language: Default language if detection fails
        """
        super().__init__(
            embedding_model=embedding_model,
            llm_generator=llm_generator,
            num_hypothetical_docs=num_hypothetical_docs,
            doc_length=doc_length
        )

        self.default_language = default_language
        self.language_detector = None

        # Try to import language detector
        try:
            from .language_detection import get_language_detector
            self.language_detector = get_language_detector()
            logger.info("Language detector loaded for multilingual HyDE")
        except ImportError:
            logger.warning("Language detector not available, using default language")

        logger.info(f"MultilingualHyDE initialized (default_language={default_language})")

    def generate_hypothetical_documents(self, query: str, language: Optional[str] = None) -> List[str]:
        """
        Generate hypothetical documents in the query's language

        Args:
            query: User query (in any language)
            language: Optional language code (auto-detected if None)

        Returns:
            List of hypothetical documents in the same language as query
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

        logger.info(f"Generating hypothetical documents in language: {language}")

        # Get prompt template for language (fallback to English)
        prompt_template = self.PROMPTS.get(language, self.PROMPTS["en"])

        hypothetical_docs = []

        for i in range(self.num_hypothetical_docs):
            try:
                # Generate variation by adding context
                if i == 0:
                    variation = ""
                elif i == 1:
                    # Add variation hints in English (LLM understands)
                    variation = "\n(Focus on technical details)"
                else:
                    variation = "\n(Focus on practical applications)"

                prompt = prompt_template.format(
                    query=query,
                    length=self.doc_length
                ) + variation

                # Generate hypothetical document
                hyp_doc = self.llm_generator.generate(
                    prompt=prompt,
                    max_tokens=self.doc_length * 2,
                    temperature=0.7 if i > 0 else 0.5
                )

                if hyp_doc and len(hyp_doc.strip()) > 50:
                    hypothetical_docs.append(hyp_doc.strip())
                    logger.debug(f"Generated hypothetical doc {i+1} ({language}): {len(hyp_doc)} chars")

            except Exception as e:
                logger.warning(f"Failed to generate hypothetical doc {i+1}: {e}")
                continue

        if not hypothetical_docs:
            logger.warning("No hypothetical documents generated, falling back to query")
            hypothetical_docs = [query]

        logger.info(f"Generated {len(hypothetical_docs)} hypothetical documents in {language}")
        return hypothetical_docs

    def retrieve(
        self,
        query: str,
        vector_store,
        top_k: int = 10,
        fusion_method: str = "average",
        metadata_filter: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve using multilingual HyDE

        Args:
            query: User query (in any language)
            vector_store: Vector store to search
            top_k: Number of results
            fusion_method: How to combine results
            metadata_filter: Optional metadata filters
            language: Optional language code (auto-detected if None)

        Returns:
            List of (chunk, score) tuples
        """
        logger.info(f"Multilingual HyDE retrieval for query: {query[:100]}...")

        # Step 1: Generate hypothetical documents in query's language
        hyp_docs = self.generate_hypothetical_documents(query, language=language)

        if not hyp_docs:
            logger.error("No hypothetical documents generated")
            return []

        # Step 2: Get embeddings (using multilingual model)
        hyp_embeddings = [
            self.embedding_model.encode_single(doc, is_query=False)
            for doc in hyp_docs
        ]

        # Step 3: Search with each hypothetical document
        if fusion_method == "average":
            # Average embeddings and do single search
            avg_embedding = np.mean(hyp_embeddings, axis=0)
            results = vector_store.search(avg_embedding, top_k, metadata_filter=metadata_filter)

        elif fusion_method == "max":
            # Search with each, take max scores
            all_results = {}
            for emb in hyp_embeddings:
                search_results = vector_store.search(emb, top_k * 2, metadata_filter=metadata_filter)
                for chunk, score in search_results:
                    chunk_id = chunk.chunk_id
                    if chunk_id not in all_results or score > all_results[chunk_id][1]:
                        all_results[chunk_id] = (chunk, score)

            # Sort and take top_k
            results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:top_k]

        elif fusion_method == "rrf":
            # Reciprocal Rank Fusion
            from .fusion import reciprocal_rank_fusion
            all_result_lists = []
            for emb in hyp_embeddings:
                search_results = vector_store.search(emb, top_k * 2, metadata_filter=metadata_filter)
                all_result_lists.append(search_results)

            results = reciprocal_rank_fusion(all_result_lists, top_k=top_k)

        else:
            logger.warning(f"Unknown fusion method: {fusion_method}, using average")
            avg_embedding = np.mean(hyp_embeddings, axis=0)
            results = vector_store.search(avg_embedding, top_k, metadata_filter=metadata_filter)

        logger.info(f"Multilingual HyDE retrieved {len(results)} results")
        return results
