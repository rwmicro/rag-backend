"""RAG evaluation endpoint.

Couples to several helpers and globals from rag.main; those are looked up
lazily to keep router-load cheap and to avoid a circular import.
"""

from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import settings
from rag.evaluation import EvaluationSample, RAGEvaluator
from rag.schemas import EvaluationRequest, EvaluationResponse

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """Evaluate retrieval (and optionally generation) against a test dataset."""
    from rag import main as main_module
    from rag.retrieval import HybridRetriever, Retriever

    try:
        logger.info(
            f"Evaluation request received with {len(request.test_dataset)} samples"
        )

        samples = [
            EvaluationSample(
                query=item["query"],
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                expected_answer=item.get("expected_answer"),
                metadata=item.get("metadata", {}),
            )
            for item in request.test_dataset
        ]

        if request.collection_id:
            collection = main_module._load_collections().get(request.collection_id)
            embedding_dimension = (
                collection.get("embedding_dimension") if collection else None
            )
            vector_store = main_module._get_or_create_vector_store(
                request.collection_id, embedding_dimension
            )
        else:
            vector_store = main_module._get_or_create_vector_store(
                "default", settings.EMBEDDING_DIMENSION
            )

        collection_embedding_model_name = main_module._get_collection_embedding_model(
            request.collection_id
        )
        current_embedding_model = main_module.get_or_create_embedding_model(
            model_name=collection_embedding_model_name
        )

        if settings.USE_HYBRID_SEARCH:
            current_retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_model=current_embedding_model,
                alpha=settings.HYBRID_ALPHA,
            )
        else:
            current_retriever = Retriever(
                vector_store=vector_store,
                embedding_model=current_embedding_model,
            )

        llm_generator = getattr(main_module, "llm_generator", None)
        evaluator = RAGEvaluator(
            retriever=current_retriever,
            llm_generator=llm_generator if request.evaluate_generation else None,
        )

        retrieval_metrics = evaluator.evaluate_retrieval(
            samples=samples,
            k_values=request.k_values,
        )

        generation_metrics = None
        if request.evaluate_generation and llm_generator:
            logger.info("Evaluating generation quality...")
            generated_answers = []
            contexts = []
            for sample in samples:
                chunks_with_scores = current_retriever.retrieve(
                    sample.query, top_k=settings.FINAL_TOP_K
                )
                answer = llm_generator.generate_rag_response(
                    query=sample.query,
                    chunks_with_scores=chunks_with_scores,
                )
                generated_answers.append(answer)
                contexts.append([chunk for chunk, _ in chunks_with_scores])

            generation_metrics = evaluator.evaluate_generation(
                samples=samples,
                generated_answers=generated_answers,
                contexts=contexts,
            )

        return EvaluationResponse(
            retrieval_metrics=asdict(retrieval_metrics),
            generation_metrics=asdict(generation_metrics) if generation_metrics else None,
            sample_count=len(samples),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=str(e))
