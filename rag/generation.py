"""
LLM Generation Module
Handles interaction with local LLMs (Ollama, vLLM, LM Studio)
"""

from typing import List, Tuple, Optional, AsyncIterator, Dict, Any
import httpx
from loguru import logger

from .chunking import Chunk
from config.settings import settings
from .llm_provider import create_llm_provider, BaseLLMProvider


class LLMGenerator:
    """
    LLM generator for RAG responses
    Compatible with OpenAI API format (Ollama, vLLM, LM Studio)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize LLM generator

        Args:
            base_url: Base URL for LLM API
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Create LLM provider using factory
        self.provider: BaseLLMProvider = create_llm_provider(
            base_url=base_url, model=model, timeout=timeout
        )

        # Detect API type for backwards compatibility
        if "11434" in base_url or "ollama" in base_url.lower():
            self.api_type = "ollama"
        else:
            self.api_type = "openai"

        logger.info(
            f"Initialized LLM generator: {model} ({self.api_type}) with new provider"
        )

    def _build_rag_prompt(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Build RAG prompt from query and retrieved chunks

        Args:
            query: User query
            chunks_with_scores: Retrieved chunks with scores
            system_prompt: Optional system prompt override
            conversation_history: Previous conversation messages

        Returns:
            Formatted prompt
        """
        # Default system prompt
        if not system_prompt:
            system_prompt = """You are a RAG AI assistant. Your ONLY job is to answer questions using EXCLUSIVELY the information provided in the context documents below.

CRITICAL RULES - FOLLOW THESE STRICTLY:
1. ONLY use information that appears in the provided documents
2. NEVER add information from your training data or general knowledge
3. If the answer is not in the documents, you MUST say "I cannot find this information in the provided documents"
4. Quote or paraphrase DIRECTLY from the documents - do not rephrase with external knowledge
5. If you're uncertain, say so rather than guessing

DOCUMENT FORMATS:
- Many documents contain MARKDOWN TABLES with pipe characters (|)
- Tables format: | Column1 | Column2 | with separator row |---------|---------|
- Read tables row by row - each row contains related data
- Lists, paragraphs, and other formats also contain important information

HOW TO ANSWER:
✓ DO: Use exact information from documents
✓ DO: Quote directly when possible
✓ DO: Say "not in documents" if information is missing
✗ DON'T: Add your own knowledge
✗ DON'T: Make assumptions beyond the documents
✗ DON'T: Mention "Document 1, Document 2" in your response
✗ DON'T: Say "based on the documents" - just answer directly

Example:
Documents show: "| Hello | Hai |"
Question: "How to say hello?"
CORRECT: "Hello is 'Hai'"
WRONG: "Hello is usually translated as 'Hai', and in many contexts..." (adds external knowledge)"""

        # Format context from chunks
        context_parts = []
        for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
            context_parts.append(f"[Document {idx}]")

            # Add metadata if available
            metadata = chunk.metadata
            if metadata.get("filename"):
                context_parts.append(f"Source: {metadata['filename']}")
            if metadata.get("title"):
                context_parts.append(f"Title: {metadata['title']}")
            if metadata.get("page_range"):
                context_parts.append(f"Pages: {metadata['page_range']}")

            context_parts.append(f"Relevance: {score:.1%}")
            context_parts.append(f"\n{chunk.content}\n")

        context = "\n".join(context_parts)

        # Build conversation history section
        conversation_section = ""
        if conversation_history:
            conv_parts = []
            for msg in conversation_history[-4:]:  # Include last 4 messages for context
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                conv_parts.append(f"{role}: {msg['content']}")
            conversation_section = f"""
PREVIOUS CONVERSATION:

{chr(10).join(conv_parts)}

"""

        # Build full prompt
        prompt = f"""{system_prompt}

CONTEXT DOCUMENTS:

{context}
{conversation_section}
USER QUESTION: {query}

ANSWER THE QUESTION USING ONLY THE INFORMATION IN THE CONTEXT DOCUMENTS ABOVE. DO NOT ADD EXTERNAL KNOWLEDGE:"""

        logger.info(f"Generated RAG prompt ({len(prompt)} chars)")
        logger.debug(f"Full prompt: {prompt}")

        return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text from prompt (synchronous)

        Args:
            prompt: Input prompt
            max_tokens: Max tokens override
            temperature: Temperature override

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Use new provider architecture
        response = self.provider.generate_sync(prompt, max_tokens, temperature)

        logger.info("Received LLM response")
        logger.debug(f"Raw response: {response}")
        return response

    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Ollama API"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                return data.get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI-compatible API"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI API generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text from prompt with streaming (async)

        Args:
            prompt: Input prompt
            max_tokens: Max tokens override
            temperature: Temperature override

        Yields:
            Generated text chunks
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Use new provider architecture
        async for chunk in self.provider.generate_stream(
            prompt, max_tokens, temperature
        ):
            yield chunk
        else:
            async for chunk in self._generate_openai_stream(
                prompt, max_tokens, temperature
            ):
                yield chunk

    async def _generate_ollama_stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        """Generate using Ollama API with streaming"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json

                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    async def _generate_openai_stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        """Generate using OpenAI-compatible API with streaming"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix

                            if line == "[DONE]":
                                break

                            try:
                                import json

                                data = json.loads(line)
                                if data["choices"][0]["delta"].get("content"):
                                    yield data["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise

    def generate_rag_response(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate RAG response from query and chunks (synchronous)

        Args:
            query: User query
            chunks_with_scores: Retrieved chunks with scores
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        prompt = self._build_rag_prompt(query, chunks_with_scores, system_prompt)
        return self.generate(prompt)

    async def generate_rag_response_stream(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate RAG response with streaming (async)

        Args:
            query: User query
            chunks_with_scores: Retrieved chunks with scores
            system_prompt: Optional system prompt
            conversation_history: Previous conversation messages for context

        Yields:
            Response chunks
        """
        # Build RAG prompt with conversation history
        prompt = self._build_rag_prompt(
            query, chunks_with_scores, system_prompt, conversation_history
        )

        async for chunk in self.generate_stream(prompt):
            yield chunk

    def contextualize_query(self, query: str, conversation_history: List[dict]) -> str:
        """
        Rewrite query to be self-contained based on conversation history
        """
        if not conversation_history:
            return query

        # Create prompt for contextualization with few-shot examples
        history_text = "\n".join(
            [
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history[-4:]
            ]
        )

        prompt = f"""Rephrase the last user question to be a standalone question that can be understood without the chat history.
If the question is already standalone, return it exactly as is.
Do NOT answer the question.

Example 1:
Chat History:
USER: Who is the CEO of Tesla?
ASSISTANT: Elon Musk is the CEO.
User Question: How old is he?
Standalone Question: How old is Elon Musk?

Example 2:
Chat History:
USER: Tell me about Paris.
ASSISTANT: Paris is the capital of France.
User Question: What is the population?
Standalone Question: What is the population of Paris?

Current Conversation:
{history_text}
User Question: {query}

Standalone Question:"""

        try:
            # Generate standalone question
            logger.info(f"Contextualizing query: {query}")
            response = self.generate(prompt, max_tokens=100, temperature=0.1)

            logger.debug(f"Raw contextualization response: '{response}'")

            cleaned_response = response.strip().strip('"').strip("'")

            # If empty, return original query
            if not cleaned_response:
                logger.warning(
                    f"Contextualization returned empty string for '{query}', using original."
                )
                return query

            logger.info(f"Contextualized query: '{query}' -> '{cleaned_response}'")
            return cleaned_response
        except Exception as e:
            logger.warning(f"Failed to contextualize query: {e}")
            return query
