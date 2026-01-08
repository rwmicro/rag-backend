"""
LLM Provider Module
Consolidated logic for both sync and streaming generation across different LLM providers
"""

from typing import Optional, AsyncIterator
import httpx
import json
from loguru import logger


class BaseLLMProvider:
    """Base class for LLM providers"""

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Synchronous generation - to be implemented by subclasses"""
        raise NotImplementedError

    async def generate_stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        """Streaming generation - to be implemented by subclasses"""
        raise NotImplementedError
        yield  # Make it a generator


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider with unified sync/stream logic"""

    def _build_payload(
        self, prompt: str, max_tokens: int, temperature: float, stream: bool
    ) -> dict:
        """Build request payload for Ollama API"""
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    def generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Synchronous generation with Ollama API"""
        url = f"{self.base_url}/api/generate"
        payload = self._build_payload(prompt, max_tokens, temperature, stream=False)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                return data.get("response", "")

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ollama HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Ollama request timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def generate_stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        """Streaming generation with Ollama API"""
        url = f"{self.base_url}/api/generate"
        payload = self._build_payload(prompt, max_tokens, temperature, stream=True)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping invalid JSON line: {line}")
                                continue

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama streaming HTTP error: {e.response.status_code}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Ollama streaming timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible provider (vLLM, LM Studio, etc.) with unified sync/stream logic"""

    def _build_payload(
        self, prompt: str, max_tokens: int, temperature: float, stream: bool
    ) -> dict:
        """Build request payload for OpenAI-compatible API"""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    def generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Synchronous generation with OpenAI-compatible API"""
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(prompt, max_tokens, temperature, stream=False)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            logger.error(
                f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(f"OpenAI API request timeout: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected OpenAI API response format: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API generation failed: {e}")
            raise

    async def generate_stream(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        """Streaming generation with OpenAI-compatible API"""
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(prompt, max_tokens, temperature, stream=True)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix

                            if line.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(line)
                                delta = data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping invalid JSON line: {line}")
                                continue
                            except (KeyError, IndexError) as e:
                                logger.debug(f"Skipping malformed chunk: {e}")
                                continue

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI streaming HTTP error: {e.response.status_code}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"OpenAI streaming timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise


def create_llm_provider(
    base_url: str, model: str, timeout: int = 120
) -> BaseLLMProvider:
    """
    Factory function to create appropriate LLM provider.

    Args:
        base_url: Base URL for LLM API
        model: Model name
        timeout: Request timeout in seconds

    Returns:
        Appropriate LLM provider instance
    """
    # Detect provider type from base_url
    if "11434" in base_url or "ollama" in base_url.lower():
        logger.info(f"Creating Ollama provider for {model}")
        return OllamaProvider(base_url, model, timeout)
    else:
        logger.info(f"Creating OpenAI-compatible provider for {model}")
        return OpenAICompatibleProvider(base_url, model, timeout)
