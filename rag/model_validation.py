"""
Model Validation Utility
Validates model availability and compatibility before use
"""

from typing import Optional, Tuple, List, Dict, Any
import requests
import subprocess
import os
from loguru import logger

from config.settings import settings
from .model_registry import get_model_info, get_model_dimension


class ModelValidator:
    """Validates model availability and compatibility"""

    def __init__(self, ollama_base_url: Optional[str] = None):
        """
        Initialize model validator

        Args:
            ollama_base_url: Ollama API base URL (uses settings if None)
        """
        self.ollama_base_url = ollama_base_url or settings.LLM_BASE_URL

    def validate_embedding_model(
        self,
        model_name: str,
        check_availability: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if embedding model is available

        Args:
            model_name: Model name or shortcut
            check_availability: Actually check if model is loaded/available

        Returns:
            (is_valid, error_message)
        """
        # Check if model is in our registry
        info = get_model_info(model_name, "embedding")

        if info:
            logger.debug(f"Model '{model_name}' found in registry: {info.name}")

            # If it's a HuggingFace model and we're checking availability
            if info.model_type == "huggingface" and check_availability:
                return self._validate_huggingface_model(info.name)

            # If it's an Ollama model
            elif info.model_type == "ollama" and check_availability:
                return self._validate_ollama_embedding(info.name)

            return True, None

        # Unknown model - check if it might be Ollama
        if ":" in model_name or "/" not in model_name:
            logger.debug(f"Unknown model '{model_name}', checking Ollama...")
            if check_availability:
                return self._validate_ollama_embedding(model_name)
            return True, None  # Assume valid if not checking

        # Assume it's a HuggingFace model
        if check_availability:
            return self._validate_huggingface_model(model_name)

        return True, None

    def _validate_huggingface_model(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if HuggingFace model exists (without loading it)

        Args:
            model_name: Full HuggingFace model name

        Returns:
            (is_valid, error_message)
        """
        try:
            # Try to fetch model info from HuggingFace Hub API
            response = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                timeout=5
            )

            if response.status_code == 200:
                logger.debug(f"HuggingFace model '{model_name}' exists")
                return True, None
            elif response.status_code == 404:
                return False, f"Model '{model_name}' not found on HuggingFace Hub"
            else:
                # API error, but don't fail - model might still work
                logger.warning(f"Could not verify HuggingFace model '{model_name}': {response.status_code}")
                return True, None

        except requests.RequestException as e:
            # Network error - don't fail, model might still be cached locally
            logger.warning(f"Could not verify HuggingFace model '{model_name}': {e}")
            return True, None

    def _validate_ollama_embedding(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if Ollama embedding model is available

        Args:
            model_name: Ollama model name

        Returns:
            (is_valid, error_message)
        """
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=settings.OLLAMA_HEALTH_CHECK_TIMEOUT
            )

            if response.status_code != 200:
                return False, f"Ollama API not responding (status: {response.status_code})"

            # Check if model is in the list
            data = response.json()
            models = data.get("models", [])

            # Check for exact match or partial match (without tag)
            model_base = model_name.split(":")[0] if ":" in model_name else model_name

            for model in models:
                model_full = model.get("name", "")
                if model_full == model_name or model_full.startswith(f"{model_base}:"):
                    logger.debug(f"Ollama model '{model_name}' is available")
                    return True, None

            # Model not found
            available_models = [m.get("name") for m in models]
            return False, (
                f"Ollama model '{model_name}' not found.\n"
                f"Available models: {', '.join(available_models)}\n"
                f"Install it with: ollama pull {model_name}"
            )

        except requests.RequestException as e:
            return False, f"Could not connect to Ollama API at {self.ollama_base_url}: {e}"

    def validate_ollama_model(
        self,
        model_name: str,
        base_url: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if Ollama LLM model is pulled and available

        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b")
            base_url: Optional Ollama base URL

        Returns:
            (is_valid, error_message)
        """
        url = base_url or self.ollama_base_url

        try:
            response = requests.get(
                f"{url}/api/tags",
                timeout=settings.OLLAMA_HEALTH_CHECK_TIMEOUT
            )

            if response.status_code != 200:
                return False, f"Ollama API not responding at {url}"

            data = response.json()
            models = data.get("models", [])

            # Check for model
            for model in models:
                if model.get("name") == model_name:
                    logger.debug(f"Ollama LLM model '{model_name}' is available")
                    return True, None

            available = [m.get("name") for m in models]
            return False, (
                f"LLM model '{model_name}' not found in Ollama.\n"
                f"Available: {', '.join(available)}\n"
                f"Install with: ollama pull {model_name}"
            )

        except requests.RequestException as e:
            return False, f"Could not connect to Ollama: {e}"

    def validate_reranker_model(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if reranker model is available

        Args:
            model_name: Reranker model name

        Returns:
            (is_valid, error_message)
        """
        info = get_model_info(model_name, "reranker")

        if info:
            logger.debug(f"Reranker '{model_name}' found in registry: {info.name}")

            # Most rerankers are HuggingFace
            if info.model_type == "huggingface":
                return self._validate_huggingface_model(info.name)

            return True, None

        # Unknown reranker - assume HuggingFace
        return self._validate_huggingface_model(model_name)

    def get_fallback_model(self, model_type: str) -> str:
        """
        Return default fallback for a model type

        Args:
            model_type: "embedding", "llm", or "reranker"

        Returns:
            Default model name
        """
        fallbacks = {
            "embedding": settings.EMBEDDING_MODEL,
            "llm": settings.LLM_MODEL,
            "reranker": settings.RERANKER_MODEL,
        }

        return fallbacks.get(model_type, settings.EMBEDDING_MODEL)

    def validate_and_fallback(
        self,
        model_name: str,
        model_type: str,
        check_availability: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Validate model and return fallback if invalid

        Args:
            model_name: Model to validate
            model_type: "embedding", "llm", or "reranker"
            check_availability: Check actual availability

        Returns:
            (model_to_use, warning_message)
        """
        if model_type == "embedding":
            is_valid, error = self.validate_embedding_model(model_name, check_availability)
        elif model_type == "llm":
            is_valid, error = self.validate_ollama_model(model_name)
        elif model_type == "reranker":
            is_valid, error = self.validate_reranker_model(model_name)
        else:
            return model_name, None

        if is_valid:
            return model_name, None

        # Fall back to default
        fallback = self.get_fallback_model(model_type)
        warning = (
            f"Model '{model_name}' validation failed: {error}\n"
            f"Falling back to default: {fallback}"
        )

        logger.warning(warning)
        return fallback, warning


# Global validator instance
_model_validator: Optional[ModelValidator] = None


def get_model_validator() -> ModelValidator:
    """Get or create the global model validator"""
    global _model_validator

    if _model_validator is None:
        _model_validator = ModelValidator()

    return _model_validator


def detect_ollama_embedding_dimension(model_name: str, ollama_base_url: Optional[str] = None) -> Optional[int]:
    """
    Detect embedding dimension by querying Ollama with a test prompt

    Args:
        model_name: Ollama model name
        ollama_base_url: Optional Ollama base URL

    Returns:
        Embedding dimension if successful, None otherwise
    """
    url = ollama_base_url or settings.LLM_BASE_URL

    try:
        # Use the /api/embeddings endpoint with a test prompt
        response = requests.post(
            f"{url}/api/embeddings",
            json={
                "model": model_name,
                "prompt": "test"
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data.get("embedding", [])
            if embedding:
                dimension = len(embedding)
                logger.info(f"Detected Ollama model '{model_name}' dimension: {dimension}")
                return dimension

        logger.warning(f"Could not detect dimension for Ollama model '{model_name}': {response.status_code}")
        return None

    except Exception as e:
        logger.error(f"Failed to detect Ollama embedding dimension for '{model_name}': {e}")
        return None


def ensure_model_available(
    model_name: str,
    model_type: str = "embedding",
    auto_download: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Ensure model is available, optionally auto-downloading if missing

    Args:
        model_name: Model name to check
        model_type: "embedding", "reranker", or "llm"
        auto_download: Whether to auto-download if missing

    Returns:
        (is_available, error_message)
    """
    validator = get_model_validator()

    # Check if model is available
    if model_type == "embedding":
        is_valid, error_msg = validator.validate_embedding_model(model_name, check_availability=True)
    elif model_type == "reranker":
        is_valid, error_msg = validator.validate_reranker_model(model_name)
    elif model_type == "llm":
        is_valid, error_msg = validator.validate_ollama_model(model_name)
    else:
        return False, f"Unknown model type: {model_type}"

    if is_valid:
        return True, None

    # Model not available
    if not auto_download:
        return False, error_msg

    # Attempt auto-download
    logger.info(f"Attempting to auto-download {model_type} model: {model_name}")

    # Check if it's an Ollama model
    info = get_model_info(model_name, model_type)
    if info and info.model_type == "ollama":
        success, download_error = _pull_ollama_model(model_name)
        if success:
            logger.info(f"✓ Successfully downloaded Ollama model: {model_name}")
            return True, None
        return False, f"Failed to download Ollama model: {download_error}"

    # Check if it's a HuggingFace model
    elif model_type in ["embedding", "reranker"]:
        success, download_error = _download_huggingface_model(model_name, model_type)
        if success:
            logger.info(f"✓ Successfully downloaded HuggingFace model: {model_name}")
            return True, None
        return False, f"Failed to download HuggingFace model: {download_error}"

    return False, f"Auto-download not supported for this model type: {model_type}"


def _pull_ollama_model(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Pull an Ollama model using the ollama CLI

    Args:
        model_name: Ollama model name

    Returns:
        (success, error_message)
    """
    try:
        logger.info(f"Pulling Ollama model: {model_name}")

        # Use subprocess to call ollama pull
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"Ollama pull successful: {model_name}")
            return True, None
        else:
            error_msg = result.stderr or result.stdout
            logger.error(f"Ollama pull failed: {error_msg}")
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "Ollama pull timed out after 10 minutes"
    except FileNotFoundError:
        return False, "Ollama CLI not found. Please install Ollama."
    except Exception as e:
        return False, str(e)


def _download_huggingface_model(model_name: str, model_type: str) -> Tuple[bool, Optional[str]]:
    """
    Download a HuggingFace model (lazy download on first use)

    Args:
        model_name: HuggingFace model name
        model_type: "embedding" or "reranker"

    Returns:
        (success, error_message)
    """
    try:
        logger.info(f"Downloading HuggingFace model: {model_name}")

        if model_type == "embedding":
            from sentence_transformers import SentenceTransformer
            # This will download the model if not cached
            model = SentenceTransformer(model_name)
            logger.info(f"✓ Downloaded embedding model: {model_name}")
            return True, None

        elif model_type == "reranker":
            from sentence_transformers import CrossEncoder
            # This will download the model if not cached
            model = CrossEncoder(model_name)
            logger.info(f"✓ Downloaded reranker model: {model_name}")
            return True, None

        else:
            return False, f"Unsupported model type for HuggingFace: {model_type}"

    except Exception as e:
        logger.error(f"Failed to download HuggingFace model '{model_name}': {e}")
        return False, str(e)


def preload_models(model_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Preload models on startup to warm up caches

    Args:
        model_names: List of model names to preload (uses settings if None)

    Returns:
        Dictionary mapping model names to success status
    """
    if model_names is None:
        model_names = settings.PRELOAD_MODELS

    if not model_names:
        logger.info("No models configured for preloading")
        return {}

    logger.info(f"Preloading {len(model_names)} models...")
    results = {}

    for model_name in model_names:
        logger.info(f"Preloading model: {model_name}")

        try:
            # Determine model type based on registry
            info = get_model_info(model_name, "embedding")
            if not info:
                info = get_model_info(model_name, "reranker")

            if not info:
                logger.warning(f"Unknown model '{model_name}', skipping preload")
                results[model_name] = False
                continue

            # Preload based on model type
            if info.model_type == "huggingface":
                # Import and initialize to trigger download/cache
                from sentence_transformers import SentenceTransformer, CrossEncoder

                # Try as embedding model first
                try:
                    model = SentenceTransformer(model_name)
                    logger.info(f"✓ Preloaded embedding model: {model_name}")
                    results[model_name] = True
                except:
                    # Try as reranker
                    try:
                        model = CrossEncoder(model_name)
                        logger.info(f"✓ Preloaded reranker model: {model_name}")
                        results[model_name] = True
                    except Exception as e:
                        logger.error(f"Failed to preload '{model_name}': {e}")
                        results[model_name] = False

            elif info.model_type == "ollama":
                # Just validate that Ollama has the model
                validator = get_model_validator()
                is_valid, error = validator.validate_ollama_model(model_name)
                if is_valid:
                    logger.info(f"✓ Verified Ollama model: {model_name}")
                    results[model_name] = True
                else:
                    logger.warning(f"Ollama model not available: {model_name} - {error}")
                    results[model_name] = False
            else:
                logger.warning(f"Unknown model type for '{model_name}'")
                results[model_name] = False

        except Exception as e:
            logger.error(f"Failed to preload '{model_name}': {e}")
            results[model_name] = False

    successful = sum(1 for v in results.values() if v)
    logger.info(f"Preloading complete: {successful}/{len(model_names)} models loaded successfully")

    return results
