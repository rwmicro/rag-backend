"""
Language Detection Module for Multilingual RAG

This module provides language detection capabilities using FastText's language
identification model. It supports 100+ languages and is optimized for both
short queries and longer document texts.

Usage:
    detector = LanguageDetector()
    lang, confidence = detector.detect("Bonjour le monde")
    # Returns: ("fr", 0.98)
"""

from typing import Tuple, Optional
import fasttext
from functools import lru_cache
import os
import logging

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects language of queries and documents.
    Uses FastText for accuracy on short texts.
    """

    def __init__(self, model_path: str = "lid.176.ftz"):
        """
        Initialize with FastText language identification model.

        Download model if not present:
        wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

        Args:
            model_path: Path to FastText language ID model
        """
        self.model_path = model_path
        self.model = None

        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(
                f"FastText language model not found at {model_path}. "
                f"Download it with:\n"
                f"wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            )
            logger.warning("Language detection will use fallback to 'en' until model is available")
        else:
            try:
                # Suppress FastText warnings
                fasttext.FastText.eprint = lambda x: None
                self.model = fasttext.load_model(model_path)
                logger.info(f"FastText language detection model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load FastText model: {e}")
                logger.warning("Language detection will use fallback to 'en'")

    @lru_cache(maxsize=1000)
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text with confidence score.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence_score)
            e.g., ("en", 0.95) for English with 95% confidence
        """
        if not text or not text.strip():
            return ("en", 0.0)

        if self.model is None:
            # Fallback if model not loaded
            return ("en", 0.5)

        try:
            # Clean text (FastText expects single line)
            clean_text = text.replace("\n", " ").strip()

            # Predict language
            predictions = self.model.predict(clean_text, k=1)

            # Extract language code and confidence
            lang_label = predictions[0][0]  # e.g., "__label__en"
            confidence = float(predictions[1][0])

            # Remove "__label__" prefix
            lang_code = lang_label.replace("__label__", "")

            return (lang_code, confidence)

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return ("en", 0.0)

    def detect_with_fallback(
        self,
        text: str,
        default: str = "en",
        min_confidence: float = 0.5
    ) -> str:
        """
        Detect language with fallback to default if confidence is low.

        Args:
            text: Text to analyze
            default: Default language code if detection fails
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Language code (falls back to default if confidence < threshold)
        """
        lang, confidence = self.detect(text)

        if confidence >= min_confidence:
            return lang
        else:
            logger.debug(
                f"Low confidence language detection ({confidence:.2f} < {min_confidence}), "
                f"using default: {default}"
            )
            return default

    def is_cjk(self, language: str) -> bool:
        """
        Check if language is CJK (Chinese, Japanese, Korean).

        Args:
            language: Language code (e.g., "zh", "ja", "ko")

        Returns:
            True if language is CJK
        """
        cjk_languages = {
            "zh",  # Chinese (all variants)
            "ja",  # Japanese
            "ko",  # Korean
            "zh-CN",  # Simplified Chinese
            "zh-TW",  # Traditional Chinese
            "zh-HK",  # Hong Kong Chinese
        }
        return language in cjk_languages

    def needs_special_tokenization(self, language: str) -> bool:
        """
        Check if language requires special tokenization.

        Languages without word boundaries (CJK, Thai, Lao, Burmese, Khmer)
        need specialized tokenizers instead of simple whitespace splitting.

        Args:
            language: Language code

        Returns:
            True if language needs special tokenization
        """
        special_tokenization_languages = {
            # CJK languages
            "zh", "zh-CN", "zh-TW", "zh-HK",  # Chinese
            "ja",  # Japanese
            "ko",  # Korean
            # Southeast Asian languages
            "th",  # Thai
            "lo",  # Lao
            "my",  # Burmese
            "km",  # Khmer (Cambodian)
            "bo",  # Tibetan
        }
        return language in special_tokenization_languages

    def get_language_family(self, language: str) -> str:
        """
        Get the language family for a given language code.

        Useful for selecting appropriate tokenization strategies.

        Args:
            language: Language code

        Returns:
            Language family name
        """
        language_families = {
            # Romance languages
            "es": "romance", "fr": "romance", "it": "romance",
            "pt": "romance", "ro": "romance", "ca": "romance",

            # Germanic languages
            "en": "germanic", "de": "germanic", "nl": "germanic",
            "sv": "germanic", "no": "germanic", "da": "germanic",

            # Slavic languages
            "ru": "slavic", "pl": "slavic", "cs": "slavic",
            "uk": "slavic", "bg": "slavic", "sr": "slavic",

            # CJK languages
            "zh": "sino-tibetan", "ja": "japonic", "ko": "koreanic",

            # Indic languages
            "hi": "indic", "bn": "indic", "te": "indic",
            "ta": "indic", "ur": "indic", "mr": "indic",

            # Semitic languages
            "ar": "semitic", "he": "semitic",

            # Turkic languages
            "tr": "turkic", "az": "turkic", "kk": "turkic",

            # Southeast Asian languages
            "th": "tai-kadai", "vi": "austroasiatic",
            "id": "austronesian", "ms": "austronesian",
        }

        return language_families.get(language, "other")


# Global instance (lazy initialization)
_global_detector = None


def get_language_detector(model_path: str = "lid.176.ftz") -> LanguageDetector:
    """
    Get global LanguageDetector instance (singleton pattern).

    Args:
        model_path: Path to FastText model

    Returns:
        LanguageDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = LanguageDetector(model_path=model_path)
    return _global_detector
