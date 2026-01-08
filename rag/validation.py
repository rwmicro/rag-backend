"""
Input Validation Utilities
Validates user inputs for API endpoints to prevent resource exhaustion and security issues
"""

from typing import Optional
from fastapi import HTTPException, UploadFile
from loguru import logger


# Configuration constants
MAX_FILE_SIZE_MB = 50  # Maximum file upload size in megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_QUERY_LENGTH = 10000  # Maximum query text length
MAX_TOP_K = 500  # Maximum number of results to return
MIN_TOP_K = 1
MAX_CHUNK_SIZE = 10000  # Maximum chunk size in characters
MIN_CHUNK_SIZE = 10
MAX_CHUNK_OVERLAP = 5000
MAX_BATCH_SIZE = 1000  # Maximum batch size for embeddings

ALLOWED_FILE_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".doc",
    ".docx",
    ".html",
    ".json",
    ".csv",
    ".xml",
}


class ValidationError(HTTPException):
    """Custom validation error exception"""

    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)


def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file for size and type.

    Args:
        file: Uploaded file from FastAPI

    Raises:
        ValidationError: If file is invalid
    """
    # Check filename exists
    if not file.filename:
        raise ValidationError("No filename provided")

    # Check file extension
    file_ext = None
    if "." in file.filename:
        file_ext = "." + file.filename.rsplit(".", 1)[-1].lower()

    if file_ext not in ALLOWED_FILE_EXTENSIONS:
        raise ValidationError(
            f"File type '{file_ext}' not allowed. "
            f"Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
        )

    # Check file size (if available from content-length header)
    if hasattr(file, "size") and file.size:
        if file.size > MAX_FILE_SIZE_BYTES:
            raise ValidationError(
                f"File size ({file.size / 1024 / 1024:.1f}MB) exceeds "
                f"maximum allowed size ({MAX_FILE_SIZE_MB}MB)"
            )

    logger.info(f"File upload validation passed: {file.filename}")


def validate_query_text(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Validate query text.

    Args:
        query: Query text
        max_length: Maximum allowed length

    Returns:
        Validated and stripped query text

    Raises:
        ValidationError: If query is invalid
    """
    if not query:
        raise ValidationError("Query cannot be empty")

    query = query.strip()

    if not query:
        raise ValidationError("Query cannot be empty or whitespace only")

    if len(query) > max_length:
        raise ValidationError(
            f"Query length ({len(query)}) exceeds maximum allowed length ({max_length})"
        )

    return query


def validate_top_k(top_k: Optional[int], default: int = 10) -> int:
    """
    Validate top_k parameter.

    Args:
        top_k: Number of results to return (None uses default)
        default: Default value if top_k is None

    Returns:
        Validated top_k value

    Raises:
        ValidationError: If top_k is invalid
    """
    if top_k is None:
        return default

    if not isinstance(top_k, int):
        raise ValidationError("top_k must be an integer")

    if top_k < MIN_TOP_K:
        raise ValidationError(f"top_k must be at least {MIN_TOP_K}")

    if top_k > MAX_TOP_K:
        raise ValidationError(
            f"top_k ({top_k}) exceeds maximum allowed value ({MAX_TOP_K})"
        )

    return top_k


def validate_chunk_params(
    chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
) -> tuple[int, int]:
    """
    Validate chunking parameters.

    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Tuple of (validated_chunk_size, validated_chunk_overlap)

    Raises:
        ValidationError: If parameters are invalid
    """
    # Default values
    validated_size = chunk_size if chunk_size is not None else 1000
    validated_overlap = chunk_overlap if chunk_overlap is not None else 200

    # Validate chunk_size
    if validated_size < MIN_CHUNK_SIZE:
        raise ValidationError(f"chunk_size must be at least {MIN_CHUNK_SIZE}")

    if validated_size > MAX_CHUNK_SIZE:
        raise ValidationError(
            f"chunk_size ({validated_size}) exceeds maximum ({MAX_CHUNK_SIZE})"
        )

    # Validate chunk_overlap
    if validated_overlap < 0:
        raise ValidationError("chunk_overlap cannot be negative")

    if validated_overlap > MAX_CHUNK_OVERLAP:
        raise ValidationError(
            f"chunk_overlap ({validated_overlap}) exceeds maximum ({MAX_CHUNK_OVERLAP})"
        )

    # Validate relationship between size and overlap
    if validated_overlap >= validated_size:
        raise ValidationError(
            f"chunk_overlap ({validated_overlap}) must be less than "
            f"chunk_size ({validated_size})"
        )

    return validated_size, validated_overlap


def validate_temperature(temperature: Optional[float], default: float = 0.7) -> float:
    """
    Validate LLM temperature parameter.

    Args:
        temperature: LLM temperature (None uses default)
        default: Default value if temperature is None

    Returns:
        Validated temperature value

    Raises:
        ValidationError: If temperature is invalid
    """
    if temperature is None:
        return default

    if not isinstance(temperature, (int, float)):
        raise ValidationError("temperature must be a number")

    if temperature < 0.0 or temperature > 2.0:
        raise ValidationError("temperature must be between 0.0 and 2.0")

    return float(temperature)


def validate_max_tokens(max_tokens: Optional[int], default: int = 2000) -> int:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Maximum tokens to generate (None uses default)
        default: Default value if max_tokens is None

    Returns:
        Validated max_tokens value

    Raises:
        ValidationError: If max_tokens is invalid
    """
    if max_tokens is None:
        return default

    if not isinstance(max_tokens, int):
        raise ValidationError("max_tokens must be an integer")

    if max_tokens < 1:
        raise ValidationError("max_tokens must be at least 1")

    if max_tokens > 32000:  # Reasonable upper limit
        raise ValidationError("max_tokens cannot exceed 32000")

    return max_tokens


def validate_collection_id(collection_id: Optional[str]) -> Optional[str]:
    """
    Validate collection ID for safety.

    Args:
        collection_id: Collection identifier

    Returns:
        Validated collection_id or None

    Raises:
        ValidationError: If collection_id contains invalid characters
    """
    if collection_id is None:
        return None

    collection_id = collection_id.strip()

    if not collection_id:
        return None

    # Only allow alphanumeric, hyphens, and underscores
    if not all(c.isalnum() or c in "-_" for c in collection_id):
        raise ValidationError(
            "collection_id can only contain alphanumeric characters, hyphens, and underscores"
        )

    if len(collection_id) > 100:
        raise ValidationError("collection_id cannot exceed 100 characters")

    return collection_id


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename (basename only, no path components)
    """
    import os

    # Get basename (removes any path components)
    filename = os.path.basename(filename)

    # Remove any remaining path separators
    filename = filename.replace("/", "_").replace("\\", "_")

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[: 255 - len(ext)] + ext

    return filename
