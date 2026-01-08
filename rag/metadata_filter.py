"""
Metadata Filtering Utility
Centralized metadata filtering logic used across vector stores
"""

from typing import Dict, Any
import re
from loguru import logger


class MetadataFilter:
    """
    Centralized metadata filtering logic for vector stores.

    Supports various operators:
    - Direct equality: {"filename": "doc.pdf"}
    - Comparison: {"page": {"$gte": 5, "$lte": 10}}
    - Membership: {"category": {"$in": ["tech", "science"]}}
    - Negation: {"status": {"$ne": "archived"}}
    - Regex: {"title": {"$regex": "^Chapter"}}
    - Substring: {"filename": {"$contains": "report"}}
    """

    @staticmethod
    def matches(metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter criteria.

        Args:
            metadata: Chunk metadata dictionary
            filter_dict: Filter criteria dictionary

        Returns:
            True if metadata matches all filter criteria, False otherwise

        Examples:
            >>> MetadataFilter.matches({"page": 5}, {"page": {"$gte": 3}})
            True

            >>> MetadataFilter.matches({"filename": "report.pdf"}, {"filename": "report"})
            True

            >>> MetadataFilter.matches({"tags": ["ai", "ml"]}, {"tags": {"$in": ["ai"]}})
            True
        """
        for key, value in filter_dict.items():
            if key not in metadata:
                logger.debug(f"Filter key '{key}' not found in metadata")
                return False

            # Handle operator-based filters
            if isinstance(value, dict):
                metadata_value = metadata[key]
                for op, op_value in value.items():
                    if not MetadataFilter._check_operator(metadata_value, op, op_value):
                        return False
            else:
                # Direct value comparison with smart matching
                if not MetadataFilter._direct_match(metadata[key], value, key):
                    return False

        return True

    @staticmethod
    def _check_operator(metadata_value: Any, op: str, op_value: Any) -> bool:
        """
        Check if metadata value matches operator condition.

        Args:
            metadata_value: Value from metadata
            op: Operator ($gte, $lte, $gt, $lt, $ne, $in, $regex, $contains)
            op_value: Operator value to compare against

        Returns:
            True if condition matches, False otherwise
        """
        try:
            if op == "$gte":
                return metadata_value >= op_value
            elif op == "$lte":
                return metadata_value <= op_value
            elif op == "$gt":
                return metadata_value > op_value
            elif op == "$lt":
                return metadata_value < op_value
            elif op == "$ne":
                return metadata_value != op_value
            elif op == "$in":
                # Handle both list membership and value in list
                if isinstance(op_value, list):
                    return metadata_value in op_value
                else:
                    logger.warning(f"$in operator expects list, got {type(op_value)}")
                    return False
            elif op == "$regex":
                # Regex matching (case-insensitive by default)
                pattern = re.compile(op_value, re.IGNORECASE)
                return bool(pattern.search(str(metadata_value)))
            elif op == "$contains":
                # Substring matching (case-insensitive)
                return op_value.lower() in str(metadata_value).lower()
            else:
                logger.warning(f"Unknown operator: {op}")
                return False
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Error comparing {metadata_value} with {op} {op_value}: {e}"
            )
            return False

    @staticmethod
    def _direct_match(metadata_value: Any, filter_value: Any, key: str) -> bool:
        """
        Direct value matching with smart comparison.

        For filename fields: Use substring matching (case-insensitive)
        For other fields: Use exact equality

        Args:
            metadata_value: Value from metadata
            filter_value: Value to match against
            key: Metadata key (used for special handling)

        Returns:
            True if values match, False otherwise
        """
        # Special handling for filename: substring matching
        if (
            key == "filename"
            and isinstance(filter_value, str)
            and isinstance(metadata_value, str)
        ):
            return filter_value.lower() in metadata_value.lower()

        # Default: exact equality
        return metadata_value == filter_value


def build_chroma_filter(filter_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert generic filter dict to ChromaDB where clause format.

    ChromaDB supports nested filters with $and, $or operators.
    This function translates simple filters to ChromaDB format.

    Args:
        filter_dict: Generic filter dictionary

    Returns:
        ChromaDB-compatible where clause

    Example:
        >>> build_chroma_filter({"page": 5, "source": "web"})
        {'page': {'$eq': 5}, 'source': {'$eq': 'web'}}
    """
    chroma_filter = {}

    for key, value in filter_dict.items():
        if isinstance(value, dict):
            # Already has operators, pass through
            chroma_filter[key] = value
        else:
            # Convert direct equality to $eq operator
            chroma_filter[key] = {"$eq": value}

    return chroma_filter
