"""
Metadata Store for Fast Filtering
Enables pre-filtering before vector search to reduce compute waste
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import json
import re
from loguru import logger


@dataclass
class MetadataFilter:
    """
    Defines a metadata filter to apply before vector search.
    """
    field: str
    operator: str  # "eq", "ne", "gt", "lt", "gte", "lte", "in", "contains", "range", "regex"
    value: Any

    def __repr__(self):
        return f"MetadataFilter({self.field} {self.operator} {self.value})"


class FilterBuilder:
    """
    Builds metadata filters from query analysis or explicit criteria.
    """

    @staticmethod
    def from_query_analysis(query: str, query_analysis: Optional[Dict] = None) -> List[MetadataFilter]:
        """
        Extract implicit filters from query.

        Examples:
        - "2023 annual report" -> date_year = 2023, doc_type = "annual_report"
        - "in the security section" -> section contains "security"
        - "from document X" -> filename = "X"
        """
        filters = []

        # Temporal extraction - Year
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', query)
        if year_match:
            filters.append(MetadataFilter(
                field="date_year",
                operator="eq",
                value=int(year_match.group())
            ))
            logger.debug(f"Extracted year filter: {year_match.group()}")

        # Quarter extraction
        quarter_match = re.search(r'\b(q[1-4]|Q[1-4]|quarter\s+[1-4])\b', query, re.IGNORECASE)
        if quarter_match:
            quarter_text = quarter_match.group().lower()
            if 'q1' in quarter_text or 'quarter 1' in quarter_text:
                quarter_num = 1
            elif 'q2' in quarter_text or 'quarter 2' in quarter_text:
                quarter_num = 2
            elif 'q3' in quarter_text or 'quarter 3' in quarter_text:
                quarter_num = 3
            elif 'q4' in quarter_text or 'quarter 4' in quarter_text:
                quarter_num = 4
            else:
                quarter_num = None

            if quarter_num:
                filters.append(MetadataFilter(
                    field="quarter",
                    operator="eq",
                    value=quarter_num
                ))
                logger.debug(f"Extracted quarter filter: Q{quarter_num}")

        # Document type extraction
        doc_types = {
            "report": r'\b(report|annual report|quarterly report)\b',
            "policy": r'\b(policy|policies|guideline)\b',
            "contract": r'\b(contract|agreement|terms)\b',
            "invoice": r'\b(invoice|bill|receipt)\b',
            "manual": r'\b(manual|guide|documentation)\b',
            "specification": r'\b(spec|specification|requirements?)\b',
        }

        for doc_type, pattern in doc_types.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters.append(MetadataFilter(
                    field="doc_type",
                    operator="contains",
                    value=doc_type
                ))
                logger.debug(f"Extracted doc_type filter: {doc_type}")
                break  # Only match one doc type

        # Section extraction
        section_match = re.search(r'\b(?:in|from)\s+(?:the\s+)?(\w+)\s+section\b', query, re.IGNORECASE)
        if section_match:
            section_name = section_match.group(1)
            filters.append(MetadataFilter(
                field="section",
                operator="contains",
                value=section_name
            ))
            logger.debug(f"Extracted section filter: {section_name}")

        # Filename extraction
        filename_match = re.search(r'\b(?:from|in)\s+(?:document|file)\s+["\']?([^"\']+)["\']?\b', query, re.IGNORECASE)
        if filename_match:
            filename = filename_match.group(1)
            filters.append(MetadataFilter(
                field="filename",
                operator="contains",
                value=filename
            ))
            logger.debug(f"Extracted filename filter: {filename}")

        # Table filter
        if re.search(r'\btable\b', query, re.IGNORECASE):
            filters.append(MetadataFilter(
                field="is_table",
                operator="eq",
                value=True
            ))
            logger.debug("Extracted is_table filter: True")

        return filters

    @staticmethod
    def build_vector_store_filter(
        filters: List[MetadataFilter],
        vector_store_type: str
    ) -> Any:
        """
        Convert generic filters to vector-store-specific format.

        For FAISS: Return None (will use metadata store to get chunk IDs)
        For LanceDB: Return SQL-like where clause
        For ChromaDB: Return where dict
        """
        if not filters:
            return None

        if vector_store_type == "lancedb":
            return FilterBuilder._build_lancedb_filter(filters)
        elif vector_store_type == "chroma":
            return FilterBuilder._build_chroma_filter(filters)
        else:
            # FAISS doesn't support native filtering
            return None

    @staticmethod
    def _build_lancedb_filter(filters: List[MetadataFilter]) -> str:
        """Build LanceDB SQL where clause."""
        clauses = []

        for f in filters:
            # LanceDB stores metadata as JSON, need to extract
            if f.operator == "eq":
                if isinstance(f.value, str):
                    clauses.append(f"json_extract(metadata, '$.{f.field}') = '{f.value}'")
                else:
                    clauses.append(f"json_extract(metadata, '$.{f.field}') = {f.value}")
            elif f.operator == "contains":
                clauses.append(f"json_extract(metadata, '$.{f.field}') LIKE '%{f.value}%'")
            elif f.operator == "gt":
                clauses.append(f"json_extract(metadata, '$.{f.field}') > {f.value}")
            elif f.operator == "lt":
                clauses.append(f"json_extract(metadata, '$.{f.field}') < {f.value}")
            elif f.operator == "gte":
                clauses.append(f"json_extract(metadata, '$.{f.field}') >= {f.value}")
            elif f.operator == "lte":
                clauses.append(f"json_extract(metadata, '$.{f.field}') <= {f.value}")

        return " AND ".join(clauses) if clauses else None

    @staticmethod
    def _build_chroma_filter(filters: List[MetadataFilter]) -> Dict[str, Any]:
        """Build ChromaDB where clause dict."""
        chroma_filter = {}

        for f in filters:
            if f.operator == "eq":
                chroma_filter[f.field] = {"$eq": f.value}
            elif f.operator == "ne":
                chroma_filter[f.field] = {"$ne": f.value}
            elif f.operator == "gt":
                chroma_filter[f.field] = {"$gt": f.value}
            elif f.operator == "lt":
                chroma_filter[f.field] = {"$lt": f.value}
            elif f.operator == "gte":
                chroma_filter[f.field] = {"$gte": f.value}
            elif f.operator == "lte":
                chroma_filter[f.field] = {"$lte": f.value}
            elif f.operator == "in":
                chroma_filter[f.field] = {"$in": f.value}
            elif f.operator == "contains":
                # ChromaDB doesn't have native contains, use regex
                chroma_filter[f.field] = {"$regex": f".*{f.value}.*"}

        return chroma_filter if chroma_filter else None


class MetadataStore:
    """
    Fast metadata storage and filtering using SQLite.
    Used alongside vector stores that don't support native filtering (like FAISS).
    """

    def __init__(self, db_path: str = "./data/metadata.db"):
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._create_indexes()

        logger.info(f"MetadataStore initialized at {db_path}")

    def _create_tables(self):
        """Create metadata tables with extensive indexing."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                chunk_id TEXT PRIMARY KEY,
                filename TEXT,
                doc_type TEXT,
                date_created TEXT,
                date_year INTEGER,
                quarter INTEGER,
                section TEXT,
                page_number INTEGER,
                is_table BOOLEAN DEFAULT 0,
                table_type TEXT,
                entity_types TEXT,  -- JSON array
                custom_metadata TEXT,  -- JSON for extensibility
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _create_indexes(self):
        """Create indexes for fast filtering."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_filename ON chunk_metadata(filename)",
            "CREATE INDEX IF NOT EXISTS idx_doc_type ON chunk_metadata(doc_type)",
            "CREATE INDEX IF NOT EXISTS idx_date_year ON chunk_metadata(date_year)",
            "CREATE INDEX IF NOT EXISTS idx_quarter ON chunk_metadata(quarter)",
            "CREATE INDEX IF NOT EXISTS idx_section ON chunk_metadata(section)",
            "CREATE INDEX IF NOT EXISTS idx_is_table ON chunk_metadata(is_table)",
            "CREATE INDEX IF NOT EXISTS idx_page_number ON chunk_metadata(page_number)",
        ]
        for idx in indexes:
            self.conn.execute(idx)
        self.conn.commit()
        logger.debug(f"Created {len(indexes)} metadata indexes")

    def add_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]):
        """Add or update metadata for a chunk."""
        # Extract standard fields
        filename = metadata.get("filename")
        doc_type = metadata.get("doc_type")
        date_created = metadata.get("date_created")
        date_year = metadata.get("date_year") or metadata.get("year")
        quarter = metadata.get("quarter")
        section = metadata.get("section")
        page_number = metadata.get("page_number") or metadata.get("page")
        is_table = metadata.get("is_table", False)
        table_type = metadata.get("table_type")

        # Entity types as JSON
        entity_types = metadata.get("entity_types", [])
        if entity_types and isinstance(entity_types, list):
            entity_types_json = json.dumps(entity_types)
        else:
            entity_types_json = None

        # Custom metadata (everything else)
        standard_fields = {
            "filename", "doc_type", "date_created", "date_year", "year", "quarter",
            "section", "page_number", "page", "is_table", "table_type", "entity_types"
        }
        custom = {k: v for k, v in metadata.items() if k not in standard_fields}
        custom_json = json.dumps(custom) if custom else None

        self.conn.execute("""
            INSERT OR REPLACE INTO chunk_metadata
            (chunk_id, filename, doc_type, date_created, date_year, quarter, section,
             page_number, is_table, table_type, entity_types, custom_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id, filename, doc_type, date_created, date_year, quarter, section,
            page_number, is_table, table_type, entity_types_json, custom_json
        ))
        self.conn.commit()

    def add_bulk(self, chunk_metadata_list: List[tuple]):
        """Bulk add metadata for efficiency."""
        self.conn.executemany("""
            INSERT OR REPLACE INTO chunk_metadata
            (chunk_id, filename, doc_type, date_created, date_year, quarter, section,
             page_number, is_table, table_type, entity_types, custom_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, chunk_metadata_list)
        self.conn.commit()
        logger.debug(f"Bulk added {len(chunk_metadata_list)} chunk metadata entries")

    def filter(self, filters: List[MetadataFilter]) -> List[str]:
        """
        Return chunk IDs matching all filters.

        Args:
            filters: List of MetadataFilter objects

        Returns:
            List of chunk IDs that match all filters
        """
        if not filters:
            # No filters - return all chunk IDs
            cursor = self.conn.execute("SELECT chunk_id FROM chunk_metadata")
            return [row[0] for row in cursor.fetchall()]

        where_clauses = []
        params = []

        for f in filters:
            if f.operator == "eq":
                where_clauses.append(f"{f.field} = ?")
                params.append(f.value)
            elif f.operator == "ne":
                where_clauses.append(f"{f.field} != ?")
                params.append(f.value)
            elif f.operator == "gt":
                where_clauses.append(f"{f.field} > ?")
                params.append(f.value)
            elif f.operator == "lt":
                where_clauses.append(f"{f.field} < ?")
                params.append(f.value)
            elif f.operator == "gte":
                where_clauses.append(f"{f.field} >= ?")
                params.append(f.value)
            elif f.operator == "lte":
                where_clauses.append(f"{f.field} <= ?")
                params.append(f.value)
            elif f.operator == "in":
                if isinstance(f.value, list):
                    placeholders = ",".join(["?"] * len(f.value))
                    where_clauses.append(f"{f.field} IN ({placeholders})")
                    params.extend(f.value)
            elif f.operator == "contains":
                where_clauses.append(f"{f.field} LIKE ?")
                params.append(f"%{f.value}%")
            elif f.operator == "regex":
                # SQLite doesn't have native regex, use LIKE as approximation
                where_clauses.append(f"{f.field} LIKE ?")
                params.append(f"%{f.value}%")
            elif f.operator == "range":
                # Value should be a tuple (min, max)
                if isinstance(f.value, (tuple, list)) and len(f.value) == 2:
                    where_clauses.append(f"{f.field} BETWEEN ? AND ?")
                    params.extend(f.value)

        if not where_clauses:
            cursor = self.conn.execute("SELECT chunk_id FROM chunk_metadata")
        else:
            query = f"""
                SELECT chunk_id FROM chunk_metadata
                WHERE {' AND '.join(where_clauses)}
            """
            cursor = self.conn.execute(query, params)

        chunk_ids = [row[0] for row in cursor.fetchall()]

        logger.info(f"Filter matched {len(chunk_ids)} chunks out of {self.count_total()} total")
        logger.debug(f"Filters applied: {filters}")

        return chunk_ids

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific chunk."""
        cursor = self.conn.execute("""
            SELECT filename, doc_type, date_created, date_year, quarter, section,
                   page_number, is_table, table_type, entity_types, custom_metadata
            FROM chunk_metadata
            WHERE chunk_id = ?
        """, (chunk_id,))

        row = cursor.fetchone()
        if not row:
            return None

        metadata = {
            "filename": row[0],
            "doc_type": row[1],
            "date_created": row[2],
            "date_year": row[3],
            "quarter": row[4],
            "section": row[5],
            "page_number": row[6],
            "is_table": bool(row[7]),
            "table_type": row[8],
        }

        # Parse JSON fields
        if row[9]:  # entity_types
            try:
                metadata["entity_types"] = json.loads(row[9])
            except json.JSONDecodeError:
                metadata["entity_types"] = []

        if row[10]:  # custom_metadata
            try:
                custom = json.loads(row[10])
                metadata.update(custom)
            except json.JSONDecodeError:
                pass

        return metadata

    def delete_by_filename(self, filename: str):
        """Delete all metadata for chunks from a specific file."""
        self.conn.execute("DELETE FROM chunk_metadata WHERE filename = ?", (filename,))
        self.conn.commit()
        logger.info(f"Deleted metadata for file: {filename}")

    def delete_chunk(self, chunk_id: str):
        """Delete metadata for a specific chunk."""
        self.conn.execute("DELETE FROM chunk_metadata WHERE chunk_id = ?", (chunk_id,))
        self.conn.commit()

    def count_total(self) -> int:
        """Count total chunks in metadata store."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunk_metadata")
        return cursor.fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored metadata."""
        stats = {}

        # Total chunks
        stats["total_chunks"] = self.count_total()

        # Unique filenames
        cursor = self.conn.execute("SELECT COUNT(DISTINCT filename) FROM chunk_metadata")
        stats["unique_files"] = cursor.fetchone()[0]

        # Doc type distribution
        cursor = self.conn.execute("""
            SELECT doc_type, COUNT(*) as count
            FROM chunk_metadata
            WHERE doc_type IS NOT NULL
            GROUP BY doc_type
            ORDER BY count DESC
        """)
        stats["doc_types"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Year distribution
        cursor = self.conn.execute("""
            SELECT date_year, COUNT(*) as count
            FROM chunk_metadata
            WHERE date_year IS NOT NULL
            GROUP BY date_year
            ORDER BY date_year DESC
        """)
        stats["years"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Table chunks count
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunk_metadata WHERE is_table = 1")
        stats["table_chunks"] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("MetadataStore connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
