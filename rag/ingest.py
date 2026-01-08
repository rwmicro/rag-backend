"""
Document Ingestion & Parsing Module
Supports PDF and Markdown with advanced extraction
Uses pymupdf_layout for enhanced layout analysis
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import pymupdf  # PyMuPDF (fitz)

# IMPORTANT: Import pymupdf.layout BEFORE pymupdf4llm to activate layout features
try:
    import pymupdf.layout  # Enables advanced layout analysis
    LAYOUT_ENABLED = True
except ImportError:
    LAYOUT_ENABLED = False

import pymupdf4llm
from markdown_it import MarkdownIt
from loguru import logger


@dataclass
class Document:
    """Represents a parsed document"""

    content: str
    metadata: Dict[str, Any]
    doc_type: str  # "pdf" or "markdown"


def parse_pdf(file_path: Union[str, Path], **kwargs) -> Document:
    """
    Parse PDF file using PyMuPDF with advanced layout analysis

    Args:
        file_path: Path to PDF file
        **kwargs: Additional parsing options
            - output_format: str = "markdown" ("markdown", "json", or "text")
            - page_chunks: bool = False (return chunks per page)
            - write_images: bool = False (extract images)
            - ignore_headers: bool = True (exclude headers from output)
            - ignore_footers: bool = True (exclude footers from output)
            - use_ocr: bool = True (auto OCR for scanned pages)

    Returns:
        Document object with content and metadata
    """
    file_path = Path(file_path)
    layout_status = "with layout analysis" if LAYOUT_ENABLED else "standard"
    logger.info(f"Parsing PDF ({layout_status}): {file_path.name}")

    try:
        # Open PDF with PyMuPDF
        doc = pymupdf.open(file_path)

        # Check if PDF is empty
        if len(doc) == 0:
            raise ValueError("PDF file is empty (0 pages)")

        # Extract metadata
        metadata = {
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": "pdf",
            "total_pages": len(doc),
            "layout_analysis": LAYOUT_ENABLED,
        }

        # Add PDF metadata if available
        pdf_metadata = doc.metadata
        if pdf_metadata:
            if pdf_metadata.get("title"):
                metadata["title"] = pdf_metadata["title"]
            if pdf_metadata.get("author"):
                metadata["author"] = pdf_metadata["author"]
            if pdf_metadata.get("subject"):
                metadata["subject"] = pdf_metadata["subject"]
            if pdf_metadata.get("keywords"):
                metadata["keywords"] = pdf_metadata["keywords"]

        # Configure extraction options
        output_format = kwargs.get("output_format", "markdown")
        page_chunks = kwargs.get("page_chunks", False)
        write_images = kwargs.get("write_images", False)
        ignore_headers = kwargs.get("ignore_headers", True)
        ignore_footers = kwargs.get("ignore_footers", True)
        use_ocr = kwargs.get("use_ocr", True)

        # Extract with enhanced layout analysis (if available)
        # pymupdf_layout provides:
        # - Better table detection
        # - Header/footer detection and exclusion
        # - Footnote detection
        # - List item detection
        # - Better paragraph detection
        # - Auto OCR for scanned pages

        extraction_kwargs = {
            "page_chunks": page_chunks,
            "write_images": write_images,
        }

        # Add header/footer control if using layout analysis
        if LAYOUT_ENABLED:
            extraction_kwargs["header"] = not ignore_headers
            extraction_kwargs["footer"] = not ignore_footers
            extraction_kwargs["use_ocr"] = use_ocr

        # Extract content based on format with fallback on error
        try:
            if output_format == "json":
                # JSON format provides structured data
                if hasattr(pymupdf4llm, "to_json"):
                    import json
                    content_data = pymupdf4llm.to_json(doc, **extraction_kwargs)
                    # If it returns a string, keep it; if dict, convert to formatted JSON
                    if isinstance(content_data, str):
                        content = content_data
                    else:
                        content = json.dumps(content_data, indent=2, ensure_ascii=False)
                    metadata["output_format"] = "json"
                else:
                    # Fallback to markdown if to_json not available
                    logger.warning("to_json() not available, using markdown instead")
                    content = pymupdf4llm.to_markdown(doc, **extraction_kwargs)
                    metadata["output_format"] = "markdown"
            elif output_format == "text":
                # Plain text format
                if hasattr(pymupdf4llm, "to_text"):
                    content = pymupdf4llm.to_text(doc, **extraction_kwargs)
                    metadata["output_format"] = "text"
                else:
                    # Fallback to markdown
                    logger.warning("to_text() not available, using markdown instead")
                    content = pymupdf4llm.to_markdown(doc, **extraction_kwargs)
                    metadata["output_format"] = "markdown"
            else:
                # Markdown format (default)
                # This preserves structure: headers, lists, tables, etc.
                content = pymupdf4llm.to_markdown(doc, **extraction_kwargs)
                metadata["output_format"] = "markdown"
        except Exception as parse_error:
            # If advanced parsing fails, fallback to basic text extraction
            logger.warning(f"Advanced PDF parsing failed ({parse_error}), falling back to basic text extraction")
            content_parts = []
            for page_num, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text("text")
                    if page_text.strip():
                        content_parts.append(f"\n\n--- Page {page_num} ---\n\n")
                        content_parts.append(page_text)
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num}: {page_error}")
                    continue

            content = "".join(content_parts)
            metadata["output_format"] = "text"
            metadata["extraction_method"] = "fallback"

            if not content or not content.strip():
                raise ValueError("Failed to extract any text from PDF using both advanced and basic methods")

        # Check if content is empty
        if not content or not content.strip():
            raise ValueError("PDF parsing produced empty content. File may be scanned/image-only or corrupted.")

        # Calculate page statistics
        page_count = 0
        total_chars = 0
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                page_count += 1
                total_chars += len(page_text)

        # Ensure we have at least some pages with text
        if page_count == 0:
            logger.warning("No text found in PDF - may be image-based or scanned")
            # Don't raise error, but log warning

        # Store simple scalar metadata
        metadata["page_count"] = page_count
        metadata["total_chars"] = total_chars
        metadata["avg_chars_per_page"] = total_chars / page_count if page_count > 0 else 0

        doc.close()

        features = []
        if LAYOUT_ENABLED:
            features.append("layout analysis")
            if ignore_headers:
                features.append("no headers")
            if ignore_footers:
                features.append("no footers")
            if use_ocr:
                features.append("auto OCR")

        feature_str = f" ({', '.join(features)})" if features else ""
        logger.info(f"✓ Parsed PDF: {page_count} pages, {len(content)} chars{feature_str}")

        return Document(
            content=content,
            metadata=metadata,
            doc_type="pdf"
        )

    except Exception as e:
        logger.error(f"Error parsing PDF {file_path}: {e}")
        raise


def parse_markdown(file_path: Union[str, Path], **kwargs) -> Document:
    """
    Parse Markdown file with structure extraction

    Args:
        file_path: Path to Markdown file
        **kwargs: Additional parsing options

    Returns:
        Document object with content and metadata
    """
    file_path = Path(file_path)
    logger.info(f"Parsing Markdown: {file_path.name}")

    try:
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if content is empty
        if not content or not content.strip():
            raise ValueError(f"Markdown file is empty: {file_path}")

        # Parse with markdown-it to extract structure
        md = MarkdownIt()
        tokens = md.parse(content)

        # Extract headers for metadata
        headers = []
        for token in tokens:
            if token.type == "heading_open":
                # Find the inline content token that follows
                idx = tokens.index(token)
                if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
                    # Safely extract level from tag (e.g., "h1" -> 1)
                    try:
                        level = int(token.tag[1]) if len(token.tag) > 1 else 1
                    except (ValueError, IndexError):
                        level = 1
                    headers.append({
                        "level": level,
                        "text": tokens[idx + 1].content
                    })

        # Extract title (first H1 or H2)
        title = None
        for header in headers:
            if header["level"] in [1, 2]:
                title = header["text"]
                break

        metadata = {
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": "markdown",
            "title": title or "",
            "headers": headers,
            "header_count": len(headers),
            "char_count": len(content),
        }

        logger.info(f"✓ Parsed Markdown: {len(headers)} headers, {len(content)} chars")

        return Document(
            content=content,
            metadata=metadata,
            doc_type="markdown"
        )

    except Exception as e:
        logger.error(f"Error parsing Markdown {file_path}: {e}")
        raise


class DocumentIngestor:
    """
    Document ingestion orchestrator
    Handles multiple file types and batch processing
    """

    def __init__(self):
        self.supported_extensions = {
            ".pdf": parse_pdf,
            ".md": parse_markdown,
            ".markdown": parse_markdown,
            ".csv": parse_csv,
            ".tsv": parse_csv,
        }

    def ingest_file(self, file_path: Union[str, Path], **kwargs) -> Document:
        """
        Ingest a single file

        Args:
            file_path: Path to file
            **kwargs: Parser-specific options

        Returns:
            Parsed Document
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self.supported_extensions.keys())}"
            )

        parser = self.supported_extensions[ext]
        return parser(file_path, **kwargs)

    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Ingest all supported files from a directory

        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            **kwargs: Parser-specific options

        Returns:
            List of parsed Documents
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        logger.info(f"Ingesting directory: {directory}")

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.ingest_file(file_path, **kwargs)
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path.name}: {e}")
                    continue

        logger.info(f"✓ Ingested {len(documents)} documents from {directory}")
        return documents

    def ingest_from_buffer(
        self,
        content: bytes,
        filename: str,
        doc_type: str,
        **kwargs
    ) -> Document:
        """
        Ingest document from memory buffer (for API uploads)

        Args:
            content: File content as bytes
            filename: Original filename
            doc_type: Document type ("pdf", "markdown", "csv", "tsv")
            **kwargs: Parser-specific options

        Returns:
            Parsed Document
        """
        import tempfile

        logger.info(f"Ingesting from buffer: {filename} ({len(content)} bytes)")

        # Check if content is empty
        if not content or len(content) == 0:
            raise ValueError(f"File '{filename}' is empty (0 bytes). Please upload a file with content.")

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{doc_type}",
            mode="wb"
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            doc = self.ingest_file(tmp_path, **kwargs)
            # Update filename in metadata
            doc.metadata["filename"] = filename
            return doc
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def ingest_url(self, url: str, **kwargs) -> Document:
        """
        Ingest content from a URL

        Args:
            url: Website URL
            **kwargs: Additional options

        Returns:
            Parsed Document
        """
        return scrape_url(url, **kwargs)


def parse_csv(file_path: Union[str, Path], **kwargs) -> Document:
    """
    Parse CSV/TSV file into Markdown table format
    """
    import pandas as pd
    import csv
    
    file_path = Path(file_path)
    logger.info(f"Parsing CSV/TSV: {file_path.name}")

    try:
        # Determine separator based on extension
        sep = '\t' if file_path.suffix.lower() == '.tsv' else ','
        
        # Read CSV
        df = pd.read_csv(file_path, sep=sep)
        
        # Convert to markdown
        content = df.to_markdown(index=False)
        
        metadata = {
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": "csv" if sep == ',' else "tsv",
            "rows": len(df),
            "columns": list(df.columns),
        }
        
        logger.info(f"✓ Parsed CSV/TSV: {len(df)} rows, {len(df.columns)} columns")
        
        return Document(
            content=content,
            metadata=metadata,
            doc_type="markdown" # Treat as markdown for chunking
        )
    except Exception as e:
        logger.error(f"Error parsing CSV {file_path}: {e}")
        raise


def scrape_url(url: str, **kwargs) -> Document:
    """
    Scrape content from a URL and convert to Markdown
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    
    logger.info(f"Scraping URL: {url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "iframe"]):
            script.decompose()
            
        # Get text
        title = soup.title.string if soup.title else url
        
        # Simple HTML to Markdown conversion (can be improved with libraries like markdownify)
        # For now, we'll just extract text and preserve some structure
        
        content = f"# {title}\n\n"
        
        # Process headings and paragraphs
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                content += f"{'#' * level} {element.get_text().strip()}\n\n"
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    content += f"{text}\n\n"
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    content += f"- {li.get_text().strip()}\n"
                content += "\n"
                
        metadata = {
            "filename": url,
            "source": url,
            "file_type": "url",
            "title": title,
        }
        
        logger.info(f"✓ Scraped URL: {title} ({len(content)} chars)")
        
        return Document(
            content=content,
            metadata=metadata,
            doc_type="markdown"
        )
        
    except Exception as e:
        logger.error(f"Error scraping URL {url}: {e}")
        raise
