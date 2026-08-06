"""
Document loader using industry-standard libraries.
Replaces ad-hoc multimodal_loader.py with proper tools.

Uses:
- pymupdf4llm: For PDF to Markdown conversion (optimized for LLM workflows)
- pandas: For CSV/JSON loading
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Union
from io import BytesIO, StringIO

from .models import (
    ExtractionMode,
    ContentType,
    ExtractedContent,
    DocumentResult,
    ColumnMapping,
)

# Lazy imports for optional dependencies
_pymupdf4llm = None
_PIL = None


def _get_pymupdf4llm():
    global _pymupdf4llm
    if _pymupdf4llm is None:
        try:
            import pymupdf4llm
            _pymupdf4llm = pymupdf4llm
        except ImportError:
            raise ImportError(
                "pymupdf4llm not installed. Run: pip install pymupdf4llm"
            )
    return _pymupdf4llm


def _get_pil():
    global _PIL
    if _PIL is None:
        try:
            from PIL import Image
            _PIL = Image
        except ImportError:
            raise ImportError("Pillow not installed. Run: pip install Pillow")
    return _PIL


# ============================================================================
# JSON/CSV Loading (Simple, no reinvention needed)
# ============================================================================

def load_json(content: bytes) -> pd.DataFrame:
    """Load JSON or JSONL into DataFrame."""
    text = content.decode("utf-8")

    # Try JSONL first (one object per line)
    lines = text.strip().split("\n")
    if len(lines) > 1:
        try:
            records = [json.loads(line) for line in lines if line.strip()]
            return pd.DataFrame(records)
        except json.JSONDecodeError:
            pass

    # Regular JSON
    data = json.loads(text)
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        # Check for common wrapper keys
        for key in ("data", "records", "samples", "items", "examples"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        return pd.DataFrame([data])

    raise ValueError("Unsupported JSON structure")


def load_csv(content: bytes) -> pd.DataFrame:
    """Load CSV with encoding detection."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(BytesIO(content), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("Could not parse CSV with any common encoding")


# ============================================================================
# PDF Loading using pymupdf4llm
# ============================================================================

def load_pdf(
    content: bytes,
    filename: str = "",
    mode: ExtractionMode = ExtractionMode.MARKDOWN,
) -> DocumentResult:
    """
    Load PDF using pymupdf4llm.

    Args:
        content: PDF file bytes
        filename: Original filename
        mode: Extraction mode (markdown, text, or chunks)

    Returns:
        DocumentResult with extracted content
    """
    pymupdf4llm = _get_pymupdf4llm()
    import fitz  # PyMuPDF, installed with pymupdf4llm

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(doc)

        if mode == ExtractionMode.MARKDOWN:
            # Get markdown output (best for LLM consumption)
            md_text = pymupdf4llm.to_markdown(doc)
            extracted = [
                ExtractedContent(
                    content_type=ContentType.TEXT,
                    text=md_text,
                    source_file=filename,
                    metadata={"format": "markdown", "page_count": page_count},
                )
            ]

        elif mode == ExtractionMode.CHUNKS:
            # Get chunked output (good for RAG)
            chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
            extracted = [
                ExtractedContent(
                    content_type=ContentType.PAGE,
                    text=chunk["text"] if isinstance(chunk, dict) else str(chunk),
                    source_file=filename,
                    page=i + 1,
                    metadata=chunk.get("metadata", {}) if isinstance(chunk, dict) else {},
                )
                for i, chunk in enumerate(chunks)
            ]

        else:  # TEXT mode
            # Plain text extraction
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            extracted = [
                ExtractedContent(
                    content_type=ContentType.TEXT,
                    text="\n\n".join(text_parts),
                    source_file=filename,
                    metadata={"format": "plain_text", "page_count": page_count},
                )
            ]

        doc.close()
        return DocumentResult(
            filename=filename,
            content=extracted,
            page_count=page_count,
            extraction_mode=mode,
        )

    except Exception as e:
        return DocumentResult(
            filename=filename,
            content=[],
            page_count=0,
            extraction_mode=mode,
            error=str(e),
        )


# ============================================================================
# Image Loading
# ============================================================================

def load_image(content: bytes, filename: str = "") -> DocumentResult:
    """Load image and extract basic metadata."""
    Image = _get_pil()

    try:
        img = Image.open(BytesIO(content))
        metadata = {
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "size_bytes": len(content),
        }

        # Note: OCR would require pytesseract, keeping it simple
        extracted = [
            ExtractedContent(
                content_type=ContentType.IMAGE,
                text=f"[Image: {img.width}x{img.height} {img.format}]",
                source_file=filename,
                metadata=metadata,
            )
        ]

        return DocumentResult(
            filename=filename,
            content=extracted,
            page_count=1,
            extraction_mode=ExtractionMode.TEXT,
        )

    except Exception as e:
        return DocumentResult(
            filename=filename,
            content=[],
            page_count=0,
            extraction_mode=ExtractionMode.TEXT,
            error=str(e),
        )


# ============================================================================
# Unified Loader
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".webp": "image",
}


def load_file(
    uploaded_file,
    mode: ExtractionMode = ExtractionMode.MARKDOWN,
) -> pd.DataFrame:
    """
    Load any supported file type into a DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object
        mode: Extraction mode for PDFs

    Returns:
        DataFrame with loaded content
    """
    if uploaded_file is None:
        raise ValueError("No file provided")

    content = uploaded_file.getvalue()
    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()

    file_type = SUPPORTED_EXTENSIONS.get(ext)
    if not file_type:
        raise ValueError(f"Unsupported file type: {ext}")

    if file_type == "json":
        return load_json(content)
    elif file_type == "csv":
        return load_csv(content)
    elif file_type == "pdf":
        result = load_pdf(content, filename, mode)
        if result.error:
            raise ValueError(result.error)
        # Convert to DataFrame for consistency
        return pd.DataFrame([
            {
                "text": c.text,
                "type": c.content_type.value,
                "page": c.page,
                "source_file": c.source_file,
                **c.metadata,
            }
            for c in result.content
        ])
    elif file_type == "image":
        result = load_image(content, filename)
        if result.error:
            raise ValueError(result.error)
        return pd.DataFrame([
            {
                "text": c.text,
                "type": c.content_type.value,
                "source_file": c.source_file,
                **c.metadata,
            }
            for c in result.content
        ])

    raise ValueError(f"Unknown file type: {file_type}")


def load_files(
    uploaded_files: List,
    mode: ExtractionMode = ExtractionMode.MARKDOWN,
    progress_callback=None,
) -> pd.DataFrame:
    """Load multiple files and combine into a single DataFrame."""
    dfs = []
    total = len(uploaded_files)

    for i, f in enumerate(uploaded_files):
        if progress_callback:
            progress_callback((i + 1) / total, f"Loading {f.name}")
        try:
            df = load_file(f, mode)
            dfs.append(df)
        except Exception as e:
            # Add error row instead of failing silently
            dfs.append(pd.DataFrame([{
                "source_file": f.name,
                "error": str(e),
                "text": "",
            }]))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ============================================================================
# Column Inference
# ============================================================================

def infer_columns(df: pd.DataFrame) -> ColumnMapping:
    """Infer semantic column roles from column names."""
    mapping = ColumnMapping()
    cols_lower = {c: c.lower() for c in df.columns}

    patterns = {
        "input_col": ["input", "prompt", "question", "query", "text", "source"],
        "output_col": ["output", "response", "prediction", "generated", "completion"],
        "expected_col": ["expected", "reference", "target", "answer", "gold", "label"],
    }

    for field, keywords in patterns.items():
        for col, col_lower in cols_lower.items():
            if any(kw in col_lower for kw in keywords):
                setattr(mapping, field, col)
                break

    return mapping
