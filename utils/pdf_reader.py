"""
pdf_reader.py
-------------
Handles extraction of raw text from uploaded PDF files using pypdf.
Supports both file paths and file-like objects (e.g., Streamlit UploadedFile).
"""

import io
import re
from collections import Counter
from typing import List, Union

try:
    import pypdf as PyPDF2
except ImportError:  # pragma: no cover - fallback for older environments
    import PyPDF2


def _is_page_number(line: str) -> bool:
    """Return True for simple standalone page-number lines."""
    cleaned = re.sub(r"[\u00a0\u200b]", " ", line).strip()
    if not cleaned:
        return False

    return bool(re.fullmatch(r"(?:page|p\.?)\s*\d+", cleaned, flags=re.IGNORECASE)) or bool(
        re.fullmatch(r"\d{1,3}", cleaned)
    )


def _clean_page_text(page_text: str) -> List[str]:
    """Clean one page of extracted text by removing page numbers and repeated boundary text."""
    lines = []
    for raw_line in page_text.splitlines():
        cleaned = re.sub(r"[\u00a0\u200b]", " ", raw_line).strip()
        if not cleaned:
            continue
        if _is_page_number(cleaned):
            continue
        lines.append(cleaned)

    return lines


def _remove_repeated_boundary_lines(page_lines: List[List[str]]) -> List[List[str]]:
    """Remove repeated first/last lines that appear across pages, typically headers/footers."""
    if not page_lines:
        return []

    cleaned_pages = [list(lines) for lines in page_lines]
    for position in ("start", "end"):
        candidates = []
        for lines in cleaned_pages:
            if not lines:
                continue
            candidate = lines[0] if position == "start" else lines[-1]
            candidates.append(candidate)

        counts = Counter(candidates)
        repeated = {
            line
            for line, count in counts.items()
            if count > 1 and len(line) <= 60 and not _is_page_number(line)
        }

        for index, lines in enumerate(cleaned_pages):
            if not lines:
                continue
            if position == "start" and lines and lines[0] in repeated:
                cleaned_pages[index] = lines[1:]
            elif position == "end" and lines and lines[-1] in repeated:
                cleaned_pages[index] = lines[:-1]

    return cleaned_pages


def _normalize_whitespace(page_lines: List[List[str]]) -> str:
    """Join cleaned lines and collapse excessive whitespace."""
    cleaned_lines = [line for lines in page_lines for line in lines]
    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def extract_text_from_pdf(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file: A file path (str), raw bytes, or a file-like object (BytesIO / UploadedFile).

    Returns:
        A single string containing all extracted text from the PDF.
        Returns an empty string if extraction fails.
    """
    page_lines = []

    try:
        # Normalise input → always work with a file-like object
        if isinstance(file, str):
            pdf_file = open(file, "rb")
            close_after = True
        elif isinstance(file, bytes):
            pdf_file = io.BytesIO(file)
            close_after = False
        else:
            # Assume it is already a file-like object (BytesIO, UploadedFile …)
            pdf_file = file
            close_after = False

        reader = PyPDF2.PdfReader(pdf_file)

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted = page.extract_text()
            if extracted:
                page_lines.append(_clean_page_text(extracted))

        if close_after:
            pdf_file.close()

    except Exception as e:
        print(f"[pdf_reader] Error reading PDF: {e}")

    if not page_lines:
        return ""

    cleaned_pages = _remove_repeated_boundary_lines(page_lines)
    return _normalize_whitespace(cleaned_pages)


def extract_texts_from_pdfs(files: list) -> dict:
    """
    Extract text from multiple PDF files.

    Args:
        files: List of file paths or file-like objects.

    Returns:
        A dict mapping file name → extracted text string.
    """
    results = {}

    for file in files:
        # Determine a display name for the file
        if hasattr(file, "name"):
            name = file.name  # Streamlit UploadedFile exposes .name
        elif isinstance(file, str):
            name = file.split("/")[-1]
        else:
            name = f"document_{len(results) + 1}.pdf"

        results[name] = extract_text_from_pdf(file)

    return results
