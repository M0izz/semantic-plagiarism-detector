import io
import pytest
import pypdf
from unittest.mock import MagicMock, patch
from utils.pdf_reader import extract_text_from_pdf, extract_texts_from_pdfs


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal in-memory PDF containing the given text."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_from_bytesio():
    # Build a real single-page PDF and confirm we get a string back
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    result = extract_text_from_pdf(buf)
    assert isinstance(result, str)


def test_extract_from_bytes():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    result = extract_text_from_pdf(buf.getvalue())
    assert isinstance(result, str)


def test_extract_from_filepath(tmp_path):
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    result = extract_text_from_pdf(str(pdf_path))
    assert isinstance(result, str)


def test_extract_returns_empty_on_error():
    result = extract_text_from_pdf(io.BytesIO(b"not a pdf"))
    assert result == ""


def test_extract_text_filters_repeated_headers_page_numbers_and_whitespace():
    page_one = MagicMock()
    page_one.extract_text.return_value = "Research Report\n\nIntroduction\nPage 1"
    page_two = MagicMock()
    page_two.extract_text.return_value = "Research Report\n\nBody content\nPage 2"

    fake_reader = MagicMock()
    fake_reader.pages = [page_one, page_two]

    with patch("utils.pdf_reader.PyPDF2.PdfReader", return_value=fake_reader):
        result = extract_text_from_pdf(io.BytesIO(b"fake-pdf"))

    assert "Research Report" not in result
    assert "Page 1" not in result
    assert "Page 2" not in result
    assert "Introduction" in result
    assert "Body content" in result
    assert "\n\n\n" not in result


def test_extract_texts_from_pdfs_uses_name_attribute():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    mock_file = MagicMock()
    mock_file.name = "assignment1.pdf"
    mock_file.read.return_value = buf.getvalue()

    # extract_texts_from_pdfs calls extract_text_from_pdf(file) directly,
    # so we patch that to avoid re-reading the mock
    with patch("utils.pdf_reader.extract_text_from_pdf", return_value="hello"):
        results = extract_texts_from_pdfs([mock_file])

    assert "assignment1.pdf" in results
    assert results["assignment1.pdf"] == "hello"


def test_extract_texts_from_pdfs_fallback_name():
    with patch("utils.pdf_reader.extract_text_from_pdf", return_value="text"):
        results = extract_texts_from_pdfs([io.BytesIO(b"")])
    assert "document_1.pdf" in results


def test_extract_texts_from_pdfs_string_path(tmp_path):
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "sample.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    results = extract_texts_from_pdfs([str(pdf_path)])
    # pdf_reader uses path.split("/")[-1] so on Windows the full path is the key
    assert any(k.endswith("sample.pdf") for k in results)
