"""
PDF Text Extractor for MORPHE Ingestion.
Uses pypdfium2 / pdfplumber to extract complete readable text from uploaded PDFs.
"""

from __future__ import annotations

import io
from typing import Union


def extract_text_from_pdf(pdf_bytes_or_path: Union[bytes, str]) -> str:
    """
    Extracts text preserving multi-page order and paragraph structure.
    Robust fallback across pdfplumber, pypdfium2, and raw utf-8 text.
    """
    # 1. Try pdfplumber first (standard for in-memory stream bytes)
    try:
        import pdfplumber
        if isinstance(pdf_bytes_or_path, bytes):
            stream = io.BytesIO(pdf_bytes_or_path)
            with pdfplumber.open(stream) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n\n".join(pages).strip()
                if text:
                    return text
        else:
            with pdfplumber.open(str(pdf_bytes_or_path)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n\n".join(pages).strip()
                if text:
                    return text
    except Exception:
        pass

    # 2. Try pypdfium2
    try:
        import pypdfium2 as pdfium
        if isinstance(pdf_bytes_or_path, bytes):
            stream = io.BytesIO(pdf_bytes_or_path)
            pdf = pdfium.PdfDocument(stream)
        else:
            pdf = pdfium.PdfDocument(str(pdf_bytes_or_path))

        text_parts = []
        for page in pdf:
            textpage = page.get_textpage()
            text_parts.append(textpage.get_text_range())

        full_text = "\n\n".join(text_parts).strip()
        if full_text:
            return full_text
    except Exception:
        pass

    # 3. Fallback: if user uploaded raw text file named .pdf or markdown
    if isinstance(pdf_bytes_or_path, bytes):
        try:
            decoded = pdf_bytes_or_path.decode("utf-8", errors="ignore").strip()
            if len(decoded) > 20:
                return decoded
        except Exception:
            pass

    raise RuntimeError("Could not extract readable text from the uploaded file.")

