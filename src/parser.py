import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, one dict per page.

    Returns a list of dicts:
        [{"page_number": 1, "text": "..."}, ...]

    Skips pages with no extractable text (scanned images, blank pages).
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page_number": i, "text": text.strip()})

    return pages
