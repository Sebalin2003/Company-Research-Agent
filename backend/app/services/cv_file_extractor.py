from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


class CVExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedCV:
    filename: str
    content_type: str | None
    text: str

    @property
    def character_count(self) -> int:
        return len(self.text)


def extract_cv_text(
    filename: str,
    content_type: str | None,
    content: bytes,
    max_characters: int,
) -> ExtractedCV:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf_text(content)
    elif suffix == ".docx":
        text = extract_docx_text(content)
    else:
        raise CVExtractionError("Solo se aceptan archivos PDF o DOCX.")

    cleaned = clean_cv_text(text)
    if not cleaned:
        raise CVExtractionError("No se pudo extraer texto legible del CV.")
    if len(cleaned) > max_characters:
        raise CVExtractionError("El CV supera el limite permitido para esta version.")
    return ExtractedCV(filename=filename, content_type=content_type, text=cleaned)


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx_text(content: bytes) -> str:
    document = Document(BytesIO(content))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def clean_cv_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
