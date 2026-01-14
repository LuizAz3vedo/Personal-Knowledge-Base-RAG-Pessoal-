"""Ingestores de documentos."""

from src.ingestors.base import BaseIngestor, Document
from src.ingestors.markdown import MarkdownIngestor
from src.ingestors.pdf import PDFIngestor
from src.ingestors.web import WebIngestor

__all__ = [
    "BaseIngestor",
    "Document",
    "MarkdownIngestor",
    "PDFIngestor",
    "WebIngestor",
]
