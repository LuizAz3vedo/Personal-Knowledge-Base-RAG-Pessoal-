"""Classe base para ingestores de documentos."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Representa um documento processado."""

    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    doc_type: str = ""

    def __post_init__(self) -> None:
        """Preenche metadados padrao."""
        if "source" not in self.metadata:
            self.metadata["source"] = self.source
        if "doc_type" not in self.metadata:
            self.metadata["doc_type"] = self.doc_type
        if "ingested_at" not in self.metadata:
            self.metadata["ingested_at"] = datetime.now().isoformat()


class BaseIngestor(ABC):
    """Classe base abstrata para ingestores."""

    doc_type: str = "generic"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Inicializa o ingestor.

        Args:
            chunk_size: Tamanho dos chunks (usa config se nao especificado).
            chunk_overlap: Sobreposicao dos chunks.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    @abstractmethod
    def load(self, source: Path | str) -> Iterator[Document]:
        """
        Carrega documentos da fonte.

        Args:
            source: Caminho ou URL da fonte.

        Yields:
            Documentos carregados.
        """
        ...

    def chunk_document(self, document: Document) -> list[Document]:
        """
        Divide um documento em chunks menores.

        Args:
            document: Documento a ser dividido.

        Returns:
            Lista de documentos (chunks).
        """
        if not document.content.strip():
            return []

        chunks = self.text_splitter.split_text(document.content)

        return [
            Document(
                content=chunk,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
                source=document.source,
                doc_type=document.doc_type,
            )
            for i, chunk in enumerate(chunks)
        ]

    def ingest(self, source: Path | str) -> list[Document]:
        """
        Carrega e processa documentos da fonte.

        Args:
            source: Caminho ou URL da fonte.

        Returns:
            Lista de chunks prontos para indexacao.
        """
        all_chunks: list[Document] = []

        for document in self.load(source):
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
            logger.debug(
                f"Documento '{document.source}' dividido em {len(chunks)} chunks"
            )

        logger.info(f"Ingestao completa: {len(all_chunks)} chunks gerados")
        return all_chunks
