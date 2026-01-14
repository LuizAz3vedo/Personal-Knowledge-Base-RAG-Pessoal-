"""Estrategias de chunking para documentos."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.config import settings

if TYPE_CHECKING:
    from src.ingestors.base import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Representa um chunk de documento."""

    content: str
    metadata: dict
    chunk_index: int
    total_chunks: int

    @property
    def id(self) -> str:
        """Gera ID unico para o chunk."""
        source = self.metadata.get("source", "unknown")
        return f"{source}::chunk_{self.chunk_index}"


class BaseChunker(ABC):
    """Classe base para estrategias de chunking."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Inicializa o chunker.

        Args:
            chunk_size: Tamanho maximo do chunk em caracteres.
            chunk_overlap: Sobreposicao entre chunks.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Divide um documento em chunks.

        Args:
            document: Documento a ser dividido.

        Returns:
            Lista de chunks.
        """
        ...


class RecursiveChunker(BaseChunker):
    """
    Chunker recursivo que tenta manter estrutura semantica.

    Divide por paragrafos primeiro, depois por sentencas,
    e finalmente por caracteres se necessario.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """
        Inicializa o chunker recursivo.

        Args:
            chunk_size: Tamanho maximo do chunk.
            chunk_overlap: Sobreposicao entre chunks.
            separators: Lista de separadores em ordem de preferencia.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n",  # Paragrafos
            "\n",  # Linhas
            ". ",  # Sentencas
            ", ",  # Clausulas
            " ",  # Palavras
            "",  # Caracteres
        ]

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide o documento recursivamente."""
        text = document.content
        chunks_text = self._split_text(text, self.separators)

        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk = Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "doc_type": document.doc_type,
                },
                chunk_index=i,
                total_chunks=len(chunks_text),
            )
            chunks.append(chunk)

        logger.debug(
            f"Documento '{document.source}' dividido em {len(chunks)} chunks"
        )
        return chunks

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Divide texto recursivamente usando separadores."""
        if not text:
            return []

        # Tentar cada separador
        for sep in separators:
            if sep == "":
                # Ultimo recurso: dividir por caracteres
                return self._split_by_size(text)

            if sep in text:
                splits = text.split(sep)

                # Recombinar splits pequenos
                chunks = []
                current_chunk = ""

                for split in splits:
                    test_chunk = (
                        current_chunk + sep + split if current_chunk else split
                    )

                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # Se o split individual e muito grande, recursao
                        if len(split) > self.chunk_size:
                            remaining_seps = separators[separators.index(sep) + 1 :]
                            chunks.extend(self._split_text(split, remaining_seps))
                            current_chunk = ""
                        else:
                            current_chunk = split

                if current_chunk:
                    chunks.append(current_chunk)

                # Adicionar overlap
                if self.chunk_overlap > 0:
                    chunks = self._add_overlap(chunks)

                return chunks

        return [text] if text else []

    def _split_by_size(self, text: str) -> list[str]:
        """Divide texto por tamanho fixo."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Adiciona sobreposicao entre chunks consecutivos."""
        if len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Pegar final do chunk anterior
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap :]

                # Encontrar limite de palavra
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1 :]

                chunk = overlap_text + chunk

            overlapped.append(chunk)

        return overlapped


class MarkdownChunker(BaseChunker):
    """Chunker especializado para Markdown."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *,
        respect_headers: bool = True,
    ) -> None:
        """
        Inicializa o chunker de Markdown.

        Args:
            chunk_size: Tamanho maximo do chunk.
            chunk_overlap: Sobreposicao entre chunks.
            respect_headers: Se deve manter headers com seus conteudos.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.respect_headers = respect_headers

    def chunk(self, document: Document) -> list[Chunk]:
        """Divide documento Markdown respeitando estrutura."""
        text = document.content

        # Dividir por secoes (headers)
        if self.respect_headers:
            sections = self._split_by_headers(text)
        else:
            sections = [text]

        # Aplicar chunking recursivo em cada secao
        recursive_chunker = RecursiveChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = []
        chunk_idx = 0

        for section in sections:
            # Criar documento temporario para a secao
            temp_doc = type(document)(
                content=section,
                metadata=document.metadata.copy(),
                source=document.source,
                doc_type=document.doc_type,
            )

            section_chunks = recursive_chunker.chunk(temp_doc)

            for chunk in section_chunks:
                chunk.chunk_index = chunk_idx
                chunks.append(chunk)
                chunk_idx += 1

        # Atualizar total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_by_headers(self, text: str) -> list[str]:
        """Divide texto por headers Markdown."""
        # Pattern para headers (# ## ### etc)
        header_pattern = r"^(#{1,6})\s+(.+)$"

        lines = text.split("\n")
        sections = []
        current_section: list[str] = []

        for line in lines:
            if re.match(header_pattern, line):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections


def get_chunker(doc_type: str) -> BaseChunker:
    """
    Retorna o chunker apropriado para o tipo de documento.

    Args:
        doc_type: Tipo do documento ('markdown', 'pdf', etc).

    Returns:
        Instancia do chunker apropriado.
    """
    chunkers: dict[str, type[BaseChunker]] = {
        "markdown": MarkdownChunker,
        "pdf": RecursiveChunker,
        "web": RecursiveChunker,
    }

    chunker_class = chunkers.get(doc_type, RecursiveChunker)
    return chunker_class()
