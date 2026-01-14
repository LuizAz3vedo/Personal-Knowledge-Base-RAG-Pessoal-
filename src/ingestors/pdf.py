"""Ingestor para arquivos PDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber

from src.ingestors.base import BaseIngestor, Document

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class PDFIngestor(BaseIngestor):
    """Ingestor para arquivos PDF."""

    doc_type: str = "pdf"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        extract_tables: bool = False,
        pages_per_chunk: int | None = None,
    ) -> None:
        """
        Inicializa o ingestor PDF.

        Args:
            chunk_size: Tamanho dos chunks.
            chunk_overlap: Sobreposicao dos chunks.
            extract_tables: Tentar extrair tabelas como texto.
            pages_per_chunk: Agrupar N paginas por documento (None = todas juntas).
        """
        super().__init__(chunk_size, chunk_overlap)
        self.extract_tables = extract_tables
        self.pages_per_chunk = pages_per_chunk

    def _extract_text_from_page(self, page: pdfplumber.page.Page) -> str:
        """Extrai texto de uma pagina."""
        text_parts: list[str] = []

        # Extrair texto principal
        text = page.extract_text()
        if text:
            text_parts.append(text)

        # Extrair tabelas se configurado
        if self.extract_tables:
            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_text = self._table_to_text(table)
                    if table_text:
                        text_parts.append(f"\n[Tabela]\n{table_text}")

        return "\n".join(text_parts)

    def _table_to_text(self, table: list) -> str:
        """Converte tabela para texto."""
        rows = []
        for row in table:
            if row:
                cells = [str(cell) if cell else "" for cell in row]
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _parse_pdf(self, file_path: Path) -> Iterator[Document]:
        """
        Faz parse de um arquivo PDF.

        Args:
            file_path: Caminho do arquivo.

        Yields:
            Documentos processados.
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processando PDF: {file_path.name} ({total_pages} paginas)")

                metadata_base = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "title": file_path.stem,
                    "total_pages": total_pages,
                }

                # Se pages_per_chunk definido, agrupa paginas
                if self.pages_per_chunk:
                    for start in range(0, total_pages, self.pages_per_chunk):
                        end = min(start + self.pages_per_chunk, total_pages)
                        pages_text = []

                        for i in range(start, end):
                            text = self._extract_text_from_page(pdf.pages[i])
                            if text.strip():
                                pages_text.append(f"[Pagina {i + 1}]\n{text}")

                        if pages_text:
                            yield Document(
                                content="\n\n".join(pages_text),
                                metadata={
                                    **metadata_base,
                                    "page_start": start + 1,
                                    "page_end": end,
                                },
                                source=str(file_path),
                                doc_type=self.doc_type,
                            )
                else:
                    # Todas as paginas em um documento
                    all_text = []
                    for i, page in enumerate(pdf.pages):
                        text = self._extract_text_from_page(page)
                        if text.strip():
                            all_text.append(f"[Pagina {i + 1}]\n{text}")

                    if all_text:
                        yield Document(
                            content="\n\n".join(all_text),
                            metadata=metadata_base,
                            source=str(file_path),
                            doc_type=self.doc_type,
                        )

        except Exception as e:
            logger.error(f"Erro ao processar PDF {file_path}: {e}")

    def load(self, source: Path | str) -> Iterator[Document]:
        """
        Carrega arquivos PDF de um diretorio ou arquivo.

        Args:
            source: Caminho do arquivo ou diretorio.

        Yields:
            Documentos carregados.
        """
        source_path = Path(source)

        if source_path.is_file():
            if source_path.suffix.lower() == ".pdf":
                yield from self._parse_pdf(source_path)
        elif source_path.is_dir():
            pdf_files = list(source_path.rglob("*.pdf"))
            logger.info(f"Encontrados {len(pdf_files)} arquivos PDF")

            for file_path in pdf_files:
                yield from self._parse_pdf(file_path)
        else:
            logger.warning(f"Fonte nao encontrada: {source_path}")
