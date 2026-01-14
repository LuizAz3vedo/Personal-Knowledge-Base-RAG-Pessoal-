"""Ingestor para arquivos Markdown (Obsidian)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from src.ingestors.base import BaseIngestor, Document
from src.utils.chunking import MarkdownChunker

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class MarkdownIngestor(BaseIngestor):
    """Ingestor para arquivos Markdown com suporte a Obsidian."""

    doc_type: str = "markdown"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        extract_links: bool = True,
        extract_tags: bool = True,
        use_markdown_chunker: bool = True,
    ) -> None:
        """
        Inicializa o ingestor Markdown.

        Args:
            chunk_size: Tamanho dos chunks.
            chunk_overlap: Sobreposicao dos chunks.
            extract_links: Extrair wikilinks do Obsidian.
            extract_tags: Extrair tags (#tag).
            use_markdown_chunker: Usar chunker especializado para Markdown.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.extract_links = extract_links
        self.extract_tags = extract_tags
        self.use_markdown_chunker = use_markdown_chunker

        if use_markdown_chunker:
            self._markdown_chunker = MarkdownChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                respect_headers=True,
            )

    def _extract_wikilinks(self, content: str) -> list[str]:
        """Extrai wikilinks [[link]] do conteudo."""
        pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        return re.findall(pattern, content)

    def _extract_tags(self, content: str) -> list[str]:
        """Extrai tags #tag do conteudo."""
        pattern = r"(?:^|\s)#([a-zA-Z0-9_-]+)"
        return list(set(re.findall(pattern, content)))

    def _clean_content(self, content: str) -> str:
        """Remove elementos que nao agregam ao contexto."""
        # Remove blocos de codigo muito grandes (manter pequenos)
        content = re.sub(r"```[\s\S]{500,}?```", "[codigo omitido]", content)

        # Remove imagens embarcadas
        content = re.sub(r"!\[\[.*?\]\]", "", content)
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

        # Limpa espacos extras
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Converte valores nao suportados pelo ChromaDB para strings."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # ChromaDB nao suporta listas, converter para string
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                # Ignorar dicts aninhados
                continue
            elif isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                # Converter outros tipos para string
                sanitized[key] = str(value)
        return sanitized

    def _parse_file(self, file_path: Path) -> Document | None:
        """
        Faz parse de um arquivo Markdown.

        Args:
            file_path: Caminho do arquivo.

        Returns:
            Documento processado ou None se vazio/invalido.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                post = frontmatter.load(f)
        except Exception as e:
            logger.warning(f"Erro ao ler {file_path}: {e}")
            return None

        content = str(post.content)
        if not content.strip():
            return None

        # Metadados do frontmatter
        metadata: dict = dict(post.metadata) if post.metadata else {}

        # Extrair informacoes adicionais
        if self.extract_links:
            links = self._extract_wikilinks(content)
            if links:
                metadata["wikilinks"] = links

        if self.extract_tags:
            tags = self._extract_tags(content)
            # Combinar com tags do frontmatter
            existing_tags = metadata.get("tags", [])
            if isinstance(existing_tags, str):
                existing_tags = [existing_tags]
            all_tags = list(set(existing_tags + tags))
            if all_tags:
                metadata["tags"] = all_tags

        # Adicionar metadados do arquivo
        metadata["file_name"] = file_path.name
        metadata["file_path"] = str(file_path)

        # Extrair titulo (primeira linha H1 ou nome do arquivo)
        title_match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            metadata["title"] = file_path.stem

        # Limpar conteudo
        clean_content = self._clean_content(content)

        # Sanitizar metadados para ChromaDB
        sanitized_metadata = self._sanitize_metadata(metadata)

        return Document(
            content=clean_content,
            metadata=sanitized_metadata,
            source=str(file_path),
            doc_type=self.doc_type,
        )

    def chunk_document(self, document: Document) -> list[Document]:
        """
        Divide um documento Markdown em chunks respeitando headers.

        Args:
            document: Documento a ser dividido.

        Returns:
            Lista de documentos (chunks).
        """
        if not self.use_markdown_chunker:
            # Usar chunker padrao da classe base
            return super().chunk_document(document)

        if not document.content.strip():
            return []

        # Usar MarkdownChunker especializado
        chunks = self._markdown_chunker.chunk(document)

        # Converter Chunk para Document
        return [
            Document(
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                },
                source=document.source,
                doc_type=document.doc_type,
            )
            for chunk in chunks
        ]

    def load(self, source: Path | str) -> Iterator[Document]:
        """
        Carrega arquivos Markdown de um diretorio ou arquivo.

        Args:
            source: Caminho do arquivo ou diretorio.

        Yields:
            Documentos carregados.
        """
        source_path = Path(source)

        if source_path.is_file():
            if source_path.suffix.lower() == ".md":
                doc = self._parse_file(source_path)
                if doc:
                    yield doc
        elif source_path.is_dir():
            md_files = list(source_path.rglob("*.md"))
            logger.info(f"Encontrados {len(md_files)} arquivos Markdown")

            for file_path in md_files:
                # Ignorar arquivos de configuracao do Obsidian
                if ".obsidian" in str(file_path):
                    continue

                doc = self._parse_file(file_path)
                if doc:
                    yield doc
        else:
            logger.warning(f"Fonte nao encontrada: {source_path}")
