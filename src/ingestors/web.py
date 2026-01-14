"""Ingestor para conteudo web (bookmarks e URLs)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup

from src.ingestors.base import BaseIngestor, Document

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class WebIngestor(BaseIngestor):
    """Ingestor para conteudo web e bookmarks."""

    doc_type: str = "web"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        include_comments: bool = False,
        favor_precision: bool = True,
    ) -> None:
        """
        Inicializa o ingestor web.

        Args:
            chunk_size: Tamanho dos chunks.
            chunk_overlap: Sobreposicao dos chunks.
            include_comments: Incluir comentarios das paginas.
            favor_precision: Priorizar precisao sobre recall na extracao.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.include_comments = include_comments
        self.favor_precision = favor_precision

    def _fetch_url(self, url: str) -> Document | None:
        """
        Busca e extrai conteudo de uma URL.

        Args:
            url: URL para buscar.

        Returns:
            Documento com o conteudo ou None se falhar.
        """
        try:
            # Usar trafilatura para extracao limpa
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.warning(f"Falha ao baixar: {url}")
                return None

            # Extrair texto principal
            text = trafilatura.extract(
                downloaded,
                include_comments=self.include_comments,
                favor_precision=self.favor_precision,
                include_tables=True,
                include_links=False,
                output_format="txt",
            )

            if not text or not text.strip():
                logger.warning(f"Conteudo vazio: {url}")
                return None

            # Extrair metadados
            metadata_raw = trafilatura.extract_metadata(downloaded)
            metadata: dict[str, Any] = {}

            if metadata_raw:
                if metadata_raw.title:
                    metadata["title"] = metadata_raw.title
                if metadata_raw.author:
                    metadata["author"] = metadata_raw.author
                if metadata_raw.date:
                    metadata["date"] = metadata_raw.date
                if metadata_raw.description:
                    metadata["description"] = metadata_raw.description

            # Informacoes da URL
            parsed = urlparse(url)
            metadata["url"] = url
            metadata["domain"] = parsed.netloc

            if "title" not in metadata:
                metadata["title"] = parsed.path.split("/")[-1] or parsed.netloc

            return Document(
                content=text,
                metadata=metadata,
                source=url,
                doc_type=self.doc_type,
            )

        except Exception as e:
            logger.error(f"Erro ao processar URL {url}: {e}")
            return None

    def _parse_bookmarks_html(self, file_path: Path) -> list[dict]:
        """Parse arquivo HTML de bookmarks (Chrome/Firefox)."""
        bookmarks: list[dict] = []

        try:
            with open(file_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            for link in soup.find_all("a"):
                href = link.get("href", "")
                if href.startswith(("http://", "https://")):
                    bookmarks.append({
                        "url": href,
                        "title": link.get_text(strip=True),
                        "add_date": link.get("add_date"),
                        "tags": link.get("tags", "").split(",") if link.get("tags") else [],
                    })

        except Exception as e:
            logger.error(f"Erro ao parsear bookmarks HTML: {e}")

        return bookmarks

    def _parse_bookmarks_json(self, file_path: Path) -> list[dict]:
        """Parse arquivo JSON de bookmarks."""
        bookmarks: list[dict] = []

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Formato pode variar - tentar extrair URLs
            def extract_urls(obj: Any, path: str = "") -> None:
                if isinstance(obj, dict):
                    if "url" in obj:
                        bookmarks.append({
                            "url": obj["url"],
                            "title": obj.get("title", obj.get("name", "")),
                            "tags": obj.get("tags", []),
                        })
                    for key, value in obj.items():
                        extract_urls(value, f"{path}/{key}")
                elif isinstance(obj, list):
                    for item in obj:
                        extract_urls(item, path)

            extract_urls(data)

        except Exception as e:
            logger.error(f"Erro ao parsear bookmarks JSON: {e}")

        return bookmarks

    def _load_from_bookmarks_file(self, file_path: Path) -> Iterator[Document]:
        """Carrega conteudo a partir de arquivo de bookmarks."""
        suffix = file_path.suffix.lower()

        if suffix == ".html":
            bookmarks = self._parse_bookmarks_html(file_path)
        elif suffix == ".json":
            bookmarks = self._parse_bookmarks_json(file_path)
        else:
            logger.warning(f"Formato de bookmarks nao suportado: {suffix}")
            return

        logger.info(f"Encontrados {len(bookmarks)} bookmarks em {file_path.name}")

        for bookmark in bookmarks:
            url = bookmark.get("url", "")
            if not url:
                continue

            doc = self._fetch_url(url)
            if doc:
                # Adicionar metadados do bookmark
                if bookmark.get("title"):
                    doc.metadata.setdefault("bookmark_title", bookmark["title"])
                if bookmark.get("tags"):
                    doc.metadata["tags"] = bookmark["tags"]

                yield doc

    def load(self, source: Path | str) -> Iterator[Document]:
        """
        Carrega conteudo web.

        Args:
            source: URL, arquivo de bookmarks, ou diretorio com bookmarks.

        Yields:
            Documentos carregados.
        """
        source_str = str(source)

        # Se for URL direta
        if source_str.startswith(("http://", "https://")):
            doc = self._fetch_url(source_str)
            if doc:
                yield doc
            return

        # Se for arquivo ou diretorio
        source_path = Path(source)

        if source_path.is_file():
            yield from self._load_from_bookmarks_file(source_path)
        elif source_path.is_dir():
            # Procurar arquivos de bookmarks
            for pattern in ["*.html", "*.json"]:
                for file_path in source_path.glob(pattern):
                    yield from self._load_from_bookmarks_file(file_path)
        else:
            logger.warning(f"Fonte nao encontrada: {source_path}")
