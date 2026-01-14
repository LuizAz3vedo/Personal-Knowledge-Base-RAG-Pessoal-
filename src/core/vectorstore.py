"""Gerenciamento do Vector Store com ChromaDB."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings

if TYPE_CHECKING:
    from chromadb import Collection
    from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Gerencia operacoes com o ChromaDB."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Inicializa o gerenciador do vector store.

        Args:
            persist_directory: Diretorio para persistencia.
            collection_name: Nome da collection.
        """
        self.persist_directory = persist_directory or settings.chroma_db_path
        self.collection_name = collection_name or settings.chroma_collection_name

        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    @property
    def client(self) -> ClientAPI:
        """Retorna o cliente ChromaDB (lazy loading)."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def collection(self) -> Collection:
        """Retorna a collection (lazy loading)."""
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def _create_client(self) -> ClientAPI:
        """Cria cliente ChromaDB com persistencia."""
        logger.info(f"Inicializando ChromaDB em: {self.persist_directory}")

        # Garantir que o diretorio existe
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        return chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    def _get_or_create_collection(self) -> Collection:
        """Obtem ou cria a collection."""
        logger.info(f"Obtendo collection: {self.collection_name}")

        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Personal Knowledge Base"},
        )

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """
        Adiciona documentos ao vector store.

        Args:
            documents: Lista de textos dos documentos.
            metadatas: Lista de metadados para cada documento.
            ids: Lista de IDs unicos (gerados automaticamente se nao fornecidos).
            embeddings: Embeddings pre-computados (opcional).
        """
        if not documents:
            logger.warning("Nenhum documento para adicionar")
            return

        # Gerar IDs se nao fornecidos
        if ids is None:
            ids = [
                hashlib.md5(doc.encode()).hexdigest()[:16] for doc in documents
            ]

        # Garantir metadatas
        if metadatas is None:
            metadatas = [{} for _ in documents]

        logger.info(f"Adicionando {len(documents)} documentos ao vector store")

        # Adicionar em batches para evitar problemas de memoria
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            add_kwargs: dict = {
                "documents": batch_docs,
                "metadatas": batch_meta,
                "ids": batch_ids,
            }

            if embeddings is not None:
                add_kwargs["embeddings"] = embeddings[i : i + batch_size]

            self.collection.add(**add_kwargs)

        logger.info("Documentos adicionados com sucesso")

    def query(
        self,
        query_text: str,
        n_results: int | None = None,
        where: dict | None = None,
        query_embedding: list[float] | None = None,
    ) -> dict:
        """
        Busca documentos similares.

        Args:
            query_text: Texto da consulta.
            n_results: Numero de resultados.
            where: Filtros de metadados.
            query_embedding: Embedding pre-computado da query.

        Returns:
            Dicionario com resultados da busca.
        """
        n_results = n_results or settings.top_k_results

        query_kwargs: dict = {
            "n_results": n_results,
        }

        if query_embedding is not None:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query_text]

        if where is not None:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        logger.debug(f"Query retornou {len(results['ids'][0])} resultados")

        return results

    def delete_documents(self, ids: list[str]) -> None:
        """Remove documentos pelo ID."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Removidos {len(ids)} documentos")

    def get_stats(self) -> dict:
        """Retorna estatisticas do vector store."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(self.persist_directory),
        }

    def reset(self) -> None:
        """Remove todos os documentos da collection."""
        logger.warning("Resetando collection...")
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("Collection resetada")


# Instancia global
vector_store = VectorStoreManager()
