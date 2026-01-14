"""Sistema de retrieval semantico com busca hibrida."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.config import settings
from src.core.embeddings import get_embedding_service
from src.core.vectorstore import vector_store

if TYPE_CHECKING:
    from src.core.hybrid_search import BM25, HybridSearcher
    from src.core.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Resultado de uma busca."""

    content: str
    source: str
    score: float
    metadata: dict


class Retriever:
    """Sistema de retrieval semantico com suporte a re-ranking e busca hibrida."""

    def __init__(
        self,
        top_k: int | None = None,
        threshold: float | None = None,
        use_reranker: bool = False,
        rerank_top_k: int | None = None,
        use_hybrid: bool = False,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        """
        Inicializa o retriever.

        Args:
            top_k: Numero de resultados a retornar.
            threshold: Threshold minimo de similaridade.
            use_reranker: Se deve usar re-ranking.
            rerank_top_k: Numero de resultados apos re-ranking.
            use_hybrid: Se deve usar busca hibrida (semantica + BM25).
            semantic_weight: Peso da busca semantica na busca hibrida.
            bm25_weight: Peso do BM25 na busca hibrida.
        """
        self.top_k = top_k or settings.top_k_results
        self.threshold = threshold or settings.similarity_threshold
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k or self.top_k
        self.use_hybrid = use_hybrid
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._reranker: Reranker | None = None
        self._hybrid_searcher: HybridSearcher | None = None
        self._bm25_initialized: bool = False

    @property
    def reranker(self) -> Reranker:
        """Carrega o reranker de forma lazy."""
        if self._reranker is None:
            from src.core.reranker import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    @property
    def hybrid_searcher(self) -> HybridSearcher:
        """Carrega o buscador hibrido de forma lazy."""
        if self._hybrid_searcher is None:
            from src.core.hybrid_search import HybridSearcher

            self._hybrid_searcher = HybridSearcher(
                semantic_weight=self.semantic_weight,
                bm25_weight=self.bm25_weight,
            )
        return self._hybrid_searcher

    def _ensure_bm25_index(self) -> None:
        """Garante que o indice BM25 esta construido."""
        if not self._bm25_initialized:
            from src.core.hybrid_search import build_bm25_index_from_vectorstore

            build_bm25_index_from_vectorstore()
            self._bm25_initialized = True

    def retrieve(
        self,
        query: str,
        *,
        filter_doc_type: str | None = None,
        use_reranker: bool | None = None,
        use_hybrid: bool | None = None,
    ) -> list[RetrievalResult]:
        """
        Busca documentos relevantes.

        Args:
            query: Pergunta do usuario.
            filter_doc_type: Filtrar por tipo de documento.
            use_reranker: Override para usar ou nao re-ranking.
            use_hybrid: Override para usar ou nao busca hibrida.

        Returns:
            Lista de resultados ordenados por relevancia.
        """
        should_rerank = use_reranker if use_reranker is not None else self.use_reranker
        should_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        # Se vai usar reranker, buscar mais resultados inicialmente
        initial_top_k = self.top_k * 3 if should_rerank else self.top_k

        logger.info(f"Buscando documentos para: '{query[:50]}...'")

        # Gerar embedding da query
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.embed_text(query)

        # Construir filtros
        where_filter = None
        if filter_doc_type:
            where_filter = {"doc_type": filter_doc_type}

        # Buscar no vector store (busca semantica)
        results = vector_store.query(
            query_text=query,
            query_embedding=query_embedding,
            n_results=initial_top_k,
            where=where_filter,
        )

        # Processar resultados da busca semantica
        retrieval_results = []

        if results and results["ids"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                # Converter distancia para score de similaridade
                # ChromaDB usa distancia L2, menor = mais similar
                distance = distances[i] if distances else 0
                score = 1.0 / (1.0 + distance)

                # Sem reranker e sem hybrid, aplicar threshold aqui
                if not should_rerank and not should_hybrid and score < self.threshold:
                    continue

                result = RetrievalResult(
                    content=doc,
                    source=meta.get("source", "unknown"),
                    score=score,
                    metadata={**meta, "search_type": "semantic"},
                )
                retrieval_results.append(result)

        logger.info(f"Busca semantica: {len(retrieval_results)} resultados")

        # Aplicar busca hibrida se configurado
        if should_hybrid and retrieval_results:
            from src.core.hybrid_search import get_bm25_index

            self._ensure_bm25_index()
            bm25 = get_bm25_index()
            bm25_results = bm25.search(query, top_k=initial_top_k)

            logger.info(f"Busca BM25: {len(bm25_results)} resultados")

            # Combinar resultados
            retrieval_results = self.hybrid_searcher.combine_results(
                semantic_results=retrieval_results,
                bm25_results=bm25_results,
                top_k=self.top_k if not should_rerank else initial_top_k,
            )
            logger.info(f"Busca hibrida: {len(retrieval_results)} resultados")

        # Aplicar re-ranking se configurado
        if should_rerank and retrieval_results:
            retrieval_results = self.reranker.rerank(
                query=query,
                results=retrieval_results,
                top_k=self.rerank_top_k,
            )
            logger.info(f"Apos re-ranking: {len(retrieval_results)} resultados")

        return retrieval_results

    def build_context(
        self,
        results: list[RetrievalResult],
        *,
        max_length: int = 4000,
    ) -> str:
        """
        Constroi contexto a partir dos resultados.

        Args:
            results: Lista de resultados do retrieval.
            max_length: Tamanho maximo do contexto.

        Returns:
            Contexto formatado para o LLM.
        """
        if not results:
            return "Nenhuma informacao relevante encontrada nas notas."

        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            # Formatar cada resultado
            source = result.source
            score_pct = f"{result.score * 100:.1f}%"

            chunk = f"""
---
Fonte {i}: {source} (Relevancia: {score_pct})
---
{result.content}
"""
            chunk_length = len(chunk)

            if current_length + chunk_length > max_length:
                # Truncar se necessario
                remaining = max_length - current_length
                if remaining > 100:
                    chunk = chunk[:remaining] + "\n...[truncado]"
                    context_parts.append(chunk)
                break

            context_parts.append(chunk)
            current_length += chunk_length

        return "\n".join(context_parts)


# Instancia global (lazy initialization)
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Retorna a instancia global do retriever."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
