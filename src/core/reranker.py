"""Sistema de re-ranking usando Cross-Encoder."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from src.core.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Modelos de cross-encoder recomendados (do menor para maior)
# - ms-marco-MiniLM-L-6-v2: rapido, bom para uso geral (~80MB)
# - ms-marco-MiniLM-L-12-v2: equilibrado (~120MB)
# - ms-marco-TinyBERT-L-2-v2: muito rapido, menor precisao (~60MB)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-ranker usando Cross-Encoder para melhorar relevancia."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
    ) -> None:
        """
        Inicializa o reranker.

        Args:
            model_name: Nome do modelo cross-encoder.
            device: Dispositivo ('cpu', 'cuda', ou None para auto).
        """
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._device = device

    @property
    def model(self) -> CrossEncoder:
        """Carrega o modelo de forma lazy."""
        if self._model is None:
            logger.info(f"Carregando modelo de re-ranking: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self._device,
            )
            logger.info("Modelo de re-ranking carregado")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Re-rankeia resultados usando cross-encoder.

        O cross-encoder avalia cada par (query, documento) de forma mais
        precisa que a busca por similaridade de embeddings, pois considera
        a interacao entre query e documento.

        Args:
            query: Pergunta do usuario.
            results: Resultados da busca inicial.
            top_k: Numero de resultados a retornar apos re-ranking.
                   Se None, retorna todos re-rankeados.

        Returns:
            Lista de resultados reordenados por relevancia.
        """
        if not results:
            return []

        if len(results) == 1:
            return results

        logger.info(f"Re-ranking {len(results)} resultados...")

        # Criar pares (query, documento) para o cross-encoder
        pairs = [(query, result.content) for result in results]

        # Obter scores do cross-encoder
        scores = self.model.predict(pairs)

        # Atualizar scores nos resultados
        for result, score in zip(results, scores):
            result.score = float(score)

        # Ordenar por score (maior = mais relevante)
        reranked = sorted(results, key=lambda x: x.score, reverse=True)

        # Limitar se top_k especificado
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(
            f"Re-ranking concluido. Top score: {reranked[0].score:.3f}"
            if reranked
            else "Re-ranking concluido. Sem resultados."
        )

        return reranked

    def score_pair(self, query: str, document: str) -> float:
        """
        Calcula score de relevancia para um par query-documento.

        Args:
            query: Pergunta.
            document: Texto do documento.

        Returns:
            Score de relevancia (quanto maior, mais relevante).
        """
        return float(self.model.predict([(query, document)])[0])


# Instancia global (lazy initialization)
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Retorna a instancia global do reranker."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
