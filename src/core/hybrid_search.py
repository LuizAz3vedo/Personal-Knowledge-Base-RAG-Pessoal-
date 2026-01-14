"""Busca hibrida combinando BM25 e busca semantica."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Resultado de busca BM25."""

    content: str
    source: str
    score: float
    metadata: dict


class BM25:
    """Implementacao de BM25 para busca por keywords."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Inicializa o BM25.

        Args:
            k1: Parametro de saturacao de termo.
            b: Parametro de normalizacao por tamanho.
        """
        self.k1 = k1
        self.b = b
        self._documents: list[dict] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_term_freqs: list[dict[str, int]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Tokeniza texto em palavras."""
        # Converter para minusculas e remover pontuacao
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        # Dividir em tokens e filtrar vazios
        tokens = [t.strip() for t in text.split() if t.strip()]
        return tokens

    def fit(self, documents: list[dict]) -> None:
        """
        Indexa documentos para busca.

        Args:
            documents: Lista de dicts com 'content', 'source', 'metadata'.
        """
        self._documents = documents
        self._doc_lengths = []
        self._doc_term_freqs = []
        self._doc_freqs = Counter()

        # Processar cada documento
        for doc in documents:
            tokens = self._tokenize(doc["content"])
            self._doc_lengths.append(len(tokens))

            # Frequencia de termos no documento
            term_freqs = Counter(tokens)
            self._doc_term_freqs.append(dict(term_freqs))

            # Frequencia de documentos por termo
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] += 1

        # Calcular media de tamanho
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 0

        # Calcular IDF para cada termo
        n_docs = len(documents)
        for term, df in self._doc_freqs.items():
            # IDF com suavizacao
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        logger.info(f"BM25: Indexados {n_docs} documentos")

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """
        Busca documentos usando BM25.

        Args:
            query: Query de busca.
            top_k: Numero de resultados.

        Returns:
            Lista de resultados ordenados por score.
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []

        for i, doc in enumerate(self._documents):
            score = 0.0
            doc_length = self._doc_lengths[i]
            term_freqs = self._doc_term_freqs[i]

            for term in query_tokens:
                if term not in term_freqs:
                    continue

                tf = term_freqs[term]
                idf = self._idf.get(term, 0)

                # Formula BM25
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / self._avg_doc_length
                )
                score += idf * (numerator / denominator)

            scores.append((i, score))

        # Ordenar por score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Retornar top_k
        results = []
        for i, score in scores[:top_k]:
            if score > 0:
                doc = self._documents[i]
                results.append(
                    BM25Result(
                        content=doc["content"],
                        source=doc["source"],
                        score=score,
                        metadata=doc["metadata"],
                    )
                )

        return results


class HybridSearcher:
    """Combina busca semantica com BM25."""

    def __init__(
        self,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        """
        Inicializa o buscador hibrido.

        Args:
            semantic_weight: Peso da busca semantica (0-1).
            bm25_weight: Peso do BM25 (0-1).
        """
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._bm25: BM25 | None = None

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normaliza scores para [0, 1]."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def combine_results(
        self,
        semantic_results: list,
        bm25_results: list[BM25Result],
        top_k: int = 10,
    ) -> list:
        """
        Combina resultados de busca semantica e BM25.

        Args:
            semantic_results: Resultados da busca semantica (RetrievalResult).
            bm25_results: Resultados do BM25.
            top_k: Numero de resultados finais.

        Returns:
            Lista combinada ordenada por score hibrido.
        """
        # Criar mapas por source
        semantic_map: dict[str, tuple[float, object]] = {}
        for r in semantic_results:
            key = f"{r.source}:{r.content[:50]}"
            semantic_map[key] = (r.score, r)

        bm25_map: dict[str, float] = {}
        for r in bm25_results:
            key = f"{r.source}:{r.content[:50]}"
            bm25_map[key] = r.score

        # Normalizar scores do BM25
        if bm25_map:
            bm25_scores = list(bm25_map.values())
            normalized = self._normalize_scores(bm25_scores)
            keys = list(bm25_map.keys())
            bm25_map = dict(zip(keys, normalized))

        # Combinar scores
        combined: dict[str, tuple[float, object]] = {}

        for key, (sem_score, result) in semantic_map.items():
            bm25_score = bm25_map.get(key, 0.0)
            hybrid_score = (
                self.semantic_weight * sem_score + self.bm25_weight * bm25_score
            )
            combined[key] = (hybrid_score, result)

        # Adicionar resultados que so aparecem no BM25
        for key, bm25_score in bm25_map.items():
            if key not in combined:
                # Criar resultado a partir do BM25
                for r in bm25_results:
                    if f"{r.source}:{r.content[:50]}" == key:
                        from src.core.retriever import RetrievalResult

                        result = RetrievalResult(
                            content=r.content,
                            source=r.source,
                            score=self.bm25_weight * bm25_score,
                            metadata=r.metadata,
                        )
                        combined[key] = (self.bm25_weight * bm25_score, result)
                        break

        # Ordenar e retornar
        sorted_results = sorted(combined.values(), key=lambda x: x[0], reverse=True)

        # Atualizar scores nos resultados
        final_results = []
        for score, result in sorted_results[:top_k]:
            # Criar novo resultado com score hibrido
            from src.core.retriever import RetrievalResult

            final_results.append(
                RetrievalResult(
                    content=result.content,
                    source=result.source,
                    score=score,
                    metadata={**result.metadata, "search_type": "hybrid"},
                )
            )

        return final_results


# Instancia global do BM25
_bm25_index: BM25 | None = None


def get_bm25_index() -> BM25:
    """Retorna a instancia global do BM25."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25()
    return _bm25_index


def build_bm25_index_from_vectorstore() -> BM25:
    """Constroi indice BM25 a partir do vector store."""
    from src.core.vectorstore import vector_store

    bm25 = get_bm25_index()

    # Obter todos os documentos do ChromaDB
    collection = vector_store.collection
    results = collection.get(include=["documents", "metadatas"])

    if results["ids"]:
        documents = []
        for doc_id, content, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            documents.append(
                {
                    "id": doc_id,
                    "content": content,
                    "source": metadata.get("source", "unknown"),
                    "metadata": metadata,
                }
            )

        bm25.fit(documents)
        logger.info(f"Indice BM25 construido com {len(documents)} documentos")

    return bm25
