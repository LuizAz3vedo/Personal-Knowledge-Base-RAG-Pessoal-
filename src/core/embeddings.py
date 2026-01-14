"""Servico de embeddings usando Ollama com cache."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import ollama

from src.core.config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache simples para embeddings usando LRU."""

    def __init__(self, maxsize: int = 1000) -> None:
        """
        Inicializa o cache de embeddings.

        Args:
            maxsize: Tamanho maximo do cache.
        """
        self._cache: dict[str, list[float]] = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Gera hash do texto para usar como chave."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """Busca embedding no cache."""
        key = self._hash_text(text)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """Armazena embedding no cache."""
        if len(self._cache) >= self._maxsize:
            # Remove o item mais antigo (FIFO simples)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._hash_text(text)
        self._cache[key] = embedding

    def clear(self) -> None:
        """Limpa o cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        """Retorna estatisticas do cache."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


class EmbeddingService:
    """Servico de embeddings usando Ollama com cache."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        cache_size: int = 1000,
        use_cache: bool = True,
    ) -> None:
        """
        Inicializa o servico de embeddings.

        Args:
            model: Nome do modelo de embeddings.
            base_url: URL do servidor Ollama.
            cache_size: Tamanho maximo do cache.
            use_cache: Se deve usar cache.
        """
        self.model = model or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self.use_cache = use_cache

        # Configurar cliente Ollama
        self._client = ollama.Client(host=self.base_url)

        # Inicializar cache
        self._cache = EmbeddingCache(maxsize=cache_size) if use_cache else None

        # Verificar se modelo esta disponivel
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Verifica se o modelo esta disponivel."""
        try:
            response = self._client.list()
            # API retorna objetos Model, nao dicionarios
            model_names = [m.model for m in response.models]

            # Verificar variacoes do nome do modelo
            available = any(
                self.model in name or name in self.model for name in model_names
            )

            if not available:
                logger.warning(
                    f"Modelo {self.model} nao encontrado. "
                    f"Disponiveis: {model_names}"
                )
                logger.info(f"Tentando baixar {self.model}...")
                self._client.pull(self.model)

        except Exception as e:
            logger.error(f"Erro ao verificar modelos: {e}")
            raise RuntimeError(
                f"Nao foi possivel conectar ao Ollama em {self.base_url}. "
                "Verifique se o servidor esta rodando."
            ) from e

    def embed_text(self, text: str) -> list[float]:
        """
        Gera embedding para um texto (com cache).

        Args:
            text: Texto para gerar embedding.

        Returns:
            Vetor de embedding.
        """
        # Verificar cache primeiro
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                return cached

        # Gerar embedding via Ollama
        response = self._client.embed(
            model=self.model,
            input=text,
        )
        embedding = list(response.embeddings[0])

        # Armazenar no cache
        if self._cache is not None:
            self._cache.set(text, embedding)

        return embedding

    @property
    def cache_stats(self) -> dict | None:
        """Retorna estatisticas do cache."""
        if self._cache is None:
            return None
        return self._cache.stats

    def clear_cache(self) -> None:
        """Limpa o cache de embeddings."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Cache de embeddings limpo")

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Gera embeddings para multiplos textos.

        Args:
            texts: Lista de textos.
            batch_size: Tamanho do batch.
            show_progress: Se deve mostrar progresso.

        Returns:
            Lista de vetores de embedding.
        """
        embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]

            for text in batch:
                embedding = self.embed_text(text)
                embeddings.append(embedding)

            if show_progress:
                progress = min(i + batch_size, total)
                logger.info(f"Embeddings: {progress}/{total}")

        return embeddings


# Instancia global (lazy initialization)
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Retorna a instancia global do servico de embeddings."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
