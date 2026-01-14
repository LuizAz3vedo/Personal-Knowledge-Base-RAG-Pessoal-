"""Integracao com LLM usando Ollama."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ollama

from src.core.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class LLMService:
    """Servico de LLM usando Ollama."""

    DEFAULT_SYSTEM_PROMPT = """Voce e um assistente pessoal inteligente que ajuda o usuario a encontrar informacoes em suas proprias notas e documentos.

Instrucoes:
1. Responda APENAS com base no contexto fornecido
2. Se a informacao nao estiver no contexto, diga que nao encontrou nas notas
3. Cite as fontes quando possivel (nome do arquivo/documento)
4. Seja conciso e direto
5. Use formatacao Markdown quando apropriado

Contexto das notas do usuario:
{context}
"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Inicializa o servico LLM.

        Args:
            model: Nome do modelo LLM.
            base_url: URL do servidor Ollama.
        """
        self.model = model or settings.ollama_llm_model
        self.base_url = base_url or settings.ollama_base_url

        self._client = ollama.Client(host=self.base_url)
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Verifica se o modelo esta disponivel."""
        try:
            response = self._client.list()
            model_names = [m.model for m in response.models]

            available = any(
                self.model in name or name in self.model for name in model_names
            )

            if not available:
                logger.warning(f"Modelo {self.model} nao encontrado")
                logger.info(f"Tentando baixar {self.model}...")
                self._client.pull(self.model)

        except Exception as e:
            logger.error(f"Erro ao conectar com Ollama: {e}")
            raise RuntimeError(
                f"Nao foi possivel conectar ao Ollama em {self.base_url}"
            ) from e

    def generate(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Gera resposta do LLM.

        Args:
            prompt: Prompt do usuario.
            context: Contexto recuperado das notas.
            system_prompt: System prompt customizado.
            temperature: Temperatura para geracao.
            max_tokens: Maximo de tokens na resposta.

        Returns:
            Resposta gerada.
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT.format(context=context)
        elif context:
            system_prompt = f"{system_prompt}\n\nContexto:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return response.message.content

    def generate_stream(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str | None = None,
        *,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """
        Gera resposta em streaming.

        Args:
            prompt: Prompt do usuario.
            context: Contexto recuperado.
            system_prompt: System prompt customizado.
            temperature: Temperatura para geracao.

        Yields:
            Tokens da resposta.
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT.format(context=context)
        elif context:
            system_prompt = f"{system_prompt}\n\nContexto:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        stream = self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": temperature},
        )

        for chunk in stream:
            if chunk.message and chunk.message.content:
                yield chunk.message.content


# Instancia global (lazy initialization)
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Retorna a instancia global do servico LLM."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
