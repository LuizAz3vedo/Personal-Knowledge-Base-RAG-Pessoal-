"""Sistema de citacoes e referencias."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.retriever import RetrievalResult


@dataclass
class Citation:
    """Representa uma citacao."""

    source: str
    relevance_score: float
    excerpt: str = ""
    url: str | None = None
    page: int | None = None
    doc_type: str = "unknown"

    def format_markdown(self) -> str:
        """Formata citacao como Markdown."""
        parts = [f"**{self.source}**"]

        if self.page:
            parts.append(f" (pagina {self.page})")

        if self.relevance_score:
            parts.append(f" - Relevancia: {self.relevance_score:.0%}")

        if self.url:
            parts.append(f"\n  [Link]({self.url})")

        return "".join(parts)

    def format_obsidian_link(self, vault_path: Path | None = None) -> str:
        """Formata como link do Obsidian."""
        # Remover extensao para link
        note_name = Path(self.source).stem
        return f"[[{note_name}]]"


@dataclass
class CitedResponse:
    """Resposta com citacoes."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    total_sources: int = 0

    def format_full_response(self) -> str:
        """Formata resposta completa com citacoes."""
        output = [self.answer]

        if self.citations:
            output.append("\n\n---")
            output.append("### Fontes consultadas\n")

            for i, citation in enumerate(self.citations, 1):
                output.append(f"{i}. {citation.format_markdown()}")

        return "\n".join(output)

    def get_obsidian_links(self) -> list[str]:
        """Retorna links no formato Obsidian."""
        return [
            c.format_obsidian_link() for c in self.citations if c.doc_type == "markdown"
        ]


class CitationExtractor:
    """Extrai e formata citacoes dos resultados."""

    def __init__(self, max_excerpt_length: int = 200) -> None:
        """
        Inicializa o extrator de citacoes.

        Args:
            max_excerpt_length: Tamanho maximo do trecho citado.
        """
        self.max_excerpt_length = max_excerpt_length

    def extract_citations(
        self,
        retrieval_results: list[RetrievalResult],
        *,
        min_score: float = 0.0,
    ) -> list[Citation]:
        """
        Extrai citacoes dos resultados de retrieval.

        Args:
            retrieval_results: Lista de RetrievalResult.
            min_score: Score minimo para incluir citacao.

        Returns:
            Lista de citacoes formatadas.
        """
        citations = []
        seen_sources: set[str] = set()

        for result in retrieval_results:
            if result.score < min_score:
                continue

            # Evitar duplicatas da mesma fonte
            if result.source in seen_sources:
                continue
            seen_sources.add(result.source)

            # Extrair trecho relevante
            excerpt = self._extract_excerpt(result.content)

            # Determinar URL se for web
            url = None
            doc_type = result.metadata.get("doc_type", "unknown")
            if doc_type == "web":
                url = result.metadata.get("url", result.source)

            # Extrair pagina se for PDF
            page = None
            if doc_type == "pdf":
                page_match = re.search(r"\[Pagina (\d+)\]", result.content)
                if page_match:
                    page = int(page_match.group(1))

            citation = Citation(
                source=result.source,
                relevance_score=result.score,
                excerpt=excerpt,
                url=url,
                page=page,
                doc_type=doc_type,
            )
            citations.append(citation)

        return citations

    def _extract_excerpt(self, content: str) -> str:
        """Extrai um trecho representativo do conteudo."""
        # Pegar primeiro paragrafo significativo
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Ignorar paragrafos muito curtos
                if len(para) <= self.max_excerpt_length:
                    return para
                return para[: self.max_excerpt_length] + "..."

        # Fallback: primeiros caracteres
        if len(content) <= self.max_excerpt_length:
            return content
        return content[: self.max_excerpt_length] + "..."

    def create_cited_response(
        self,
        answer: str,
        retrieval_results: list[RetrievalResult],
    ) -> CitedResponse:
        """
        Cria resposta citada a partir dos resultados.

        Args:
            answer: Resposta gerada pelo LLM.
            retrieval_results: Resultados do retrieval.

        Returns:
            Resposta com citacoes.
        """
        citations = self.extract_citations(retrieval_results)

        return CitedResponse(
            answer=answer,
            citations=citations,
            total_sources=len(retrieval_results),
        )


# Instancia global
citation_extractor = CitationExtractor()


def get_citation_extractor() -> CitationExtractor:
    """Retorna a instancia global do extrator de citacoes."""
    return citation_extractor
