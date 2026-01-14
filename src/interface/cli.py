"""Interface de linha de comando."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

# Forcar UTF-8 no Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from src.core.config import settings
from src.core.embeddings import get_embedding_service
from src.core.llm import get_llm_service
from src.core.retriever import Retriever
from src.core.vectorstore import vector_store
from src.ingestors.markdown import MarkdownIngestor
from src.ingestors.pdf import PDFIngestor

app = typer.Typer(
    name="pkb",
    help="Personal Knowledge Base - CLI",
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    use_reranker: bool = typer.Option(
        False, "--rerank", "-r", help="Usar re-ranking"
    ),
) -> None:
    """Inicia modo de chat interativo."""
    console.print(
        Panel(
            "[bold blue]Personal Knowledge Base[/bold blue]\n"
            "Digite suas perguntas ou 'sair' para encerrar.",
            title="Chat",
        )
    )

    retriever = Retriever(use_reranker=use_reranker)
    llm_service = get_llm_service()

    while True:
        try:
            query = Prompt.ask("\n[bold green]Voce[/bold green]")

            if query.lower() in ("sair", "exit", "quit"):
                console.print("[yellow]Ate logo![/yellow]")
                break

            if not query.strip():
                continue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Buscando nas notas...", total=None)

                # Retrieval
                results = retriever.retrieve(query)
                context = retriever.build_context(results)

            # Gerar resposta
            console.print("\n[bold blue]Assistente[/bold blue]")

            full_response = ""
            for chunk in llm_service.generate_stream(query, context):
                console.print(chunk, end="")
                full_response += chunk

            console.print()

            # Mostrar fontes
            if results:
                console.print("\n[dim]Fontes:[/dim]")
                for i, result in enumerate(results[:3], 1):
                    console.print(
                        f"  [dim]{i}. {result.source} "
                        f"({result.score:.0%})[/dim]"
                    )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrompido.[/yellow]")
            break


@app.command()
def ingest(
    source: str = typer.Argument(
        ...,
        help="Caminho para o diretorio ou arquivo fonte",
    ),
    doc_type: str = typer.Option(
        "markdown",
        "--type",
        "-t",
        help="Tipo de documento: markdown, pdf",
    ),
) -> None:
    """Ingere documentos no vector store."""
    source_path = Path(source)

    if not source_path.exists():
        console.print(f"[red]Erro: '{source}' nao encontrado[/red]")
        raise typer.Exit(1)

    # Selecionar ingestor
    ingestors = {
        "markdown": MarkdownIngestor,
        "pdf": PDFIngestor,
    }

    if doc_type not in ingestors:
        console.print(f"[red]Tipo nao suportado: {doc_type}[/red]")
        console.print(f"Tipos disponiveis: {', '.join(ingestors.keys())}")
        raise typer.Exit(1)

    console.print(f"[blue]Ingerindo {doc_type} de: {source}[/blue]")

    ingestor_class = ingestors[doc_type]
    ingestor = ingestor_class()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processando documentos...", total=None)

        # Ingerir documentos (ja faz chunking)
        chunks = ingestor.ingest(source_path)

        if not chunks:
            console.print("[yellow]Nenhum documento encontrado.[/yellow]")
            raise typer.Exit(0)

        progress.update(task, description="Gerando embeddings...")

        # Gerar embeddings
        embedding_service = get_embedding_service()
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts, show_progress=False)

        progress.update(task, description="Salvando no vector store...")

        # Preparar metadados e IDs
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [
            f"{chunk.source}::chunk_{chunk.metadata.get('chunk_index', i)}"
            for i, chunk in enumerate(chunks)
        ]

        # Adicionar ao vector store
        vector_store.add_documents(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    console.print(
        f"[green]Concluido![/green]\n" f"   Chunks indexados: {len(chunks)}"
    )


@app.command()
def stats() -> None:
    """Mostra estatisticas do knowledge base."""
    stats_data = vector_store.get_stats()

    table = Table(title="Estatisticas")
    table.add_column("Metrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Collection", stats_data["collection_name"])
    table.add_row("Documentos indexados", str(stats_data["document_count"]))
    table.add_row("Diretorio", stats_data["persist_directory"])

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Texto para buscar"),
    limit: int = typer.Option(5, "--limit", "-n", help="Numero de resultados"),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Usar re-ranking"),
) -> None:
    """Busca semantica nas notas."""
    retriever = Retriever(top_k=limit, use_reranker=rerank)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Buscando...", total=None)
        results = retriever.retrieve(query)

    if not results:
        console.print("[yellow]Nenhum resultado encontrado.[/yellow]")
        return

    console.print(f"\n[bold]Resultados para: '{query}'[/bold]\n")

    for i, result in enumerate(results[:limit], 1):
        panel = Panel(
            result.content[:500] + ("..." if len(result.content) > 500 else ""),
            title=f"[bold]{i}. {result.source}[/bold]",
            subtitle=f"Score: {result.score:.2%}",
            border_style="blue" if result.score > 0.8 else "dim",
        )
        console.print(panel)


@app.command()
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Nao pedir confirmacao"),
) -> None:
    """Limpa todos os dados do vector store."""
    if not force:
        confirm = Prompt.ask(
            "[red]Isso vai apagar todos os dados. Continuar?[/red]",
            choices=["s", "n"],
            default="n",
        )
        if confirm != "s":
            console.print("[yellow]Cancelado.[/yellow]")
            raise typer.Exit()

    vector_store.reset()
    console.print("[green]Vector store resetado.[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta para o assistente"),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Usar re-ranking"),
) -> None:
    """Faz uma pergunta unica (modo nao-interativo)."""
    retriever = Retriever(use_reranker=rerank)
    llm_service = get_llm_service()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Processando...", total=None)

        results = retriever.retrieve(question)
        context = retriever.build_context(results)

    # Gerar resposta (sem streaming para modo nao-interativo)
    response = llm_service.generate(question, context)

    console.print(Panel(response, title="Resposta", border_style="green"))

    if results:
        console.print("\n[dim]Fontes:[/dim]")
        for i, result in enumerate(results[:3], 1):
            console.print(f"  [dim]{i}. {result.source} ({result.score:.0%})[/dim]")


if __name__ == "__main__":
    app()
