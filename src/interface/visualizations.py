"""Visualizacoes para o RAG."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from src.core.retriever import RetrievalResult

logger = logging.getLogger(__name__)


def _get_short_label(result: "RetrievalResult", max_len: int = 25) -> str:
    """Extrai um label curto e legivel do resultado."""
    # Tentar usar titulo dos metadados primeiro
    title = result.metadata.get("title", "")
    if title and len(title) <= max_len:
        return title
    if title:
        return title[:max_len - 3] + "..."

    # Fallback para nome do arquivo (compativel com Windows e Linux)
    filename = os.path.basename(result.source)
    # Remover extensao
    name = os.path.splitext(filename)[0]
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def create_rag_flow_diagram(
    query: str,
    results: list[RetrievalResult],
    response: str,
    used_reranker: bool = False,
) -> go.Figure:
    """
    Cria diagrama do fluxo RAG mostrando Query -> Chunks -> Resposta.

    Args:
        query: Pergunta do usuario.
        results: Resultados do retrieval.
        response: Resposta gerada.
        used_reranker: Se o re-ranker foi usado.

    Returns:
        Figura Plotly com o diagrama.
    """
    num_chunks = len(results)

    # Definir posicoes X baseado em se tem re-ranking ou nao
    # Com re-ranking: Query(0) -> Retrieval(0.33) -> Re-rank(0.66) -> Response(1)
    # Sem re-ranking: Query(0) -> Retrieval(0.5) -> Response(1)
    if used_reranker:
        x_query = 0.0
        x_retrieval = 0.33
        x_rerank = 0.66
        x_response = 1.0
    else:
        x_query = 0.0
        x_retrieval = 0.5
        x_rerank = None
        x_response = 1.0

    # Nodes
    node_x: list[float] = [x_query]
    node_y: list[float] = [0.5]
    node_labels = [f"Query:\n{query[:25]}..."]
    node_colors = ["#FF6B6B"]  # Vermelho para query

    # Chunks no meio (retrieval)
    chunk_indices = []  # Para rastrear indices dos chunks
    if num_chunks > 0:
        chunk_spacing = 1.0 / (num_chunks + 1)
        for i, result in enumerate(results):
            node_x.append(x_retrieval)
            node_y.append((i + 1) * chunk_spacing)
            label = _get_short_label(result, max_len=18)
            node_labels.append(f"{label}\n{result.score:.0%}")
            chunk_indices.append(len(node_x) - 1)
            # Cor baseada no score
            if result.score > 0.8:
                node_colors.append("#4ECDC4")  # Verde-azulado
            elif result.score > 0.5:
                node_colors.append("#45B7D1")  # Azul
            else:
                node_colors.append("#96CEB4")  # Verde claro

    # Re-ranker node (se usado)
    rerank_idx = None
    if used_reranker and x_rerank is not None:
        node_x.append(x_rerank)
        node_y.append(0.5)
        node_labels.append("Re-Ranker\n(Cross-Encoder)")
        node_colors.append("#FFD93D")  # Amarelo para re-ranker
        rerank_idx = len(node_x) - 1

    # Resposta
    node_x.append(x_response)
    node_y.append(0.5)
    node_labels.append(f"Resposta:\n{response[:25]}...")
    node_colors.append("#95E1A3")  # Verde para resposta
    resp_idx = len(node_x) - 1

    # Criar edges (links)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    # Query -> Chunks
    for idx in chunk_indices:
        edge_x.extend([node_x[0], node_x[idx], None])
        edge_y.extend([node_y[0], node_y[idx], None])

    if used_reranker and rerank_idx is not None:
        # Chunks -> Re-ranker
        for idx in chunk_indices:
            edge_x.extend([node_x[idx], node_x[rerank_idx], None])
            edge_y.extend([node_y[idx], node_y[rerank_idx], None])
        # Re-ranker -> Resposta
        edge_x.extend([node_x[rerank_idx], node_x[resp_idx], None])
        edge_y.extend([node_y[rerank_idx], node_y[resp_idx], None])
    else:
        # Chunks -> Resposta (sem re-ranking)
        for idx in chunk_indices:
            edge_x.extend([node_x[idx], node_x[resp_idx], None])
            edge_y.extend([node_y[idx], node_y[resp_idx], None])

    # Criar figura
    fig = go.Figure()

    # Adicionar edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 2, "color": "#888"},
            hoverinfo="none",
        )
    )

    # Adicionar nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={
                "size": 40,
                "color": node_colors,
                "line": {"width": 2, "color": "#333"},
            },
            text=node_labels,
            textposition="bottom center",
            hoverinfo="text",
        )
    )

    # Titulo dinamico
    if used_reranker:
        title = "Fluxo RAG: Query → Retrieval → Re-Ranking → Geracao"
    else:
        title = "Fluxo RAG: Query → Retrieval → Geracao"

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "range": [-0.1, 1.1],
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "range": [-0.15, 1.1],
        },
        height=450,
        margin={"l": 20, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Anotacoes de fase
    fig.add_annotation(
        x=x_query, y=-0.1, text="QUERY", showarrow=False, font={"size": 11}
    )
    fig.add_annotation(
        x=x_retrieval, y=-0.1, text="RETRIEVAL", showarrow=False, font={"size": 11}
    )
    if used_reranker and x_rerank is not None:
        fig.add_annotation(
            x=x_rerank, y=-0.1, text="RE-RANKING", showarrow=False, font={"size": 11}
        )
    fig.add_annotation(
        x=x_response, y=-0.1, text="GENERATION", showarrow=False, font={"size": 11}
    )

    return fig


def create_embeddings_visualization(
    embeddings: list[list[float]],
    labels: list[str],
    doc_types: list[str],
    query_embedding: list[float] | None = None,
    query_label: str = "Query",
) -> go.Figure:
    """
    Cria visualizacao 2D dos embeddings usando UMAP.

    Args:
        embeddings: Lista de vetores de embedding.
        labels: Labels para cada ponto.
        doc_types: Tipo de documento para colorir.
        query_embedding: Embedding da query (opcional).
        query_label: Label da query.

    Returns:
        Figura Plotly com scatter plot.
    """
    from umap import UMAP

    if len(embeddings) < 2:
        # Nao da pra fazer UMAP com menos de 2 pontos
        fig = go.Figure()
        fig.add_annotation(
            text="Precisa de pelo menos 2 documentos para visualizar",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Preparar dados
    all_embeddings = embeddings.copy()
    all_labels = labels.copy()
    all_types = doc_types.copy()

    if query_embedding is not None:
        all_embeddings.append(query_embedding)
        all_labels.append(query_label)
        all_types.append("query")

    # Converter para numpy
    X = np.array(all_embeddings)

    # Aplicar UMAP
    n_neighbors = min(15, len(X) - 1)
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)

    try:
        X_2d = reducer.fit_transform(X)
    except Exception as e:
        logger.warning(f"Erro ao aplicar UMAP: {e}")
        # Fallback para PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

    # Criar dataframe para plotly
    import pandas as pd

    df = pd.DataFrame(
        {
            "x": X_2d[:, 0],
            "y": X_2d[:, 1],
            "label": all_labels,
            "type": all_types,
        }
    )

    # Criar figura
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="type",
        hover_data=["label"],
        title="Espaco de Embeddings (UMAP 2D)",
        color_discrete_map={
            "markdown": "#4ECDC4",
            "pdf": "#FF6B6B",
            "web": "#45B7D1",
            "query": "#FFE66D",
        },
    )

    # Destacar query se presente
    if query_embedding is not None:
        query_idx = len(embeddings)
        fig.add_trace(
            go.Scatter(
                x=[X_2d[query_idx, 0]],
                y=[X_2d[query_idx, 1]],
                mode="markers",
                marker={
                    "size": 20,
                    "color": "#FFE66D",
                    "symbol": "star",
                    "line": {"width": 2, "color": "#333"},
                },
                name="Query",
                hovertext=query_label,
            )
        )

    fig.update_layout(
        height=500,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Tipo",
    )

    return fig


def create_scores_chart(results: list[RetrievalResult]) -> go.Figure:
    """
    Cria grafico de barras com scores dos resultados.

    Args:
        results: Lista de resultados do retrieval.

    Returns:
        Figura Plotly com barras.
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="Nenhum resultado para visualizar",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Preparar dados
    sources = [_get_short_label(r, max_len=25) for r in results]
    scores = [r.score for r in results]
    doc_types = [r.metadata.get("doc_type", "unknown") for r in results]

    # Cores por tipo
    colors = []
    for dt in doc_types:
        if dt == "markdown":
            colors.append("#4ECDC4")
        elif dt == "pdf":
            colors.append("#FF6B6B")
        elif dt == "web":
            colors.append("#45B7D1")
        else:
            colors.append("#96CEB4")

    fig = go.Figure(
        data=[
            go.Bar(
                x=sources,
                y=scores,
                marker_color=colors,
                text=[f"{s:.0%}" for s in scores],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Scores de Relevancia",
        xaxis_title="Documento",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        height=300,
    )

    return fig


def create_retrieval_metrics_dashboard(
    results: list[RetrievalResult],
    query_time_ms: float | None = None,
    rerank_time_ms: float | None = None,
) -> go.Figure:
    """
    Cria dashboard com metricas do retrieval.

    Args:
        results: Lista de resultados.
        query_time_ms: Tempo de busca em ms.
        rerank_time_ms: Tempo de re-ranking em ms.

    Returns:
        Figura Plotly com metricas.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Resultados", "Score Medio", "Distribuicao"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "pie"}]],
    )

    # Numero de resultados
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=len(results),
            title={"text": "Chunks"},
            domain={"row": 0, "column": 0},
        ),
        row=1,
        col=1,
    )

    # Score medio
    avg_score = np.mean([r.score for r in results]) if results else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=avg_score * 100,
            number={"suffix": "%"},
            title={"text": "Score Medio"},
        ),
        row=1,
        col=2,
    )

    # Distribuicao por tipo
    if results:
        type_counts: dict[str, int] = {}
        for r in results:
            dt = r.metadata.get("doc_type", "unknown")
            type_counts[dt] = type_counts.get(dt, 0) + 1

        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                marker_colors=["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4"],
            ),
            row=1,
            col=3,
        )

    fig.update_layout(height=200, showlegend=False)

    return fig
