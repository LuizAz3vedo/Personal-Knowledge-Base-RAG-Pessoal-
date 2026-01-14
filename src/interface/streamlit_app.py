"""Interface web com Streamlit - Design moderno."""

from __future__ import annotations

import os

import streamlit as st

from src.core.config import settings
from src.core.embeddings import get_embedding_service
from src.core.llm import get_llm_service
from src.core.retriever import Retriever
from src.core.vectorstore import vector_store
from src.interface.visualizations import (
    create_embeddings_visualization,
    create_rag_flow_diagram,
    create_scores_chart,
)
from src.utils.citations import get_citation_extractor

# Configuração da página
st.set_page_config(
    page_title="Personal Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para design moderno
CUSTOM_CSS = """
<style>
    /* Importar fonte */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Aplicar fonte global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a2332 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }

    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.4);
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        background: #e8ecf1;
        color: #1e3a5f !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: #d0d7e0;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%) !important;
        color: white !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d5a87, transparent);
        margin: 1.5rem 0;
    }

    /* Card de fontes */
    .source-card {
        background: #f0f4f8;
        border-left: 4px solid #1e3a5f;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* Badge */
    .relevance-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
    }

    .welcome-card h3 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }

    .welcome-card p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }

    /* Feature pills */
    .feature-pill {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.3rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }

    /* Config section */
    .config-section {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .config-title {
        color: #58a6ff;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metric cards in sidebar */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* Sliders */
    .stSlider label {
        color: #c9d1d9 !important;
    }

    /* Toggle */
    .stToggle label {
        color: #c9d1d9 !important;
    }
</style>
"""


def inject_css() -> None:
    """Injeta CSS customizado."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    """Inicializa estado da sessão."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": settings.top_k_results,
            "threshold": settings.similarity_threshold,
            "show_sources": True,
            "use_reranker": False,
            "use_hybrid": False,
            "show_visualizations": True,
            "comparison_mode": False,
        }
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "last_used_reranker" not in st.session_state:
        st.session_state.last_used_reranker = False
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None
    if "selected_chunk" not in st.session_state:
        st.session_state.selected_chunk = None


def render_header() -> None:
    """Renderiza header principal."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🧠 Personal Knowledge Base</h1>
            <p>Converse com suas notas e documentos usando IA local</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_card() -> None:
    """Renderiza card de boas-vindas quando não há mensagens."""
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Bem-vindo ao seu Knowledge Base</h3>
            <p>Faça perguntas sobre suas notas do Obsidian, PDFs e documentos.</p>
            <div style="margin-top: 1.5rem;">
                <span class="feature-pill">📝 Markdown</span>
                <span class="feature-pill">📄 PDF</span>
                <span class="feature-pill">🔗 Bookmarks</span>
                <span class="feature-pill">🔍 Busca Semântica</span>
                <span class="feature-pill">🤖 LLM Local</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sugestões de perguntas
    st.markdown("#### Sugestões de perguntas")
    cols = st.columns(3)
    suggestions = [
        "O que é RAG?",
        "Explique machine learning",
        "Quais são as melhores práticas?",
    ]

    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(f"💬 {suggestion}", key=f"sug_{suggestion}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


def sidebar() -> None:
    """Renderiza barra lateral."""
    with st.sidebar:
        # Logo/Título
        st.markdown(
            """
            <div style="text-align: center; padding: 1.5rem 0;">
                <span style="font-size: 3rem;">🧠</span>
                <h2 style="margin: 0.5rem 0 0 0; color: #58a6ff;">PKB</h2>
                <p style="font-size: 0.85rem; color: #8b949e;">Personal Knowledge Base</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Estatísticas
        stats = vector_store.get_stats()

        st.markdown('<p class="config-title">📊 Estatísticas</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Chunks", value=stats["document_count"])
        with col2:
            st.metric(label="Top-K", value=st.session_state.settings["top_k"])

        st.divider()

        # Configurações de Retrieval
        st.markdown('<p class="config-title">⚙️ Configurações</p>', unsafe_allow_html=True)

        st.session_state.settings["top_k"] = st.select_slider(
            "Resultados por busca",
            options=list(range(1, 21)),
            value=st.session_state.settings["top_k"],
            help="Quantos chunks retornar",
        )

        st.session_state.settings["threshold"] = st.select_slider(
            "Threshold mínimo",
            options=[round(x * 0.05, 2) for x in range(0, 21)],
            value=st.session_state.settings["threshold"],
            help="Score mínimo de similaridade",
        )

        st.divider()

        # Opções de busca
        st.markdown('<p class="config-title">🔍 Busca</p>', unsafe_allow_html=True)

        st.session_state.settings["use_hybrid"] = st.toggle(
            "Busca Híbrida (BM25 + Semântica)",
            value=st.session_state.settings["use_hybrid"],
            help="Combina busca por keywords com busca semântica",
        )

        st.session_state.settings["use_reranker"] = st.toggle(
            "Re-ranking (Cross-Encoder)",
            value=st.session_state.settings["use_reranker"],
            help="Maior precisão, mais lento",
        )

        st.session_state.settings["comparison_mode"] = st.toggle(
            "Modo Comparação",
            value=st.session_state.settings["comparison_mode"],
            help="Compara resultados com e sem re-ranking",
        )

        st.divider()

        # Opções de visualização
        st.markdown('<p class="config-title">👁️ Visualização</p>', unsafe_allow_html=True)

        st.session_state.settings["show_sources"] = st.toggle(
            "Mostrar fontes",
            value=st.session_state.settings["show_sources"],
        )

        st.session_state.settings["show_visualizations"] = st.toggle(
            "Gráficos do pipeline",
            value=st.session_state.settings["show_visualizations"],
            help="Visualizações do pipeline RAG",
        )

        st.divider()

        # Ações
        st.markdown('<p class="config-title">🔄 Ações</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Limpar", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_query = None
                st.session_state.last_results = None
                st.session_state.last_response = None
                st.session_state.last_used_reranker = False
                st.rerun()

        with col2:
            if st.button("🔄 Atualizar", use_container_width=True):
                st.rerun()

        # Footer
        st.divider()
        st.markdown(
            """
            <div style="text-align: center; font-size: 0.75rem; color: #8b949e;">
                <p style="margin: 0;">Powered by Ollama + ChromaDB</p>
                <p style="margin: 0.25rem 0 0 0;">qwen2.5:14b | nomic-embed-text</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_sources_html(results: list, citation_extractor) -> str:
    """Formata fontes em HTML."""
    cited_response = citation_extractor.create_cited_response("", results)

    html_parts = []
    for i, citation in enumerate(cited_response.citations, 1):
        source_name = os.path.basename(citation.source)
        relevance_pct = f"{citation.relevance_score:.0%}"

        html_parts.append(
            f"""
            <div class="source-card">
                <strong>{i}. {source_name}</strong>
                <span class="relevance-badge">{relevance_pct}</span>
                <br><small style="color: #666;">{citation.source}</small>
            </div>
            """
        )

    return "".join(html_parts)


def chat_interface() -> None:
    """Renderiza interface de chat."""
    # Verificar se há pergunta pendente (das sugestões)
    if hasattr(st.session_state, "pending_question"):
        pending = st.session_state.pending_question
        del st.session_state.pending_question
        process_question(pending)
        return

    # Mostrar welcome card se não houver mensagens
    if not st.session_state.messages:
        render_welcome_card()

    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Ver fontes utilizadas", expanded=False):
                    st.markdown(message["sources"], unsafe_allow_html=True)

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre suas notas..."):
        process_question(prompt)


def process_question(prompt: str) -> None:
    """Processa uma pergunta do usuário."""
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Gerar resposta
    with st.chat_message("assistant", avatar="🤖"):
        # Status de progresso
        status = st.status("Processando sua pergunta...", expanded=True)

        with status:
            st.write("🔍 Buscando documentos relevantes...")

            # Criar retriever com configurações atuais
            retriever = Retriever(
                top_k=st.session_state.settings["top_k"],
                threshold=st.session_state.settings["threshold"],
                use_reranker=st.session_state.settings["use_reranker"],
                use_hybrid=st.session_state.settings["use_hybrid"],
            )

            # Retrieval
            results = retriever.retrieve(prompt)

            if st.session_state.settings["use_hybrid"]:
                st.write("🔀 Aplicando busca híbrida (BM25 + Semântica)...")

            if st.session_state.settings["use_reranker"]:
                st.write("🎯 Aplicando re-ranking com Cross-Encoder...")

            st.write(f"✅ Encontrados {len(results)} chunks relevantes")

            # Modo comparação: buscar também sem re-ranking
            comparison_results = None
            if st.session_state.settings["comparison_mode"]:
                st.write("⚖️ Gerando comparação com/sem re-ranking...")
                retriever_no_rerank = Retriever(
                    top_k=st.session_state.settings["top_k"],
                    threshold=st.session_state.settings["threshold"],
                    use_reranker=False,
                    use_hybrid=st.session_state.settings["use_hybrid"],
                )
                retriever_with_rerank = Retriever(
                    top_k=st.session_state.settings["top_k"],
                    threshold=st.session_state.settings["threshold"],
                    use_reranker=True,
                    use_hybrid=st.session_state.settings["use_hybrid"],
                )
                comparison_results = {
                    "without_rerank": retriever_no_rerank.retrieve(prompt),
                    "with_rerank": retriever_with_rerank.retrieve(prompt),
                }
                st.session_state.comparison_results = comparison_results

            # Construir contexto
            context = retriever.build_context(results)

            st.write("🤖 Gerando resposta com LLM...")

        status.update(label="Resposta gerada!", state="complete", expanded=False)

        # Gerar resposta com streaming
        llm_service = get_llm_service()
        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm_service.generate_stream(
            prompt=prompt,
            context=context,
        ):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # Formatar citações
        sources_html = ""
        if st.session_state.settings["show_sources"] and results:
            citation_extractor = get_citation_extractor()
            sources_html = format_sources_html(results, citation_extractor)

            with st.expander("📚 Ver fontes utilizadas", expanded=False):
                st.markdown(sources_html, unsafe_allow_html=True)

        # Salvar no histórico
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": sources_html,
            }
        )

        # Salvar para visualizações
        st.session_state.last_query = prompt
        st.session_state.last_results = results
        st.session_state.last_response = full_response
        st.session_state.last_used_reranker = st.session_state.settings["use_reranker"]


def render_chunk_preview() -> None:
    """Renderiza preview de chunks clicáveis."""
    if not st.session_state.last_results:
        return

    st.divider()
    st.markdown("### 📄 Preview dos Chunks Recuperados")
    st.caption("Clique em um chunk para ver o conteúdo completo")

    # Grid de chunks
    cols = st.columns(3)
    for i, result in enumerate(st.session_state.last_results):
        with cols[i % 3]:
            title = result.metadata.get("title", os.path.basename(result.source))
            if len(title) > 25:
                title = title[:22] + "..."

            # Card do chunk
            with st.container():
                if st.button(
                    f"📄 {title}\n{result.score:.0%}",
                    key=f"chunk_{i}",
                    use_container_width=True,
                ):
                    st.session_state.selected_chunk = i

    # Modal com conteúdo do chunk selecionado
    if st.session_state.selected_chunk is not None:
        idx = st.session_state.selected_chunk
        if idx < len(st.session_state.last_results):
            result = st.session_state.last_results[idx]

            st.markdown("---")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"#### 📄 {result.metadata.get('title', 'Documento')}")
            with col2:
                if st.button("✖️ Fechar", key="close_preview"):
                    st.session_state.selected_chunk = None
                    st.rerun()

            st.markdown(f"**Fonte:** `{result.source}`")
            st.markdown(f"**Score:** {result.score:.1%}")

            # Tipo de busca se disponível
            search_type = result.metadata.get("search_type", "semantic")
            st.markdown(f"**Tipo:** {search_type}")

            st.divider()
            st.markdown("**Conteúdo:**")
            st.text_area(
                "Conteúdo do chunk",
                value=result.content,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )


def render_comparison() -> None:
    """Renderiza comparação com/sem re-ranking."""
    if not st.session_state.comparison_results:
        return

    st.divider()
    st.markdown("### ⚖️ Comparação: Com vs Sem Re-ranking")
    st.caption("Veja como o re-ranking afeta a ordenação dos resultados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sem Re-ranking")
        results_no_rerank = st.session_state.comparison_results["without_rerank"]
        for i, r in enumerate(results_no_rerank[:5], 1):
            title = r.metadata.get("title", os.path.basename(r.source))
            if len(title) > 30:
                title = title[:27] + "..."
            st.markdown(
                f"**{i}.** {title} "
                f"<span style='color: #58a6ff;'>({r.score:.0%})</span>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### Com Re-ranking")
        results_with_rerank = st.session_state.comparison_results["with_rerank"]
        for i, r in enumerate(results_with_rerank[:5], 1):
            title = r.metadata.get("title", os.path.basename(r.source))
            if len(title) > 30:
                title = title[:27] + "..."
            st.markdown(
                f"**{i}.** {title} "
                f"<span style='color: #4ECDC4;'>({r.score:.0%})</span>",
                unsafe_allow_html=True,
            )

    # Análise de diferenças
    st.markdown("---")
    st.markdown("**Análise:**")

    # Verificar reordenação
    sources_no = [r.source for r in results_no_rerank[:5]]
    sources_with = [r.source for r in results_with_rerank[:5]]

    if sources_no == sources_with:
        st.info("Os resultados mantiveram a mesma ordem após re-ranking.")
    else:
        changed = sum(1 for i, s in enumerate(sources_with) if i < len(sources_no) and s != sources_no[i])
        st.success(f"Re-ranking alterou {changed} posições nos top 5 resultados.")


def render_visualizations() -> None:
    """Renderiza seção de visualizações."""
    if not (
        st.session_state.settings["show_visualizations"]
        and st.session_state.last_results
    ):
        return

    st.divider()

    # Header da seção
    st.markdown("### 📊 Visualizações do Pipeline RAG")
    st.caption("Entenda como sua pergunta foi processada")

    # Tabs com labels claros
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔀 Fluxo RAG",
        "📈 Scores de Relevância",
        "🗺️ Espaço de Embeddings",
        "📄 Preview de Chunks"
    ])

    with tab1:
        if (
            st.session_state.last_query
            and st.session_state.last_response
            and st.session_state.last_results
        ):
            st.markdown("##### Fluxo: Query → Retrieval → Generation")
            fig_flow = create_rag_flow_diagram(
                query=st.session_state.last_query,
                results=st.session_state.last_results,
                response=st.session_state.last_response,
                used_reranker=st.session_state.last_used_reranker,
            )
            st.plotly_chart(fig_flow, use_container_width=True)

    with tab2:
        if st.session_state.last_results:
            st.markdown("##### Relevância de cada chunk recuperado")
            fig_scores = create_scores_chart(st.session_state.last_results)
            st.plotly_chart(fig_scores, use_container_width=True)

            # Métricas resumidas
            scores = [r.score for r in st.session_state.last_results]
            cols = st.columns(4)
            with cols[0]:
                st.metric("Chunks", len(scores))
            with cols[1]:
                st.metric("Score Máx", f"{max(scores):.0%}")
            with cols[2]:
                st.metric("Score Mín", f"{min(scores):.0%}")
            with cols[3]:
                st.metric("Score Médio", f"{sum(scores)/len(scores):.0%}")

    with tab3:
        st.markdown("##### Projeção 2D do espaço de embeddings (UMAP)")
        st.caption(
            "Mostra a proximidade semântica entre os chunks recuperados e sua pergunta."
        )

        if st.button("Gerar Visualização", type="primary"):
            with st.spinner("Calculando embeddings e projeção UMAP..."):
                # Obter embeddings dos resultados
                embedding_service = get_embedding_service()

                # Embeddings dos chunks recuperados
                embeddings = []
                labels = []
                doc_types = []

                progress = st.progress(0, text="Gerando embeddings...")
                total = len(st.session_state.last_results)

                for i, result in enumerate(st.session_state.last_results):
                    emb = embedding_service.embed_text(result.content)
                    embeddings.append(emb)

                    # Usar título ou nome do arquivo
                    title = result.metadata.get("title", "")
                    if title:
                        label = title[:30] if len(title) <= 30 else title[:27] + "..."
                    else:
                        label = os.path.splitext(os.path.basename(result.source))[0][:30]
                    labels.append(label)
                    doc_types.append(result.metadata.get("doc_type", "unknown"))

                    progress.progress((i + 1) / total, text=f"Embedding {i+1}/{total}")

                # Embedding da query
                query_emb = embedding_service.embed_text(st.session_state.last_query)
                progress.empty()

                if len(embeddings) >= 2:
                    fig_emb = create_embeddings_visualization(
                        embeddings=embeddings,
                        labels=labels,
                        doc_types=doc_types,
                        query_embedding=query_emb,
                        query_label=st.session_state.last_query[:30],
                    )
                    st.plotly_chart(fig_emb, use_container_width=True)
                else:
                    st.warning(
                        "Precisa de pelo menos 2 chunks para visualizar."
                    )

    with tab4:
        st.markdown("##### Chunks recuperados")
        st.caption("Clique em um chunk para expandir e ver o conteúdo completo")

        for i, result in enumerate(st.session_state.last_results):
            title = result.metadata.get("title", os.path.basename(result.source))
            search_type = result.metadata.get("search_type", "semantic")
            search_badge = "🔍" if search_type == "semantic" else "🔀" if search_type == "hybrid" else "📝"

            with st.expander(f"{search_badge} **{title}** - Score: {result.score:.0%}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Fonte:** `{result.source}`")
                with col2:
                    st.markdown(f"**Tipo:** {search_type}")

                st.divider()
                st.markdown(result.content)


def main() -> None:
    """Função principal."""
    inject_css()
    init_session_state()
    sidebar()

    # Layout principal
    render_header()
    chat_interface()

    # Modo comparação
    if st.session_state.settings["comparison_mode"]:
        render_comparison()

    render_visualizations()


if __name__ == "__main__":
    main()
