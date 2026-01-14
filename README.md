# Personal Knowledge Base

Sistema RAG (Retrieval-Augmented Generation) pessoal para consultar notas do Obsidian, PDFs e bookmarks usando LLMs locais via Ollama.

## Visao Geral

Este projeto permite fazer perguntas em linguagem natural sobre seus documentos pessoais. O sistema busca os trechos mais relevantes e usa um LLM local para gerar respostas contextualizadas com citacoes das fontes.

```
Pergunta do Usuario
        |
        v
   Retriever (busca semantica)
        |
        v
   ChromaDB (banco de vetores)
        |
        v
   [Opcional: Re-Ranker]
        |
        v
   LLM (Ollama - qwen2.5:14b)
        |
        v
   Resposta + Citacoes
```

## Funcionalidades

- **Multiplos formatos**: Markdown/Obsidian, PDF, URLs e bookmarks
- **Busca semantica**: Embeddings com `nomic-embed-text`
- **Busca hibrida**: Combina busca semantica com BM25 (keyword matching)
- **Re-ranking opcional**: Cross-Encoder para maior precisao
- **Cache de embeddings**: Evita recalcular embeddings para queries repetidas
- **Modo comparacao**: Visualiza resultados com/sem re-ranking lado a lado
- **Chunking inteligente**: Respeita estrutura de headers do Markdown
- **Preview de chunks**: Expande chunks para ver conteudo completo
- **Duas interfaces**: CLI interativa e Web (Streamlit)
- **Visualizacoes**: Diagrama de fluxo RAG, scores, espaco de embeddings
- **100% local**: Nenhuma API externa, tudo roda na sua maquina

## Pre-requisitos

- Python 3.10+
- [Ollama](https://ollama.ai/) instalado e rodando
- Modelos baixados:
  ```bash
  ollama pull qwen2.5:14b       # LLM principal
  ollama pull nomic-embed-text  # Embeddings
  ```

## Instalacao

1. Clone o repositorio e entre na pasta:
   ```bash
   cd "Personal Knowledge Base"
   ```

2. Crie o ambiente virtual e instale as dependencias:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # ou: source .venv/bin/activate  # Linux/Mac

   pip install -e .
   ```

3. Configure o arquivo `.env` (copie de `.env.example`):
   ```ini
   # Diretorios de dados
   OBSIDIAN_VAULT_PATH=./data/obsidian
   PDF_DIRECTORY=./data/pdfs
   BOOKMARKS_DIRECTORY=./data/bookmarks

   # ChromaDB
   CHROMA_DB_PATH=./chroma_db
   CHROMA_COLLECTION_NAME=personal_knowledge

   # Ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_LLM_MODEL=qwen2.5:14b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text

   # Chunking
   CHUNK_SIZE=800
   CHUNK_OVERLAP=100

   # Retrieval
   TOP_K_RESULTS=5
   SIMILARITY_THRESHOLD=0.6
   ```

## Uso

### Ingestao de Documentos

Adicione seus documentos ao banco de vetores:

```bash
# Markdown/Obsidian
python -m src.interface.cli ingest "data/obsidian/minha-pasta" --type markdown

# PDFs
python -m src.interface.cli ingest "data/pdfs" --type pdf
```

### Interface CLI

```bash
# Chat interativo
python -m src.interface.cli chat

# Chat com re-ranking (mais preciso, mais lento)
python -m src.interface.cli chat --rerank

# Pergunta unica
python -m src.interface.cli ask "O que e RAG?"

# Busca semantica
python -m src.interface.cli search "machine learning" --limit 5

# Estatisticas
python -m src.interface.cli stats

# Limpar banco
python -m src.interface.cli reset --force
```

### Interface Web (Streamlit)

```bash
.venv\Scripts\streamlit.exe run src/interface/streamlit_app.py
```

Acesse: http://localhost:8501

Funcionalidades da interface web:
- Chat com historico
- Configuracoes na sidebar (top_k, threshold, re-ranker, busca hibrida)
- Modo comparacao: visualiza resultados com/sem re-ranking
- Visualizacoes interativas do pipeline RAG
- Grafico de scores de relevancia
- Visualizacao 2D do espaco de embeddings (UMAP)
- Preview de chunks com conteudo expandivel

## Arquitetura

```
src/
├── core/                    # Nucleo do sistema RAG
│   ├── config.py           # Configuracoes (Pydantic Settings)
│   ├── embeddings.py       # Servico de embeddings (Ollama) + Cache
│   ├── hybrid_search.py    # BM25 + HybridSearcher
│   ├── llm.py              # Servico LLM (Ollama)
│   ├── vectorstore.py      # Gerenciador ChromaDB
│   ├── retriever.py        # Busca semantica + hibrida
│   └── reranker.py         # Re-ranking com Cross-Encoder
│
├── ingestors/              # Processadores de documentos
│   ├── base.py             # Classe base + Document dataclass
│   ├── markdown.py         # Markdown/Obsidian (frontmatter, wikilinks, tags)
│   ├── pdf.py              # PDF (texto + tabelas)
│   └── web.py              # URLs e bookmarks
│
├── interface/              # Interfaces de usuario
│   ├── cli.py              # CLI com Typer/Rich
│   ├── streamlit_app.py    # Interface web
│   └── visualizations.py   # Graficos Plotly
│
└── utils/                  # Utilitarios
    ├── chunking.py         # Estrategias de chunking
    └── citations.py        # Extracao de citacoes
```

## Componentes Principais

### Ingestores

| Ingestor | Formatos | Funcionalidades |
|----------|----------|-----------------|
| `MarkdownIngestor` | `.md` | Frontmatter YAML, wikilinks `[[]]`, tags `#tag`, chunking por headers |
| `PDFIngestor` | `.pdf` | Extracao de texto, tabelas, paginacao |
| `WebIngestor` | URLs, HTML/JSON bookmarks | Fetch de conteudo, parsing de bookmarks |

### Chunking

O sistema usa duas estrategias de chunking:

1. **RecursiveChunker**: Divide por `\n\n` → `\n` → `. ` → ` ` → caracteres
2. **MarkdownChunker**: Respeita headers Markdown, mantendo secoes inteiras

Parametros:
- `CHUNK_SIZE=800`: Tamanho maximo do chunk em caracteres
- `CHUNK_OVERLAP=100`: Sobreposicao entre chunks

### Busca Hibrida

O sistema suporta busca hibrida combinando duas abordagens:

1. **Busca Semantica**: Usa embeddings para encontrar documentos semanticamente similares
2. **BM25**: Algoritmo classico de keyword matching (TF-IDF melhorado)

```
Query → Busca Semantica (70%) + BM25 (30%) → Scores Normalizados → Resultados Combinados
```

Pesos configuraveis via sidebar:
- `semantic_weight`: Peso da busca semantica (padrao: 0.7)
- `bm25_weight`: Peso do BM25 (padrao: 0.3)

Quando usar:
- Ativar quando a busca semantica pura nao retorna resultados satisfatorios
- Util para queries com termos tecnicos especificos

### Cache de Embeddings

O sistema cacheia embeddings de queries para evitar recalculos desnecessarios:

- Cache LRU com limite de 1000 entradas
- Hash MD5 do texto como chave
- Estatisticas de hit rate disponiveis

### Re-Ranking

O re-ranker opcional usa Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) para melhorar a precisao:

```
Busca inicial (top_k * 3) → Re-ranking → Top K final
```

Quando usar:
- Ativar para perguntas complexas que precisam de maior precisao
- Desativar para buscas rapidas ou perguntas simples

### Modo Comparacao

Permite visualizar lado a lado os resultados com e sem re-ranking:

- Compara scores e posicoes dos chunks
- Mostra diferenca de relevancia entre as abordagens
- Util para entender quando o re-ranking agrega valor

## Visualizacoes

A interface Streamlit oferece quatro visualizacoes:

1. **Fluxo RAG**: Diagrama mostrando Query → Retrieval → [Busca Hibrida] → [Re-ranking] → Generation
2. **Scores de Relevancia**: Grafico de barras com scores dos chunks recuperados
3. **Espaco de Embeddings**: Projecao 2D (UMAP) dos embeddings dos documentos
4. **Preview de Chunks**: Lista expandivel com conteudo completo dos chunks recuperados

## Configuracao Avancada

### Parametros de Retrieval

| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| `TOP_K_RESULTS` | 5 | Numero de chunks retornados |
| `SIMILARITY_THRESHOLD` | 0.6 | Score minimo para inclusao (0-1) |
| `CHUNK_SIZE` | 800 | Tamanho maximo do chunk |
| `CHUNK_OVERLAP` | 100 | Sobreposicao entre chunks |

### Modelos Ollama

Modelos testados:
- **LLM**: `qwen2.5:14b`, `llama3:8b`, `mistral:7b`
- **Embeddings**: `nomic-embed-text`, `mxbai-embed-large`

Para trocar o modelo, edite `.env`:
```ini
OLLAMA_LLM_MODEL=llama3:8b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

## Dependencias Principais

- **langchain** - Orquestracao LLM
- **chromadb** - Banco de vetores
- **ollama** - Cliente Ollama
- **sentence-transformers** - Cross-Encoder para re-ranking
- **streamlit** - Interface web
- **typer** + **rich** - CLI
- **plotly** - Visualizacoes
- **umap-learn** - Reducao de dimensionalidade
- **pdfplumber** - Extracao de PDF
- **python-frontmatter** - Parse de Markdown

## Estrutura de Dados

```
data/
├── obsidian/           # Vault do Obsidian
│   └── Study and Work2/
│       ├── 01 - Ciencia da Computacao/
│       ├── 03 - Livros/
│       └── 04 - Projetos/
├── pdfs/               # Documentos PDF
└── bookmarks/          # Bookmarks exportados

chroma_db/              # Banco de vetores (persistente)
```

## Desenvolvimento

### Formatacao de Codigo

```bash
# Formatar com Ruff
ruff format src/

# Verificar estilo
ruff check src/
```

### Testes

```bash
pytest tests/ -v
```

## Licenca

MIT

## Autor

Desenvolvido como projeto pessoal para gerenciamento de conhecimento.
