# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run initial data ingestion (required before querying)
python -m ingestion.pipeline

# CLI query interface
python main.py "What is Ford's credit rating across all agencies?"
# Or launch interactive prompt:
python main.py

# Streamlit web UI (http://localhost:8501)
streamlit run app.py

# Background daily scheduler (ingests at 06:00 UTC)
python scheduler.py

# Run LangSmith evaluation suite
python -m evaluation.langsmith_eval
```

No test suite, linter config, or CI/CD pipelines exist in this repository.

## Environment Variables

Create a `.env` file in the project root (no `.env.example` exists):

```
OPENAI_API_KEY=...          # GPT-4o LLM + text-embedding-3-small
LANGCHAIN_API_KEY=...       # LangSmith tracing
LANGCHAIN_TRACING_V2=true   # Auto-set by main.py and app.py
LANGCHAIN_PROJECT=credit-rating-agent  # Auto-set
CHROMA_PERSIST_DIR=./data/vectorstore  # Default path
REQUEST_DELAY_SECONDS=2.0              # Delay between scrape requests
MAX_DOCS_PER_AGENCY=50                 # Ingestion cap per agency
```

## Architecture

The system is a multi-agency credit rating comparison agent built on LangGraph with ChromaDB vector search, a Streamlit UI, and LangSmith observability.

### Data Flow

```
Ingestion (ingestion/pipeline.py)
  → Google News RSS feeds (4 agencies)
  → Rating-keyword filter
  → Text chunked (800 chars, 100 overlap)
  → Embedded (text-embedding-3-small)
  → ChromaDB (./data/vectorstore, one collection per agency)

User Query (main.py CLI or app.py Streamlit)
  → agent/graph.py (LangGraph state machine)
  → Response + structured rating data
```

### LangGraph Agent (`agent/graph.py`)

Six-node state machine operating on `AgentState` (TypedDict):

1. **Router** — classifies query into `single_agency`, `multi_agency`, `split_rating`, or `methodology`
2. **single_agency_rag_node** — retrieves 6 docs from one agency's Chroma collection (agency detected from query keywords: "dbrs", "fitch", "moody", "s&p")
3. **multi_agency_comparator_node** — retrieves 6 docs per agency (24 total) across all four
4. **Rating Extractor** — LLM extracts structured JSON: `{issuer, ratings: {DBRS/FITCH/MCO/SP: {rating, outlook}}}`
5. **Split Rating Detector** — compares agencies on a 22-point notch scale (AAA=22 … B-=7); gap ≥ 3 notches sets `needs_human_review=True`
6. **Synthesizer** — LLM generates final analyst-quality response; if `needs_human_review`, tags response `[PENDING COMPLIANCE REVIEW]` and prints a console warning

### Ingestion Layer (`ingestion/`)

- **agency_configs.py**: Four agencies configured with Google News RSS feeds — DBRS Morningstar, Fitch, Moody's (MCO), S&P (SP). Each config includes `agency_code`, RSS URL, and scrape selectors.
- **pipeline.py**: `ingest_all_agencies()` → fetch feed → `is_rating_content()` keyword filter → chunk → embed → upsert to Chroma. Metadata stored per doc: `agency`, `agency_name`, `source_url`, `title`, `publication_date`, `doc_type`.
- **models.py**: Pydantic v2 schemas `RatingAction` and `SplitRatingAlert`.

### Streamlit UI (`app.py`)

Four tabs: Query (main interface), Compare (side-by-side issuer comparison), Split Alerts (divergence history), Eval Metrics (LangSmith dashboard + A/B test panel). Sidebar shows vector store status and per-agency doc counts. Ratings rendered as color-coded badges (investment grade vs. high yield).

### Evaluation (`evaluation/langsmith_eval.py`)

- 5-item ground truth dataset (sovereign issuers: US, Thailand, Saudi Arabia, Bahrain)
- Two evaluators: `faithfulness_evaluator` (binary hallucination check) and `rating_accuracy_evaluator` (0.0–1.0 notch-level accuracy)
- Three prompt variants for A/B testing: `analyst_persona`, `neutral_assistant`, `compliance_focused`
- Uses LangSmith SDK 0.2+ API

## Key Conventions

**Agency codes** are `DBRS`, `FITCH`, `MCO`, `SP` — used as Chroma collection names and in `AgentState.agencies_searched`.

**Notch scale** is defined once in `agent/graph.py` as `NOTCH_SCALE` dict; any rating comparison logic must use this dict (not re-implement).

**LangChain/Chroma imports**: The project has been updated away from deprecated imports — use `langchain_community` and `langchain_openai` rather than top-level `langchain` imports for embeddings and vector stores.

**All structured rating output** from the extractor is JSON embedded in the LLM response and parsed with `json.loads()` after stripping markdown code fences — no pydantic parsing at the agent boundary.
