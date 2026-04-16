# Credit Rating Intelligence Agent

A multi-agent system that monitors, retrieves, and cross-compares credit rating actions
across **DBRS Morningstar, Fitch, Moody's, and S&P** — with a signature **split-rating detector**.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INGESTION LAYER                     │
│  RSS Feeds + HTML Scraping → Chunking → ChromaDB    │
│  (LangChain WebLoaders, text-embedding-3-small)      │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│               LANGGRAPH AGENT                        │
│                                                      │
│  [Router] → [Single Agency RAG]  ─┐                 │
│           → [Multi Agency RAG]   ─┤→ [Extractor]    │
│                                    → [Split Detector]│
│                                    → [Synthesizer]   │
│                                    → [Human Review]  │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              LANGSMITH OBSERVABILITY                 │
│  Traces · Evals · Prompt A/B Tests · Dashboards     │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
credit-rating-agent/
├── ingestion/
│   ├── agency_configs.py    # Data source URLs per agency
│   ├── models.py            # Pydantic schemas (RatingAction, SplitRatingAlert)
│   └── pipeline.py          # Scrape → chunk → embed → store
├── agent/
│   └── graph.py             # Full LangGraph agent (6 nodes)
├── evaluation/
│   └── langsmith_eval.py    # Ground truth dataset + custom evaluators
├── scheduler.py             # Daily ingestion cron
├── main.py                  # CLI entry point
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, LANGCHAIN_API_KEY

# 3. Run initial ingestion
python -m ingestion.pipeline

# 4. Ask a question
python main.py "What is Ford's rating across all agencies?"

# 5. Run evaluations
python -m evaluation.langsmith_eval
```

## Example Queries

- `"What is RBC's current rating across all four agencies?"`
- `"Has there been any divergence between DBRS and S&P on Canadian bank debt?"`
- `"Which investment grade issuers had negative outlook changes this week?"`
- `"Compare DBRS and Fitch's approach to rating covered bonds"`

## Key Features

| Feature | What it shows |
|---------|--------------|
| Multi-agency RAG | LangChain retrieval with metadata filtering |
| LangGraph routing | Conditional agent graph with 6 nodes |
| Split-rating detector | Notch-level comparison with severity scoring |
| Human-in-the-loop | Compliance review gate for significant divergences |
| LangSmith evals | Faithfulness + accuracy metrics, prompt A/B testing |
