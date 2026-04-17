"""
Credit Rating Intelligence Agent — Streamlit UI
================================================
Run with: streamlit run app.py
"""

import os
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "credit-rating-agent")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Rating Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Dark terminal background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Main header */
    .main-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        color: #f0b429;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid #30363d;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }

    /* Rating badge */
    .rating-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1rem;
    }
    .rating-ig { background: #0e3a1e; color: #3fb950; border: 1px solid #3fb950; }
    .rating-hy { background: #3d1a00; color: #f0883e; border: 1px solid #f0883e; }
    .rating-nr { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

    /* Split rating alert */
    .split-alert {
        background: #2d1b00;
        border: 1px solid #f0b429;
        border-left: 4px solid #f0b429;
        padding: 16px;
        border-radius: 6px;
        margin: 16px 0;
    }

    /* Agency card */
    .agency-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #8b949e;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px;
    }

    /* Query history item */
    .history-item {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        cursor: pointer;
        font-size: 0.85rem;
        color: #8b949e;
    }

    /* Answer box */
    .answer-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px 24px;
        line-height: 1.7;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #f0b429;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Override Streamlit button */
    .stButton > button {
        background: #f0b429 !important;
        color: #0d1117 !important;
        border: none !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
    }

    /* Input */
    .stTextInput > div > div > input, .stTextArea textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        border-radius: 6px !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea textarea:focus {
        border-color: #f0b429 !important;
        box-shadow: 0 0 0 2px rgba(240, 180, 41, 0.15) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        color: #f0b429 !important;
        border-bottom: 2px solid #f0b429 !important;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.7rem;
                color: #f0b429; letter-spacing: 0.15em; text-transform: uppercase;
                padding: 8px 0 16px 0; border-bottom: 1px solid #30363d;'>
        ◈ CRIA v1.0
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data source status
    st.markdown("**DATA SOURCES**", unsafe_allow_html=False)
    agencies = {
        "DBRS Morningstar": True,
        "Fitch Ratings": True,
        "Moody's": True,
        "S&P Global": True,
    }
    for agency, active in agencies.items():
        dot = "🟢" if active else "🔴"
        st.markdown(f"{dot} {agency}")

    st.markdown("---")

    # Ingestion controls
    st.markdown("**VECTOR STORE**")
    if st.button("⟳ Refresh Data", use_container_width=True):
        with st.spinner("Ingesting latest rating actions..."):
            try:
                from ingestion.pipeline import ingest_all_agencies
                results = ingest_all_agencies()
                st.session_state.ingestion_done = True
                total = sum(results.values())
                st.success(f"✓ {total} chunks stored")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.markdown("---")

    # Query history
    st.markdown("**RECENT QUERIES**")
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            st.markdown(
                f"<div class='history-item'>↩ {item['query'][:55]}...</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<div style='color:#8b949e; font-size:0.8rem;'>No queries yet</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # LangSmith link
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "credit-rating-agent")
    st.markdown(
        f"[📊 LangSmith Traces ↗](https://smith.langchain.com/projects/{langsmith_project})"
    )


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    Credit Rating Intelligence Agent
</div>
""", unsafe_allow_html=True)

# Tabs
tab_query, tab_compare, tab_alerts, tab_eval = st.tabs([
    "QUERY", "COMPARE", "SPLIT ALERTS", "EVAL METRICS"
])


# ── Tab 1: Query ───────────────────────────────────────────────────────────────
with tab_query:
    col_input, col_suggest = st.columns([3, 1])

    with col_input:
        query = st.text_area(
            "Ask anything about credit ratings",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g. What is Ford Motor Company's current rating across all agencies?",
            height=100,
            label_visibility="collapsed"
        )

    with col_suggest:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Quick queries:**")
        sample_queries = [
            "RBC ratings all agencies",
            "Ford split rating check",
            "IG downgrades this week",
            "DBRS covered bond methodology",
        ]
        for sq in sample_queries:
            if st.button(sq, key=f"btn_{sq}", use_container_width=True):
                st.session_state["run_query"] = sq
                st.session_state["query_input"] = sq
                st.rerun()
    run_col, clear_col = st.columns([1, 5])
    with run_col:
        run_btn = st.button("RUN QUERY →", use_container_width=True)

    if (run_btn and query) or st.session_state.get("run_query"):
        query = st.session_state.pop("run_query", query)
        with st.spinner("Agent reasoning..."):
            try:
                from agent.graph import run_agent, build_agent_graph, AgentState
                from langchain_core.messages import HumanMessage

                # Run agent and capture intermediate state
                agent = build_agent_graph()
                initial_state: AgentState = {
                    "messages": [HumanMessage(content=query)],
                    "query": query,
                    "query_type": None,
                    "retrieved_docs": [],
                    "agencies_searched": [],
                    "individual_ratings": {},
                    "split_rating_detected": False,
                    "conflict_explanation": None,
                    "final_answer": None,
                    "needs_human_review": False,
                    "confidence_score": 0.0,
                }
                result = agent.invoke(initial_state)
                st.session_state.last_result = result
                st.session_state.query_history.append({
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "split_detected": result.get("split_rating_detected", False),
                })

            except Exception as e:
                st.error(f"Agent error: {e}")
                st.info("Make sure you've run ingestion first and set up your API keys.")
                # Show mock result for UI demonstration
                st.session_state.last_result = {
                    "final_answer": f"[Demo mode — configure API keys to run live]\n\nYour query was: '{query}'\n\nIn production, this would return a full analyst-quality response comparing ratings across DBRS, Fitch, Moody's, and S&P.",
                    "query_type": "multi_agency",
                    "individual_ratings": {"DBRS": "BBB(high)", "FITCH": "BBB+", "MCO": "Baa1", "SP": "BBB+"},
                    "split_rating_detected": False,
                    "agencies_searched": ["DBRS", "FITCH", "MCO", "SP"],
                    "retrieved_docs": [],
                }

    # Display result
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown("<br>", unsafe_allow_html=True)

        # Split rating alert — shown prominently if detected
        if result.get("split_rating_detected"):
            st.markdown(f"""
            <div class='split-alert'>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 0.75rem;
                            color: #f0b429; letter-spacing: 0.1em; margin-bottom: 8px;'>
                    ⚠ SPLIT RATING DETECTED
                </div>
                <div style='color: #e6edf3; font-size: 0.9rem;'>
                    {result.get('conflict_explanation', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Per-agency rating badges
        ratings = result.get("individual_ratings", {})
        if ratings:
            st.markdown("**Ratings extracted:**")
            cols = st.columns(len(ratings))
            agency_labels = {"DBRS": "DBRS Morningstar", "FITCH": "Fitch", "MCO": "Moody's", "SP": "S&P"}
            for i, (agency, rating) in enumerate(ratings.items()):
                # Determine IG vs HY
                ig_ratings = {"AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","Aaa","Aa1","Aa2","Aa3","A1","A2","A3","Baa1","Baa2","Baa3"}
                css_class = "rating-ig" if any(r in rating for r in ["AAA","AA","A","BBB","Baa","Aaa","Aa"]) else "rating-hy"
                with cols[i]:
                    st.markdown(f"""
                    <div style='text-align:center; padding: 12px; background:#161b22;
                                border:1px solid #30363d; border-radius:8px;'>
                        <div style='font-size:0.7rem; color:#8b949e; font-family:IBM Plex Mono,monospace;
                                    letter-spacing:0.08em; margin-bottom:8px;'>
                            {agency_labels.get(agency, agency)}
                        </div>
                        <span class='rating-badge {css_class}'>{rating}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

        # Full answer
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown(result.get("final_answer", ""), unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)

        # Metadata row
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            qt = result.get("query_type", "—")
            st.markdown(f"<div style='font-size:0.75rem; color:#8b949e;'>Query type: <code>{qt}</code></div>",
                       unsafe_allow_html=True)
        with meta_col2:
            n_docs = len(result.get("retrieved_docs", []))
            st.markdown(f"<div style='font-size:0.75rem; color:#8b949e;'>Docs retrieved: <code>{n_docs}</code></div>",
                       unsafe_allow_html=True)
        with meta_col3:
            agencies_str = ", ".join(result.get("agencies_searched", []))
            st.markdown(f"<div style='font-size:0.75rem; color:#8b949e;'>Searched: <code>{agencies_str}</code></div>",
                       unsafe_allow_html=True)


# ── Tab 2: Side-by-Side Comparison ────────────────────────────────────────────
with tab_compare:
    st.markdown("**Compare a specific issuer across all agencies**")

    issuer_input = st.text_input(
        "Issuer name",
        placeholder="e.g. Royal Bank of Canada",
        label_visibility="collapsed"
    )

    if st.button("COMPARE ISSUER →") and issuer_input:
        compare_query = f"What are the current ratings and outlooks for {issuer_input} across DBRS Morningstar, Fitch, Moody's, and S&P? Include any recent rating actions."

        with st.spinner(f"Fetching ratings for {issuer_input}..."):
            try:
                from agent.graph import run_agent
                answer = run_agent(compare_query)
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            except Exception as e:
                # Demo mode
                st.markdown(f"""
                <div class='answer-box'>
                    <strong>Demo: {issuer_input}</strong><br><br>
                    Configure API keys and run ingestion to see live comparison.<br><br>
                    This view would show a structured table of all 4 agency ratings,
                    outlook trends, and a split-rating severity indicator.
                </div>
                """, unsafe_allow_html=True)

    # Sample comparison table layout
    st.markdown("---")
    st.markdown("**Sample output format:**")

    sample_data = {
        "Agency": ["DBRS Morningstar", "Fitch", "Moody's", "S&P"],
        "Rating": ["BBB (high)", "BBB+", "Baa1", "BBB+"],
        "Outlook": ["Stable", "Stable", "Stable", "Positive"],
        "Last Action": ["Affirmed Mar 2025", "Affirmed Jan 2025", "Affirmed Feb 2025", "Outlook Revised Jan 2025"],
    }
    import pandas as pd
    df = pd.DataFrame(sample_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 3: Split Rating Alerts ─────────────────────────────────────────────────
with tab_alerts:
    st.markdown("**Recent split rating detections**")

    # Show alerts from session history
    split_queries = [h for h in st.session_state.query_history if h.get("split_detected")]

    if split_queries:
        for alert in split_queries:
            st.markdown(f"""
            <div class='split-alert'>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 0.7rem;
                            color: #f0b429;'>⚠ SPLIT DETECTED</div>
                <div style='margin-top: 6px; color: #e6edf3;'>{alert['query']}</div>
                <div style='font-size: 0.7rem; color: #8b949e; margin-top: 4px;'>
                    {alert['timestamp'][:16]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='color:#8b949e; font-size:0.9rem; padding:24px;
                    border:1px dashed #30363d; border-radius:8px; text-align:center;'>
            No split rating alerts in this session.<br>
            Run multi-agency queries to detect divergences.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Known split-rating scenarios to investigate:**")
    scenarios = [
        ("Ford Motor Company", "DBRS vs S&P historically diverge on auto sector"),
        ("Government of Canada", "All agencies typically aligned — good baseline test"),
        ("Major US Regional Banks", "Post-2023 banking stress created notch gaps"),
        ("Emerging Market Corporates", "Moody's often more conservative than Fitch"),
    ]
    for issuer, note in scenarios:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown(f"**{issuer}**")
        with col_b:
            st.markdown(f"<span style='color:#8b949e; font-size:0.85rem;'>{note}</span>",
                       unsafe_allow_html=True)


# ── Tab 4: Eval Metrics ────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("**LangSmith Evaluation Dashboard**")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Faithfulness", "—", "No hallucinated ratings"),
        ("Rating Accuracy", "—", "Correct notch level"),
        ("Retrieval P@6", "—", "Relevant docs in top 6"),
        ("Avg Latency", "—", "End-to-end response"),
    ]
    for col, (label, value, note) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
                <div style='font-size:0.7rem; color:#8b949e; margin-top:6px;'>{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_run, col_link = st.columns([1, 4])
    with col_run:
        if st.button("▶ RUN EVAL SUITE"):
            with st.spinner("Running LangSmith evaluation..."):
                try:
                    from evaluation.langsmith_eval import run_evaluation
                    results = run_evaluation()
                    st.success("Eval complete — view results in LangSmith")
                except Exception as e:
                    st.info(f"Configure LangSmith API key to run evals. Error: {e}")

    st.markdown("---")
    st.markdown("**Prompt variant A/B test**")

    test_query = st.text_input("Test query for A/B comparison",
                               value="What is the outlook on investment grade Canadian bank debt?")
    if st.button("RUN A/B TEST →") and test_query:
        with st.spinner("Testing 3 prompt variants..."):
            try:
                from evaluation.langsmith_eval import run_prompt_ab_test
                results = run_prompt_ab_test(test_query)
                for variant, answer in results.items():
                    with st.expander(f"Variant: {variant}"):
                        st.write(answer)
            except Exception as e:
                st.info(f"A/B test requires API keys. Error: {e}")
