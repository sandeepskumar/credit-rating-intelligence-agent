"""
Credit Rating Agent — LangGraph Orchestration
==============================================
Models the analyst workflow as a directed graph.

Analogy: Think of this as a decision flowchart a senior analyst follows:
  - Understand the question → Route it → Research → Cross-check → Synthesize → Review

Nodes = steps in the workflow
Edges = conditional transitions between steps
State = the "notepad" passed between every node
"""

import os
from typing import TypedDict, Annotated, Optional
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/vectorstore")

# ── Shared State Schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The notepad every node reads from and writes to.
    LangGraph passes this state through every node in the graph.
    """
    # Conversation
    messages: Annotated[list, add_messages]
    query: str

    # Routing decision
    query_type: Optional[str]       # "single_agency" | "multi_agency" | "split_rating" | "methodology"

    # Retrieved context
    retrieved_docs: list[Document]
    agencies_searched: list[str]

    # Analysis outputs
    individual_ratings: dict        # { "DBRS": "BBB(high)", "SP": "BBB+", ... }
    split_rating_detected: bool
    conflict_explanation: Optional[str]

    # Final output
    final_answer: Optional[str]
    needs_human_review: bool        # Triggers human-in-the-loop gate
    confidence_score: float         # 0.0 - 1.0


# ── LLM & Vector Store ─────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=temperature)

def get_retriever(filter_agency: Optional[str] = None):
    """
    Get a retriever, optionally filtered to one agency.
    filter_agency: "DBRS" | "FITCH" | "MCO" | "SP" | None (all agencies)
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name="credit_ratings",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    search_kwargs = {"k": 6}
    if filter_agency:
        search_kwargs["filter"] = {"agency": filter_agency}
    return vs.as_retriever(search_kwargs=search_kwargs)


# ── Node 1: Router ─────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Classifies the incoming query and decides which path to take.
    Analogy: The triage desk that routes patients to the right specialist.
    """
    llm = get_llm()

    system_prompt = """You are a credit rating query classifier. 
    Classify the user query into exactly one of these types:
    - single_agency: Asks about one specific agency's rating
    - multi_agency: Asks to compare ratings across multiple agencies  
    - split_rating: Asks about rating disagreements/divergence between agencies
    - methodology: Asks about rating criteria, frameworks, or methodology
    
    Respond with ONLY the type label, nothing else."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ])

    query_type = response.content.strip().lower()
    print(f"[Router] Query classified as: {query_type}")

    return {**state, "query_type": query_type}


# ── Node 2a: Single Agency RAG ─────────────────────────────────────────────────

def single_agency_rag_node(state: AgentState) -> AgentState:
    """
    Retrieve and answer from one agency's corpus.
    Used when user asks specifically about DBRS, Fitch, etc.
    """
    # Try to detect which agency the user mentioned
    query_lower = state["query"].lower()
    agency_map = {
        "dbrs": "DBRS", "morningstar": "DBRS",
        "fitch": "FITCH",
        "moody": "MCO", "moodys": "MCO",
        "s&p": "SP", "s and p": "SP", "standard": "SP"
    }
    target_agency = next(
        (code for keyword, code in agency_map.items() if keyword in query_lower),
        None  # Default: search all
    )

    retriever = get_retriever(filter_agency=target_agency)
    docs = retriever.invoke(state["query"])

    print(f"[Single Agency RAG] Retrieved {len(docs)} docs (agency: {target_agency or 'all'})")

    return {
        **state,
        "retrieved_docs": docs,
        "agencies_searched": [target_agency] if target_agency else ["ALL"],
    }


# ── Node 2b: Multi-Agency Comparator ──────────────────────────────────────────

def multi_agency_comparator_node(state: AgentState) -> AgentState:
    """
    Retrieve from ALL agencies and gather ratings for the same issuer.
    This is the workhorse for cross-agency comparison queries.
    """
    all_docs = []
    agencies_searched = []

    for agency_code in ["DBRS", "FITCH", "MCO", "SP"]:
        retriever = get_retriever(filter_agency=agency_code)
        docs = retriever.invoke(state["query"])
        all_docs.extend(docs)
        agencies_searched.append(agency_code)
        print(f"[Comparator] {agency_code}: {len(docs)} docs retrieved")

    return {
        **state,
        "retrieved_docs": all_docs,
        "agencies_searched": agencies_searched,
    }


# ── Node 3: Rating Extractor ───────────────────────────────────────────────────

def rating_extractor_node(state: AgentState) -> AgentState:
    """
    Use LLM to extract structured rating data from retrieved documents.
    Outputs a dict of {agency: rating_string} for the issuer in question.
    """
    if not state["retrieved_docs"]:
        return {**state, "individual_ratings": {}, "split_rating_detected": False}

    llm = get_llm()
    context = "\n\n---\n\n".join([
        f"[{doc.metadata.get('agency', 'UNKNOWN')}] {doc.page_content}"
        for doc in state["retrieved_docs"][:12]  # Cap context size
    ])

    extraction_prompt = f"""
    Based on the following credit rating documents, extract the current rating and outlook 
    for each agency that has rated the issuer mentioned in the query.
    
    Query: {state['query']}
    
    Documents:
    {context}
    
    Respond ONLY as valid JSON in this exact format:
    {{
      "issuer": "<issuer name>",
      "ratings": {{
        "DBRS": {{"rating": "<rating or null>", "outlook": "<outlook or null>"}},
        "FITCH": {{"rating": "<rating or null>", "outlook": "<outlook or null>"}},
        "MCO": {{"rating": "<rating or null>", "outlook": "<outlook or null>"}},
        "SP": {{"rating": "<rating or null>", "outlook": "<outlook or null>"}}
      }}
    }}
    """

    import json
    response = llm.invoke([HumanMessage(content=extraction_prompt)])

    try:
        extracted = json.loads(response.content)
        ratings = {
            agency: data["rating"]
            for agency, data in extracted["ratings"].items()
            if data["rating"]
        }
    except Exception:
        ratings = {}

    print(f"[Extractor] Extracted ratings: {ratings}")

    return {**state, "individual_ratings": ratings}


# ── Node 4: Split Rating Detector ─────────────────────────────────────────────

RATING_SCALE = {
    # Investment grade
    "AAA": 22, "Aaa": 22,
    "AA+": 21, "Aa1": 21,
    "AA":  20, "Aa2": 20,
    "AA-": 19, "Aa3": 19,
    "A+":  18, "A1":  18,
    "A":   17, "A2":  17,
    "A-":  16, "A3":  16,
    "BBB+": 15, "Baa1": 15,
    "BBB":  14, "Baa2": 14,
    "BBB-": 13, "Baa3": 13,
    # Sub-investment grade
    "BB+": 12, "Ba1": 12,
    "BB":  11, "Ba2": 11,
    "BB-": 10, "Ba3": 10,
    "B+":  9,  "B1":  9,
    "B":   8,  "B2":  8,
    "B-":  7,  "B3":  7,
}

def split_rating_detector_node(state: AgentState) -> AgentState:
    """
    Compare numerical rating equivalents across agencies.
    If gap >= 2 notches: significant split. 1 notch: minor split.
    This is the portfolio's signature feature — automates what analysts do manually.
    """
    ratings = state["individual_ratings"]
    if len(ratings) < 2:
        return {**state, "split_rating_detected": False, "conflict_explanation": None}

    scores = {
        agency: RATING_SCALE.get(rating, None)
        for agency, rating in ratings.items()
    }
    valid_scores = {k: v for k, v in scores.items() if v is not None}

    if len(valid_scores) < 2:
        return {**state, "split_rating_detected": False}

    max_score = max(valid_scores.values())
    min_score = min(valid_scores.values())
    gap = max_score - min_score

    split_detected = gap >= 1
    severity = "minor" if gap == 1 else "moderate" if gap == 2 else "significant"

    explanation = None
    if split_detected:
        high_agency = [a for a, s in valid_scores.items() if s == max_score][0]
        low_agency = [a for a, s in valid_scores.items() if s == min_score][0]
        explanation = (
            f"{high_agency} rates {gap} notch(es) higher than {low_agency}. "
            f"Severity: {severity}. Ratings: {ratings}"
        )
        print(f"[Split Detector] SPLIT DETECTED: {explanation}")

    return {
        **state,
        "split_rating_detected": split_detected,
        "conflict_explanation": explanation,
        "needs_human_review": split_detected and severity == "significant",
    }


# ── Node 5: Synthesizer ────────────────────────────────────────────────────────

def synthesizer_node(state: AgentState) -> AgentState:
    """
    Takes all gathered intelligence and writes the final analyst-quality response.
    Analogy: The senior analyst who reviews all the research notes and writes the memo.
    """
    llm = get_llm(temperature=0.2)

    context_parts = []
    if state["individual_ratings"]:
        context_parts.append(f"Extracted ratings: {state['individual_ratings']}")
    if state["conflict_explanation"]:
        context_parts.append(f"Split rating detected: {state['conflict_explanation']}")
    if state["retrieved_docs"]:
        doc_summaries = "\n".join([
            f"- [{doc.metadata.get('agency')}] {doc.page_content[:300]}..."
            for doc in state["retrieved_docs"][:6]
        ])
        context_parts.append(f"Source documents:\n{doc_summaries}")

    system_prompt = """You are a senior credit analyst with expertise across DBRS Morningstar, 
    Fitch, Moody's, and S&P. Provide clear, structured, professional responses.
    Always cite which agency's data you're referencing.
    Flag any rating divergences prominently.
    Be precise about rating levels and outlooks."""

    user_message = f"""
    User Question: {state['query']}
    
    Research gathered:
    {chr(10).join(context_parts)}
    
    Provide a thorough analyst-quality response.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    return {**state, "final_answer": response.content}


# ── Node 6: Human Review Gate ──────────────────────────────────────────────────

def human_review_gate_node(state: AgentState) -> AgentState:
    """
    Pauses execution and surfaces answer for human review.
    Critical for compliance use cases — mirrors MNPI review workflows.
    In a real system, this would write to a review queue / Slack / email.
    """
    print("\n" + "="*60)
    print("⚠️  HUMAN REVIEW REQUIRED — Significant split rating detected")
    print("="*60)
    print(f"Conflict: {state['conflict_explanation']}")
    print(f"\nDraft Answer:\n{state['final_answer']}")
    print("\n[In production: route to compliance review queue]")
    print("="*60 + "\n")

    # Tag the answer as pending review
    return {
        **state,
        "final_answer": f"[PENDING COMPLIANCE REVIEW]\n\n{state['final_answer']}"
    }


# ── Conditional Edge Logic ─────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Decides which retrieval node to go to after routing."""
    qt = state.get("query_type", "multi_agency")
    if qt in ("multi_agency", "split_rating"):
        return "multi_agency_comparator"
    elif qt == "methodology":
        return "single_agency_rag"  # Methodology docs are per-agency
    else:
        return "single_agency_rag"


def route_after_split_detection(state: AgentState) -> str:
    """Decides whether to go to human review or straight to synthesis."""
    if state.get("needs_human_review"):
        return "human_review_gate"
    return "synthesizer"


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Assemble the full LangGraph agent.
    Think of this as drawing the flowchart — nodes are boxes, edges are arrows.
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("single_agency_rag", single_agency_rag_node)
    graph.add_node("multi_agency_comparator", multi_agency_comparator_node)
    graph.add_node("rating_extractor", rating_extractor_node)
    graph.add_node("split_rating_detector", split_rating_detector_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("human_review_gate", human_review_gate_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional routing from router
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "single_agency_rag": "single_agency_rag",
            "multi_agency_comparator": "multi_agency_comparator",
        }
    )

    # Both retrieval paths converge at extractor
    graph.add_edge("single_agency_rag", "rating_extractor")
    graph.add_edge("multi_agency_comparator", "rating_extractor")

    # Extractor → Split detector
    graph.add_edge("rating_extractor", "split_rating_detector")

    # Conditional: significant split → human review, else → synthesize
    graph.add_conditional_edges(
        "split_rating_detector",
        route_after_split_detection,
        {
            "human_review_gate": "human_review_gate",
            "synthesizer": "synthesizer",
        }
    )

    # Human review feeds into synthesizer
    graph.add_edge("human_review_gate", "synthesizer")

    # Synthesizer is terminal
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── Entry Point ────────────────────────────────────────────────────────────────

def run_agent(query: str) -> str:
    """Run the full agent graph for a given query."""
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
    return result["final_answer"]


if __name__ == "__main__":
    # Example queries to test the graph
    test_queries = [
        "What is Ford Motor Company's current credit rating across all agencies?",
        "Has there been any rating divergence on Canadian bank bonds recently?",
        "What is DBRS Morningstar's methodology for rating covered bonds?",
    ]

    for q in test_queries[:1]:  # Run first query as smoke test
        print(f"\nQuery: {q}")
        print("-" * 60)
        answer = run_agent(q)
        print(f"Answer:\n{answer}")
