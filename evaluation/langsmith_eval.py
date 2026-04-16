"""
LangSmith Evaluation Layer
===========================
This is your portfolio differentiator — most devs skip this.
Shows you think like a PM who actually measures quality.

Analogy: LangSmith is the restaurant's health inspector + training log.
Every dish (agent run) gets scored. Bad batches get analyzed. Recipes get improved.

Three things we measure:
1. Retrieval Precision — did we get the right documents?
2. Answer Faithfulness — did we hallucinate any rating data?
3. Comparative Accuracy — are cross-agency comparisons correct?
"""

import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig

load_dotenv()

client = Client()
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ── Ground Truth Dataset ───────────────────────────────────────────────────────
# Build this by hand from real DBRS/Fitch/S&P press releases you've verified.
# 20 Q&A pairs is enough to start. Quality >> quantity for evals.

EVAL_DATASET = [
    {
        "input": "What is the DBRS Morningstar rating for Royal Bank of Canada senior debt?",
        "expected_output": "AA (high)",  # Fill with actual current rating
        "notes": "Major Canadian bank — DBRS flagship issuer"
    },
    {
        "input": "Compare Fitch and S&P ratings for Ford Motor Company",
        "expected_output": "BB+ (both agencies, as of [date])",  # Fill with actuals
        "notes": "Auto sector split rating test case"
    },
    {
        "input": "Is there a split rating on any major US bank between Moody's and S&P?",
        "expected_output": "Yes — [specific example]",
        "notes": "Tests split rating detection logic"
    },
    # Add 17 more from real press releases you've read...
]


def create_eval_dataset(dataset_name: str = "credit-rating-ground-truth"):
    """
    Upload ground truth dataset to LangSmith.
    Run once to initialize — then reuse dataset_id for all future eval runs.
    """
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Ground truth Q&A pairs for credit rating agent evaluation"
    )

    examples = [
        {"inputs": {"query": item["input"]}, "outputs": {"answer": item["expected_output"]}}
        for item in EVAL_DATASET
    ]

    client.create_examples(inputs=[e["inputs"] for e in examples],
                           outputs=[e["outputs"] for e in examples],
                           dataset_id=dataset.id)

    print(f"Dataset created: {dataset.id}")
    return dataset.id


# ── Custom Evaluators ──────────────────────────────────────────────────────────

def faithfulness_evaluator(run, example) -> dict:
    """
    Checks: Does the answer make claims not supported by retrieved documents?
    This catches hallucinated rating levels — the worst failure mode for this app.
    """
    answer = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")

    prompt = f"""
    Evaluate if the following answer faithfully represents only what's in the source data,
    without fabricating rating levels or agency positions.
    
    Answer: {answer}
    Expected: {expected}
    
    Score 1 if faithful (no hallucinations), 0 if unfaithful.
    Respond ONLY with the score: 0 or 1
    """

    response = llm.invoke(prompt)
    score = int(response.content.strip())

    return {"key": "faithfulness", "score": score}


def rating_accuracy_evaluator(run, example) -> dict:
    """
    Checks: Are the specific rating levels (BBB, AA, etc.) mentioned correctly?
    Extracts rating strings from both answer and expected, then compares.
    """
    import re
    answer = run.outputs.get("answer", "").upper()
    expected = example.outputs.get("answer", "").upper()

    # Simple regex for common rating patterns
    rating_pattern = r'\b(AAA|AA[\+\-]?|A[\+\-]?|BBB[\+\-]?|BB[\+\-]?|B[\+\-]?|CCC[\+\-]?|Aaa|Aa[123]|A[123]|Baa[123]|Ba[123]|B[123])\b'

    answer_ratings = set(re.findall(rating_pattern, answer))
    expected_ratings = set(re.findall(rating_pattern, expected))

    if not expected_ratings:
        return {"key": "rating_accuracy", "score": 1}  # No specific rating to check

    overlap = answer_ratings & expected_ratings
    score = len(overlap) / len(expected_ratings)

    return {"key": "rating_accuracy", "score": score}


# ── Run Evaluation ─────────────────────────────────────────────────────────────

def run_evaluation(dataset_name: str = "credit-rating-ground-truth"):
    """
    Run the full eval suite against the agent.
    Results appear in LangSmith dashboard at smith.langchain.com
    """
    from agent.graph import run_agent

    def agent_wrapper(inputs: dict) -> dict:
        answer = run_agent(inputs["query"])
        return {"answer": answer}

    results = evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=[faithfulness_evaluator, rating_accuracy_evaluator],
        experiment_prefix="credit-agent-v1",
        metadata={"model": "gpt-4o", "version": "1.0"},
    )

    print("\n📊 Evaluation Results:")
    print(f"  Faithfulness: {results.summary_metrics.get('faithfulness', 'N/A'):.2%}")
    print(f"  Rating Accuracy: {results.summary_metrics.get('rating_accuracy', 'N/A'):.2%}")
    print(f"\nView full results at: https://smith.langchain.com")

    return results


# ── Prompt A/B Testing Setup ───────────────────────────────────────────────────

PROMPT_VARIANTS = {
    "analyst_persona": """You are a senior credit analyst with 15 years experience 
    at a major rating agency. Be precise, cite sources, flag divergences.""",

    "neutral_assistant": """You are a helpful assistant that answers questions about 
    credit ratings. Be accurate and clear.""",

    "compliance_focused": """You are a credit risk analyst at a regulated financial 
    institution. Every claim must be sourced. Flag any data gaps or conflicts explicitly.""",
}


def run_prompt_ab_test(query: str):
    """
    Test the same query across prompt variants.
    LangSmith tracks which variant produces better answers.
    Great portfolio talking point: 'I ran systematic prompt evals.'
    """
    from langchain_openai import ChatOpenAI
    from langsmith import traceable

    results = {}
    for variant_name, system_prompt in PROMPT_VARIANTS.items():

        @traceable(name=f"prompt_variant_{variant_name}", tags=[variant_name])
        def run_with_variant(q: str, sp: str) -> str:
            llm = ChatOpenAI(model="gpt-4o")
            from langchain_core.messages import SystemMessage, HumanMessage
            return llm.invoke([SystemMessage(content=sp), HumanMessage(content=q)]).content

        answer = run_with_variant(query, system_prompt)
        results[variant_name] = answer
        print(f"\n[{variant_name}]:\n{answer[:200]}...")

    return results


if __name__ == "__main__":
    print("Setting up LangSmith evaluation...")
    dataset_id = create_eval_dataset()
    print(f"\nNow run: python -m evaluation.langsmith_eval (after populating ground truth data)")
