"""
LangSmith Evaluation Layer
===========================
Updated for LangSmith SDK 0.2+
"""

import os
import re
from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

load_dotenv()

client = Client()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Ground Truth Dataset ───────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "input": "What is the S&P credit rating for the United States?",
        "expected_output": "AA+",
        "notes": "Downgraded from AAA in 2023"
    },
    {
        "input": "Has Moody's taken any recent action on Thailand's banking sector?",
        "expected_output": "Moody's cut Thai banks outlook to negative",
        "notes": "Recent action in ingested data"
    },
    {
        "input": "What is S&P's rating on Saudi Arabia?",
        "expected_output": "A+",
        "notes": "From ingested S&P feed"
    },
    {
        "input": "What did Fitch say about US credit rating?",
        "expected_output": "AA+, rising debt a ratings constraint",
        "notes": "From ingested Fitch feed"
    },
    {
        "input": "What is S&P's outlook on Bahrain?",
        "expected_output": "Negative outlook due to weak finances",
        "notes": "From ingested S&P feed"
    },
]


def create_eval_dataset(dataset_name: str = "credit-rating-ground-truth"):
    """Upload ground truth dataset to LangSmith."""
    try:
        # Check if dataset already exists
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if datasets:
            print(f"Dataset already exists: {datasets[0].id}")
            return datasets[0].id
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Ground truth Q&A pairs for credit rating agent evaluation"
    )

    client.create_examples(
        inputs=[{"query": item["input"]} for item in EVAL_DATASET],
        outputs=[{"answer": item["expected_output"]} for item in EVAL_DATASET],
        dataset_id=dataset.id
    )

    print(f"Dataset created: {dataset.id}")
    return dataset.id


# ── Custom Evaluators (updated LangSmith 0.2+ API) ────────────────────────────

def faithfulness_evaluator(run, example) -> dict:
    """Checks for hallucinated rating levels."""
    answer = run.outputs.get("answer", "") if run.outputs else ""
    expected = example.outputs.get("answer", "") if example.outputs else ""

    prompt = f"""Does this answer contain only factual claims without fabricating rating levels?
Answer: {answer}
Expected: {expected}
Respond with only: 1 (faithful) or 0 (contains hallucinations)"""

    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip()[0])
    except Exception:
        score = 0

    return {"key": "faithfulness", "score": score}


def rating_accuracy_evaluator(run, example) -> dict:
    """Checks if rating levels mentioned are correct."""
    answer = (run.outputs.get("answer", "") if run.outputs else "").upper()
    expected = (example.outputs.get("answer", "") if example.outputs else "").upper()

    rating_pattern = r'\b(AAA|AA[\+\-]?|A[\+\-]?|BBB[\+\-]?|BB[\+\-]?|B[\+\-]?|Aaa|Aa[123]|A[123]|Baa[123]|Ba[123])\b'
    answer_ratings = set(re.findall(rating_pattern, answer))
    expected_ratings = set(re.findall(rating_pattern, expected))

    if not expected_ratings:
        return {"key": "rating_accuracy", "score": 1}

    overlap = answer_ratings & expected_ratings
    score = len(overlap) / len(expected_ratings)
    return {"key": "rating_accuracy", "score": score}


# ── Run Evaluation ─────────────────────────────────────────────────────────────

def run_evaluation(dataset_name: str = "credit-rating-ground-truth"):
    """Run full eval suite. Results appear in LangSmith dashboard."""
    from agent.graph import run_agent

    def agent_wrapper(inputs: dict) -> dict:
        answer = run_agent(inputs["query"])
        return {"answer": answer}

    # Create dataset if it doesn't exist
    create_eval_dataset(dataset_name)

    results = evaluate(
        agent_wrapper,
        data=dataset_name,
        evaluators=[faithfulness_evaluator, rating_accuracy_evaluator],
        experiment_prefix="credit-agent-v1",
        metadata={"model": "gpt-4o", "version": "1.0"},
    )

    print("\n📊 Evaluation complete — view at https://smith.langchain.com")
    return results


# ── Prompt A/B Testing ─────────────────────────────────────────────────────────

PROMPT_VARIANTS = {
    "analyst_persona": "You are a senior credit analyst with 15 years experience. Be precise, cite sources, flag divergences.",
    "neutral_assistant": "You are a helpful assistant that answers questions about credit ratings accurately.",
    "compliance_focused": "You are a credit risk analyst at a regulated institution. Every claim must be sourced. Flag data gaps explicitly.",
}


def run_prompt_ab_test(query: str) -> dict:
    """Test the same query across prompt variants, tracked in LangSmith."""
    from langchain_core.messages import SystemMessage, HumanMessage

    results = {}
    for variant_name, system_prompt in PROMPT_VARIANTS.items():

        @traceable(name=f"prompt_ab_{variant_name}", tags=[variant_name])
        def run_variant(q: str, sp: str) -> str:
            return llm.invoke([
                SystemMessage(content=sp),
                HumanMessage(content=q)
            ]).content

        results[variant_name] = run_variant(query, system_prompt)

    return results


if __name__ == "__main__":
    print("Creating eval dataset...")
    create_eval_dataset()
    print("Done. Run run_evaluation() to score the agent.")
