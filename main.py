"""
CLI entry point for the Credit Rating Intelligence Agent.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure LangSmith tracing is active for every run
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "credit-rating-agent")

from agent.graph import run_agent


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Credit Rating Intelligence Agent")
        print("=" * 50)
        query = input("Enter your query: ").strip()

    if not query:
        print("No query provided.")
        sys.exit(1)

    print(f"\nProcessing: {query}\n")
    answer = run_agent(query)
    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(answer)


if __name__ == "__main__":
    main()
