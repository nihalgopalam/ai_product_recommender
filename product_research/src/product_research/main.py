#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from product_research.crew import ProductResearch

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def get_user_inputs() -> dict:
    """Prompt the user for topic and budget via manual entry."""
    print("\n=== AI Product Recommender ===\n")

    topic = input("Enter the product topic (e.g., 'Living room couch'): ").strip()
    if not topic:
        topic = "Living room couch"
        print(f"No topic entered. Using default: '{topic}'")

    budget_input = input("Enter your budget in USD (e.g., 500): ").strip()
    try:
        budget = float(budget_input)
    except (ValueError, TypeError):
        budget = 500.0
        print(f"Invalid budget. Using default: ${budget:.0f}")

    return {
        "topic": topic,
        "budget": budget,
        "current_year": str(datetime.now().year),
    }


def run():
    """
    Run the crew.
    """
    inputs = get_user_inputs()

    try:
        result = ProductResearch().crew().kickoff(inputs=inputs)
        print("\n\n=== Final Report ===\n")
        print(result.raw)
        print("\n\n Report saved to report.md ")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "Coach for living room",
        'current_year': str(datetime.now().year)
    }
    try:
        ProductResearch().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ProductResearch().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "Coach for living room",
        "current_year": str(datetime.now().year)
    }
    
    try:
        ProductResearch().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()