"""
Script to execute a query/reasoning task on the knowledge graph.
"""

import os
import pickle
import json
import logging
import argparse
from reasoning.engine import ReasoningEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def execute_query(
    abstract: str,
    title: str = "",
    pkl_path: str = "knowledge_state.pkl",
    output_path: str = ""
) -> None:
    """Loads the knowledge graph pickle, runs reasoning, and prints/saves structured output.

    Args:
        abstract: The paper abstract or research question.
        title: The paper title.
        pkl_path: Path to the binary pickle graph state.
        output_path: Path to save the output JSON.
    """
    if not os.path.exists(pkl_path):
        logger.error("Pickle graph file %s does not exist. Run graph build first.", pkl_path)
        return

    logger.info("Loading knowledge graph from %s...", pkl_path)
    with open(pkl_path, "rb") as f:
        graph = pickle.load(f)

    logger.info("Initializing ReasoningEngine...")
    engine = ReasoningEngine(graph)
    
    logger.info("Processing query...")
    result = engine.process_new_input(title, abstract)
    
    # Format and display the results
    print("\n" + "="*40)
    print("REASONING ENGINE RESULTS")
    print("="*40)
    print(json.dumps(result, indent=2))
    print("="*40 + "\n")

    if output_path:
        logger.info("Saving reasoning output to %s...", output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the MultiDiGraph reasoning engine.")
    parser.add_argument(
        "--abstract",
        type=str,
        default=(
            "We introduce AutoSOP, a method that improves Standard Operating Procedures in agent systems. "
            "While MetaGPT relies on manual SOP definition, AutoSOP automatically refines SOP rules. "
            "We build on the MetaGPT framework's Shared Blackboard Memory, but extend it by adding "
            "dynamic reinforcement feedback. Our approach competes with manual AutoGen configurations on complex tasks."
        ),
        help="The query abstract or question to analyze"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Agentic Workflow SOP Optimization",
        help="Optional title of the research paper"
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default="knowledge_state.pkl",
        help="Path to binary pickle state file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save output JSON"
    )
    args = parser.parse_args()
    execute_query(args.abstract, args.title, args.pkl, args.output)


if __name__ == "__main__":
    main()
