"""
Command Line Interface for the Research Paper Onboarding System.

Provides commands to rebuild the knowledge graph (ingest -> extract -> build)
and query the graph using a new paper abstract or question.
"""

import argparse
import sys
import os


def setup_parser() -> argparse.ArgumentParser:
    """Sets up the argument parser for the CLI.

    Returns:
        The configured ArgumentParser object.
    """
    parser = argparse.ArgumentParser(
        description="Research Paper Onboarding System - CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Rebuild subcommand
    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="Rerun ingestion -> extraction -> graph building pipeline"
    )
    rebuild_parser.add_argument(
        "--query",
        type=str,
        default="Multi-Agent LLM Orchestration Systems",
        help="Semantic Scholar search topic"
    )

    # Query subcommand
    query_parser = subparsers.add_parser(
        "query",
        help="Run reasoning walks on a new paper abstract or research question"
    )
    query_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The abstract text or research question to query"
    )

    return parser


def main() -> None:
    """Entry point of the CLI application."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "rebuild":
        print("==================================================")
        print("REBUILDING PIPELINE: Ingest -> Extract -> Graph")
        print("==================================================")
        try:
            # 1. Ingestion
            print("\n--- [Step 1/3] Running Ingestion ---")
            from ingest.query import run_ingest
            raw_output = "data/raw_papers.json"
            run_ingest(query=args.query, limit=100, output_path=raw_output)
            
            # 2. Extraction
            print("\n--- [Step 2/3] Running LLM Extraction ---")
            from extraction.run_extraction import run_extraction
            structured_output = "data/structured.json"
            run_extraction(input_path=raw_output, output_path=structured_output)
            
            # 3. Graph Building
            print("\n--- [Step 3/3] Building Knowledge Graph ---")
            from graph.build_graph import build_graph
            build_graph(
                input_path=structured_output,
                json_output_path="knowledge_state.json",
                pkl_output_path="knowledge_state.pkl"
            )
            
            print("\nPipeline rebuild completed successfully!")
            print("==================================================")
        except Exception as e:
            print("\n[ERROR] Pipeline execution failed!")
            print(f"Details: {str(e)}")
            print("Please check your network connection, API keys in .env, and try again.")
            print("==================================================")
            sys.exit(1)

    elif args.command == "query":
        pkl_path = "knowledge_state.pkl"
        if not os.path.exists(pkl_path):
            print(f"Error: Pickle file '{pkl_path}' not found. Please run 'python cli.py rebuild' first.")
            sys.exit(1)

        from reasoning.run_query import execute_query
        execute_query(abstract=args.input, title="CLI Query Abstract", pkl_path=pkl_path)


if __name__ == "__main__":
    main()
