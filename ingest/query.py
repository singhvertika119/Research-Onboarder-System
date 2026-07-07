"""
Script to query the Semantic Scholar API and save raw paper metadata.
"""

import os
import json
import argparse
from ingest.client import SemanticScholarClient


def run_ingest(query: str, limit: int = 100, output_path: str = "data/raw_papers.json") -> None:
    """Queries the Semantic Scholar API and saves the results to output_path.

    Args:
        query: The search query (e.g. 'Multi-Agent LLM Orchestration Systems').
        limit: The maximum number of papers to fetch.
        output_path: Path to save the JSON results.
    """
    print(f"Initializing Semantic Scholar query for: '{query}' (limit: {limit})")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    client = SemanticScholarClient()
    papers = client.fetch_papers_by_topic(query, limit=limit)
    
    print(f"Retrieved {len(papers)} papers. Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print("Ingestion completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Semantic Scholar and save raw papers.")
    parser.add_argument(
        "--query",
        type=str,
        default="Multi-Agent LLM Orchestration Systems",
        help="Search query to fetch papers for"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results to fetch"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_papers.json",
        help="Path to save raw JSON output"
    )
    args = parser.parse_args()
    run_ingest(args.query, limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()
