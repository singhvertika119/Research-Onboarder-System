"""
Script to build the knowledge graph from structured paper data.
"""

import os
import json
import logging
import argparse
from graph.builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_graph(
    input_path: str = "data/structured.json",
    json_output_path: str = "knowledge_state.json",
    pkl_output_path: str = "knowledge_state.pkl"
) -> None:
    """Loads structured JSON data, builds a networkx graph, and saves snapshots.

    Args:
        input_path: Path to the structured papers JSON.
        json_output_path: Path to save the human-readable JSON state.
        pkl_output_path: Path to save the binary pickle state.
    """
    if not os.path.exists(input_path):
        logger.error("Input file %s does not exist. Please run extraction first.", input_path)
        return

    logger.info("Reading structured data from %s...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        structured_papers = json.load(f)

    logger.info("Initializing GraphBuilder and building nodes...")
    builder = GraphBuilder()
    
    # Step 1: Add Paper, Author, Venue nodes and CITES/AUTHORED_BY/PUBLISHED_IN edges
    builder.add_paper_nodes(structured_papers)
    
    # Step 2: Add LLM-extracted Method nodes and relationship edges
    logger.info("Adding extracted relations and mapping connections...")
    builder.add_extracted_relations(structured_papers)
    
    # Step 3: Export graph snapshots
    logger.info("Exporting graph configurations...")
    builder.save_knowledge_state(json_output_path)
    builder.save_pickle(pkl_output_path)
    
    # Quick statistics
    g = builder.graph
    node_types = {}
    for _, d in g.nodes(data=True):
        t = d.get("type", "Unknown")
        node_types[t] = node_types.get(t, 0) + 1
        
    edge_types = {}
    for _, _, d in g.edges(data=True):
        t = d.get("type", "Unknown")
        edge_types[t] = edge_types.get(t, 0) + 1
        
    logger.info("Graph Statistics:")
    logger.info("  Nodes count: %d", g.number_of_nodes())
    for nt, count in node_types.items():
        logger.info("    - %s: %d", nt, count)
        
    logger.info("  Edges count: %d", g.number_of_edges())
    for et, count in edge_types.items():
        logger.info("    - %s: %d", et, count)

    logger.info("Graph build completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and export the MultiDiGraph from structured JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/structured.json",
        help="Path to structured papers JSON file"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="knowledge_state.json",
        help="Path to save human-readable JSON state"
    )
    parser.add_argument(
        "--pkl-output",
        type=str,
        default="knowledge_state.pkl",
        help="Path to save binary pickle state"
    )
    args = parser.parse_args()
    build_graph(args.input, args.json_output, args.pkl_output)


if __name__ == "__main__":
    main()
