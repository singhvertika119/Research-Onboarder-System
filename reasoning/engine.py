"""
Reasoning Engine that walks the networkx knowledge graph to answer queries.

Matches a new abstract or query to the graph, calculates logical reading orders,
surfaces limitations, and identifies competing approaches.
"""

from typing import Any, Dict, List
import networkx as nx


import os
import logging
from typing import Any, Dict, List, Optional
import networkx as nx
from openai import OpenAI
from extraction.extractor import PaperExtractor
from graph.builder import slugify

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Performs graph-based walks and reasoning logic over the structured knowledge graph."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        """Initializes the reasoning engine with a built knowledge graph.

        Args:
            graph: The populated networkx MultiDiGraph of papers and methods.
        """
        self.graph = graph
        self.extractor = PaperExtractor()
        
        # Groq client for confirmatory matching
        key = os.environ.get("GROQ_API_KEY")
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key
        )
        self.model = "llama-3.1-8b-instant"

    def process_new_input(self, title: str, abstract: str) -> Dict[str, Any]:
        """Processes a new, unseen paper abstract to perform onboarding reasoning.

        1. Extract candidate methods from the new input via the extraction module.
        2. Match candidate methods to existing graph Method nodes.
        3. Walk the graph from matched nodes to compile reading order, limitations,
           and competing approaches.

        Args:
            title: The new paper's title.
            abstract: The new paper's abstract.

        Returns:
            A structured dictionary containing closest prior work, reading order,
            open gaps, and competing approaches.
        """
        logger.info("Extracting candidate methods from new input...")
        extracted = self.extractor.extract_from_paper(title, abstract)
        if not extracted:
            logger.warning("Failed to extract schema fields from query. Using direct keyword match.")
            extracted = {"methods": [{"name": title or abstract, "description": abstract}], "competes_with": []}

        candidate_methods = extracted.get("methods", [])
        logger.info("Extracted %d candidate methods: %s", len(candidate_methods), [m["name"] for m in candidate_methods])

        # Match to existing methods
        matched_method_ids = []
        for cand in candidate_methods:
            match_id = self.match_method_to_graph(cand["name"], cand.get("description", ""))
            if match_id:
                logger.info("Matched candidate '%s' to graph node '%s'", cand["name"], match_id)
                matched_method_ids.append(match_id)

        # Walk graph to collect papers, limitations, and competing approaches
        closest_prior_work = set()
        suggested_reading_order = []
        open_gaps = set()
        competing_approaches = set()

        for m_id in matched_method_ids:
            # 1. Pull Papers connected to this method
            # Look for incoming edges to this method node (Paper -> Method)
            related_papers = []
            for u, v, key, data in self.graph.in_edges(m_id, keys=True, data=True):
                if self.graph.nodes[u].get("type") == "Paper":
                    rel_type = data.get("type")
                    paper_data = self.graph.nodes[u]
                    related_papers.append({
                        "id": u,
                        "year": paper_data.get("year", 0),
                        "relationship": rel_type
                    })
                    closest_prior_work.add(u)

            # Sort papers chronologically
            related_papers.sort(key=lambda x: x["year"])
            for p in related_papers:
                if p["id"] not in suggested_reading_order:
                    suggested_reading_order.append(p["id"])

            # 2. Pull ADDRESSES_LIMITATION_OF edges touching those papers
            for p_id in list(closest_prior_work):
                # Outgoing limitation edges: this paper addresses limitation of target paper
                for u, v, key, data in self.graph.out_edges(p_id, keys=True, data=True):
                    if data.get("type") == "ADDRESSES_LIMITATION_OF":
                        reason = data.get("reason", "")
                        target_title = self.graph.nodes[v].get("title", v)
                        open_gaps.add(f"{p_id} solves limitation of {v}: '{reason}'")

                # Incoming limitation edges: target paper addresses limitation of this paper
                for u, v, key, data in self.graph.in_edges(p_id, keys=True, data=True):
                    if data.get("type") == "ADDRESSES_LIMITATION_OF":
                        reason = data.get("reason", "")
                        open_gaps.add(f"{u} solves limitation of {p_id}: '{reason}'")

            # 3. Pull COMPETES_WITH edges from method node
            for u, v, key, data in self.graph.edges(m_id, keys=True, data=True):
                if data.get("type") == "COMPETES_WITH":
                    comp_name = self.graph.nodes[v].get("name", v)
                    competing_approaches.add(comp_name)

        # Deduplicate and sort reading order based on chronological dependency
        final_reading_order = self.compute_reading_order(list(closest_prior_work))

        # Compile output structure matching both user requests
        competing_list = list(competing_approaches)
        if not competing_list:
            competing_list = ["no competing approaches recorded in current dataset"]

        return {
            "closest_prior_work": list(closest_prior_work),
            "suggested_reading_order": final_reading_order,
            "known_limitations_in_this_space": list(open_gaps),
            "competing_approaches": competing_list
        }

    def match_method_to_graph(self, method_name: str, method_description: str = "") -> Optional[str]:
        """Helper to match a candidate method name to an existing Method node ID.

        Utilizes slug exact match, substring normalization, and confirmatory LLM lookup.

        Args:
            method_name: Candidate method name.
            method_description: Optional description.

        Returns:
            The matched Method node ID or None if no match is found.
        """
        cand_slug = slugify(method_name)
        
        # 1. Check exact match in graph
        if self.graph.has_node(cand_slug) and self.graph.nodes[cand_slug].get("type") == "Method":
            return cand_slug

        # Get all method nodes in graph
        existing_methods = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "Method":
                existing_methods[node_id] = data.get("name", "")

        if not existing_methods:
            return None

        # 3. LLM confirmatory lookup
        logger.info("Performing LLM confirmatory match for candidate method '%s'...", method_name)
        existing_list_str = "\n".join([f"- ID: {nid} (Name: {name})" for nid, name in existing_methods.items()])
        prompt = f"""You are matching a newly extracted method to a list of existing method nodes in a knowledge graph.
Candidate Method: {method_name} (Description: {method_description})

Existing Methods in Graph:
{existing_list_str}

Task: Determine if the Candidate Method is a synonym, minor variation, or semantic equivalent of any of the Existing Methods.
If it is equivalent, return ONLY the matching exact ID from the list.
If it is a completely new method not represented in the list, return 'None'.

Output only the matched ID or 'None', with no other text, quote marks, or explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise lookup assistant. Output only the matched ID or 'None'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            matched_id = response.choices[0].message.content.strip().strip("'\"")
            if matched_id in existing_methods:
                return matched_id
        except Exception as e:
            logger.error("LLM confirmatory match failed: %s", str(e))

        return None

    def compute_reading_order(self, paper_ids: List[str]) -> List[str]:
        """Calculates a topological/chronological reading path of papers.

        Explicitly sorts papers by their position in the INTRODUCES_METHOD -> EXTENDS_METHOD
        dependency chain and by their publication year, ensuring foundational papers are read first.

        Args:
            paper_ids: List of paper IDs.

        Returns:
            List of paper IDs ordered logically.
        """
        if not paper_ids:
            return []

        # Build a dependency DAG of papers to compute chain positions
        dep_graph = nx.DiGraph()
        dep_graph.add_nodes_from(paper_ids)

        # 1. Add method-based dependencies (INTRODUCES_METHOD -> EXTENDS_METHOD/APPLIES_METHOD chain)
        for p_id in paper_ids:
            # Find all methods extended or applied by this paper
            for u, v, key, data in self.graph.out_edges(p_id, keys=True, data=True):
                rel_type = data.get("type")
                if rel_type in ("EXTENDS_METHOD", "APPLIES_METHOD"):
                    # v is the Method node. Find the paper that introduces v
                    intro_paper = self.graph.nodes[v].get("first_introduced_by")
                    if intro_paper and intro_paper in paper_ids and intro_paper != p_id:
                        # The introducing paper must be read before the extending/applying paper
                        dep_graph.add_edge(intro_paper, p_id)

        # 2. Add citation-based dependencies (CITES)
        subgraph = self.graph.subgraph(paper_ids)
        for u, v, data in subgraph.edges(data=True):
            if data.get("type") == "CITES":
                # u cites v (u is newer, v is older). Dependency: v must be read before u
                if u in paper_ids and v in paper_ids:
                    dep_graph.add_edge(v, u)

        # 3. Compute chain positions (longest path from any root node to P in the DAG)
        chain_position = {p: 0 for p in paper_ids}
        
        try:
            # Topological sort of our dependency DAG to compute paths iteratively without recursion errors
            topo_order = list(nx.topological_sort(dep_graph))
            for node in topo_order:
                for successor in dep_graph.successors(node):
                    chain_position[successor] = max(
                        chain_position[successor],
                        chain_position[node] + 1
                    )
        except nx.NetworkXUnfeasible:
            # If a cycle exists, log a warning and fall back to chain_position = 0 (relying on year sort)
            logger.warning("Dependency graph contains cycles. Falling back to year-only sorting.")
            chain_position = {p: 0 for p in paper_ids}

        # 4. Extract years
        years = {p: self.graph.nodes[p].get("year", 0) for p in paper_ids}

        # Sort key logic:
        # Primary: position in the method/citation dependency chain (root papers first).
        # Secondary: publication year (older papers first within the same chain position level).
        def sort_key(p_id: str) -> tuple[int, int]:
            return (chain_position[p_id], years[p_id])

        sorted_order = sorted(paper_ids, key=sort_key)
        return sorted_order

