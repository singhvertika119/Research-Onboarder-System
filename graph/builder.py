"""
Knowledge Graph Builder utilizing NetworkX.

Constructs typed nodes and typed edges from the ingested and extracted research paper metadata.
Exports the serialized graph to a human-readable JSON state and a pickle.
"""

from typing import Any, Dict
import networkx as nx


import re
import pickle
import json
import logging
from typing import Any, Dict, List, Optional
import networkx as nx

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Normalizes string to lowercase slug format with hyphens."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def author_slug(text: str) -> str:
    """Normalizes author name to lowercase with underscores and no punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_")


class GraphBuilder:
    """Builds, populates, and serializes the citation and method knowledge graph."""

    def __init__(self) -> None:
        """Initializes an empty networkx.MultiDiGraph."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_paper_nodes(self, papers: List[Dict[str, Any]]) -> None:
        """Adds Paper, Author, and Venue nodes to the graph.

        Creates structural links (AUTHORED_BY, PUBLISHED_IN, CITES) directly
        from the ingest metadata.

        Args:
            papers: Ingested paper dictionaries.
        """
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("id")
            if not paper_id:
                continue

            # 1. Add Paper Node
            self.graph.add_node(
                paper_id,
                type="Paper",
                id=paper_id,
                title=paper.get("title", ""),
                year=paper.get("year"),
                abstract=paper.get("abstract", ""),
                venue=paper.get("venue", ""),
                core_contribution=paper.get("core_contribution", "")
            )

            # 2. Add Author Nodes and AUTHORED_BY Edges
            for author in paper.get("authors", []):
                disp_name = author.get("name", "")
                auth_id = author.get("authorId") or author_slug(disp_name)
                if not auth_id:
                    continue

                if not self.graph.has_node(auth_id):
                    self.graph.add_node(auth_id, type="Author", id=auth_id, name=disp_name)

                self.graph.add_edge(paper_id, auth_id, key="AUTHORED_BY", type="AUTHORED_BY")

            # 3. Add Venue Node and PUBLISHED_IN Edge
            venue_name = paper.get("venue")
            if venue_name:
                ven_id = slugify(venue_name)
                if not self.graph.has_node(ven_id):
                    self.graph.add_node(ven_id, type="Venue", id=ven_id, name=venue_name)

                self.graph.add_edge(paper_id, ven_id, key="PUBLISHED_IN", type="PUBLISHED_IN")

        # 4. Add CITES Edges (Only if both papers are present in the graph)
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("id")
            if not paper_id:
                continue

            # Add edges from references (paper_id -> cited_id)
            for ref in paper.get("references", []):
                ref_id = ref.get("paperId") or ref.get("id")
                if ref_id and self.graph.has_node(ref_id):
                    if not self.graph.has_edge(paper_id, ref_id, key="CITES"):
                        self.graph.add_edge(paper_id, ref_id, key="CITES", type="CITES")

            # Add edges from citations (citing_id -> paper_id)
            for cite in paper.get("citations", []):
                cite_id = cite.get("paperId") or cite.get("id")
                if cite_id and self.graph.has_node(cite_id):
                    if not self.graph.has_edge(cite_id, paper_id, key="CITES"):
                        self.graph.add_edge(cite_id, paper_id, key="CITES", type="CITES")

    def _find_target_paper(self, target_text: str, papers: List[Dict[str, Any]]) -> Optional[str]:
        """Helper to match a limitation text description to a paper ID."""
        target_lower = target_text.lower()
        for p in papers:
            p_id = p.get("paperId") or p.get("id", "")
            p_base = p_id.split("_")[0]  # e.g. metagpt_2023 -> metagpt
            if p_base in target_lower or p_id.lower() in target_lower:
                return p_id
            
            # Substring match on title (excluding very short words)
            title_words = [w.lower() for w in p.get("title", "").split() if len(w) > 3]
            if title_words and any(w in target_lower for w in title_words):
                return p_id
        return None

    def _find_method_node(self, method_name: str) -> Optional[str]:
        """Helper to find a Method node ID by name or slug match."""
        m_slug = slugify(method_name)
        if self.graph.has_node(m_slug):
            return m_slug
        
        # Substring matching in name
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "Method":
                node_name = data.get("name", "").lower()
                if m_slug in node_name or node_name in m_slug:
                    return node_id
        return None

    def add_extracted_relations(self, papers: List[Dict[str, Any]]) -> None:
        """Adds extracted Method nodes and semantic relationships.

        Also maps targets_limitation_of and competes_with relationships.

        Args:
            papers: Structured JSON paper dictionaries.
        """
        # First, add all Method nodes and their immediate paper relations
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("id")
            if not paper_id:
                continue

            for method in paper.get("methods", []):
                m_name = method.get("name", "")
                m_id = slugify(m_name)
                if not m_id:
                    continue

                # Add Method node if not exists
                if not self.graph.has_node(m_id):
                    self.graph.add_node(
                        m_id,
                        type="Method",
                        id=m_id,
                        name=m_name,
                        description=method.get("description", ""),
                        first_introduced_by=None
                    )

                rel = method.get("relationship")
                if rel == "INTRODUCES_METHOD":
                    self.graph.nodes[m_id]["first_introduced_by"] = paper_id
                    self.graph.add_edge(paper_id, m_id, key="INTRODUCES_METHOD", type="INTRODUCES_METHOD")
                elif rel == "APPLIES_METHOD":
                    self.graph.add_edge(paper_id, m_id, key="APPLIES_METHOD", type="APPLIES_METHOD")
                elif rel == "EXTENDS_METHOD":
                    self.graph.add_edge(paper_id, m_id, key="EXTENDS_METHOD", type="EXTENDS_METHOD")

        # Second, map limitation and competition edges
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("id")
            if not paper_id:
                continue

            for method in paper.get("methods", []):
                targets_lim = method.get("targets_limitation_of")
                if targets_lim:
                    target_paper_id = self._find_target_paper(targets_lim, papers)
                    if target_paper_id:
                        logger.info("Mapped limitation: Paper %s -> Paper %s (via %s)", paper_id, target_paper_id, targets_lim)
                        self.graph.add_edge(
                            paper_id,
                            target_paper_id,
                            key="ADDRESSES_LIMITATION_OF",
                            type="ADDRESSES_LIMITATION_OF",
                            reason=targets_lim
                        )

            # Map competing methods
            competes_list = paper.get("competes_with", [])
            for comp_name in competes_list:
                comp_id = self._find_method_node(comp_name)
                if comp_id:
                    # Find methods introduced or applied by this paper to connect
                    paper_methods = [
                        u for u, v, d in self.graph.out_edges(paper_id, data=True)
                        if d.get("type") in ("INTRODUCES_METHOD", "EXTENDS_METHOD", "APPLIES_METHOD")
                        and self.graph.nodes[v].get("type") == "Method"
                    ]
                    for pm_id in paper_methods:
                        # Add bidirectional COMPETES_WITH edges
                        if not self.graph.has_edge(pm_id, comp_id, key="COMPETES_WITH"):
                            self.graph.add_edge(pm_id, comp_id, key="COMPETES_WITH", type="COMPETES_WITH")
                        if not self.graph.has_edge(comp_id, pm_id, key="COMPETES_WITH"):
                            self.graph.add_edge(comp_id, pm_id, key="COMPETES_WITH", type="COMPETES_WITH")

    def save_knowledge_state(self, filepath: str = "knowledge_state.json") -> None:
        """Exports the graph snapshot as a human-readable JSON file.

        Args:
            filepath: Path to write the output JSON.
        """
        nodes_list = []
        for node_id, data in self.graph.nodes(data=True):
            nodes_list.append({
                "id": node_id,
                "type": data.get("type"),
                "properties": {k: v for k, v in data.items() if k != "type"}
            })

        edges_list = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edges_list.append({
                "source": u,
                "target": v,
                "type": data.get("type"),
                "properties": {k: v for k, v in data.items() if k != "type"}
            })

        state = {
            "nodes": nodes_list,
            "edges": edges_list
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info("Saved knowledge state to %s", filepath)

    def load_knowledge_state(self, filepath: str = "knowledge_state.json") -> None:
        """Restores the graph state from a human-readable JSON snapshot.

        Args:
            filepath: Path to read the input JSON.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.graph.clear()
        for node in state.get("nodes", []):
            node_id = node["id"]
            attrs = node.get("properties", {})
            attrs["type"] = node["type"]
            self.graph.add_node(node_id, **attrs)

        for edge in state.get("edges", []):
            u = edge["source"]
            v = edge["target"]
            attrs = edge.get("properties", {})
            attrs["type"] = edge["type"]
            self.graph.add_edge(u, v, key=edge["type"], **attrs)
        logger.info("Loaded knowledge state from %s", filepath)

    def save_pickle(self, filepath: str = "knowledge_state.pkl") -> None:
        """Saves a binary pickle version of the networkx graph for quick load.

        Args:
            filepath: Path to write the binary pickle.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info("Saved graph pickle to %s", filepath)

