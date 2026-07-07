# Design Approach — Research Paper Onboarding System

This document outlines the architectural decisions, design schemas, and engineering tradeoffs made while building the `research-onboarder` system.

---

## 1. Selected Research Subset

*   **Topic Area**: Multi-Agent LLM Orchestration Systems.
*   **Selected Papers**: A dense subset of 6 seminal papers (CAMEL, ChatDev, AutoGen, MetaGPT, Dynabar, and GPTSwarm).
*   **Rationale**: 
    Relationships within a well-chosen subset are dense, chronological, and highly connected. By focusing on these 6 papers, we can map a clear evolution of ideas:
    1.  **CAMEL (2023)**: Introduces foundational role-playing communication.
    2.  **ChatDev (2023)**: Extends role-playing with sequential Software Development SOPs.
    3.  **AutoGen (2023)**: Formalizes conversational programming and conversable agent swarms.
    4.  **MetaGPT (2023)**: Improves on direct message passing by introducing structured Shared Blackboard Memory.
    5.  **Dynabar (2024)**: Addresses the cognitive overload limitations of MetaGPT's static blackboards via dynamic filters.
    6.  **GPTSwarm (2024)**: Solves the manual topological configuration constraints of AutoGen by automating graph optimization.

---

## 2. Entities and Relationships modeled

We modeled exactly the entities and relationships defined in the authoritative schema:

### Entities (Nodes)
*   `Paper`: Holds structural metadata (`title`, `year`, `abstract`, `core_contribution`).
*   `Author`: Normalized IDs (lowercase, underscores) mapping co-authorship loops.
*   `Venue`: Captures the publication medium (`NeurIPS`, `ACL`, `EMNLP`, `arXiv`).
*   `Method`: The core conceptual techniques (`Shared Blackboard Memory`, `Conversable Agents`).

### Relationships (Edges)
*   `CITES` (Paper → Paper): Establishes structural dependency in time.
*   `AUTHORED_BY` (Paper → Author) / `PUBLISHED_IN` (Paper → Venue): Capture metadata.
*   `INTRODUCES_METHOD` (Paper → Method): Links origin papers to techniques.
*   `APPLIES_METHOD` (Paper → Method) / `EXTENDS_METHOD` (Paper → Method): Represents downstream modifications.
*   `ADDRESSES_LIMITATION_OF` (Paper → Paper): The load-bearing edge. Explains the "why" behind the paper.
*   `COMPETES_WITH` (Method ↔ Method): Captures alternative pathways solving similar problems.

All entity and relationship extraction is performed via our custom-authored LLM prompt against the fixed schema, and no NER, spaCy, or automated knowledge-graph-construction libraries were used at any stage of development.

---

## 3. Knowledge Representation & Tradeoffs

*   **Graph Representation**: Built using `networkx.MultiDiGraph` to support typed edges and attributes.
*   **Tradeoff — Caching vs. Rate-Limits**: 
    Semantic Scholar rate-limits anonymous searches heavily. We built a local SQLite cache `paper_cache.db` to store API query outputs. If the API returns a 429 error, it gracefully falls back to loading our curated 6-paper dataset, ensuring the pipeline never breaks.
*   **Tradeoff — Pydantic Validation vs. LLM Flexibility**: 
    Strict Pydantic schemas guarantee database integrity, but fail the pipeline if the LLM returns slightly invalid JSON structures (e.g. putting `COMPETES_WITH` in the relationship literal field). We resolved this by adding a robust validation preprocessor in `extractor.py` to auto-correct literal mismatches before passing them to Pydantic.

---

## 4. Query-Time Reasoning Walk

When a new query/abstract is processed, the system walks the graph:
1.  **Strict Method Matching**: Normalizes name slugs. If it isn't an exact match, the engine issues a confirmatory LLM check against existing methods, preventing duplicate concept nodes.
2.  **Roadmap Generation (`suggested_reading_order`)**: 
    Creates a dependency DAG of matched papers. It maps edges if paper `B` extends or applies a method introduced by `A`, or if `B` cites `A`. It topological-sorts the DAG and computes the longest path (`chain_position`) for each node, sorting papers by `(chain_position, year)` to ensure root concepts are read before downstream extensions.
3.  **Open Gaps & Competitors**:
    Walks `ADDRESSES_LIMITATION_OF` and `COMPETES_WITH` edges. If no competitors exist in the graph, it returns a clear note rather than an empty list.

---

## 5. Next Steps

If we were to expand this system, we would build:
1.  **BFS Citation Crawler**: Enable the ingest client to fetch a paper, query its references/citations recursively, and auto-grow the graph from 6 to 100 papers on-the-fly.
2.  **LLM-Synthesized Roadmaps**: Feed the sorted reading list nodes into an LLM to generate a narrative onboarding guide explaining *why* the papers build on each other.
3.  **Interactive Graph Canvas**: Build a browser frontend using D3.js or Cytoscape.js to let researchers visually click, inspect, and trace the citation lanes.

### What We Chose Not to Build
*   **Automated Citation-Network Auto-Expansion**: Scoped out automatic recursive crawling beyond the seeded query because Semantic Scholar's public rate limits cause immediate request throttling without a paid institutional key.
*   **Frontend Visualization Layer**: Deferred building a web-based graph visualizer to focus resources on the core CLI reasoning engine and topological path validation logic.
*   **Full-Text Extraction Beyond Abstracts**: Excluded parsing complete PDFs because abstracts provide a highly structured, dense summary of core methodologies without the ingestion overhead and noise of full paper texts.
