# Research Paper Onboarding System (`research-onboarder`)

A Python-based citation network onboarding system designed to map, structure, and reason over academic publications in the **Multi-Agent LLM Orchestration Systems** domain. 

Given a new paper abstract or research question, the system walks the citation and method dependency graph to generate an optimal conceptual reading path, identify the closest prior work, surface open research gaps, and list competing approaches.

---

## 1. Directory Structure

```text
Research-Paper-Onboarding/
├── .venv/                         # Python 3.12 Virtual Environment
├── requirements.txt               # Pinned library dependencies
├── .env                           # Local environment keys (e.g. GROQ_API_KEY)
├── .env.example                   # Template file for environment setup
├── cli.py                         # Unified command-line interface (rebuild & query)
├── approach.md                    # Core architectural design and rationale
├── knowledge_state.json           # Serialized knowledge graph state (human-readable JSON)
├── knowledge_state.pkl            # Serialized binary graph state (networkx pickle)
├── ingest/
│   ├── __init__.py
│   ├── client.py                  # Semantic Scholar API client with SQLite cache
│   └── query.py                   # Ingestion pipeline coordinator
├── extraction/
│   ├── __init__.py
│   ├── extractor.py               # Pydantic-validated LLM extraction logic (Groq)
│   └── run_extraction.py          # Extraction pipeline coordinator
├── graph/
│   ├── __init__.py
│   ├── builder.py                 # MultiDiGraph builder (adds nodes and edges)
│   └── build_graph.py             # Graph serialization coordinator
└── reasoning/
    ├── __init__.py
    ├── engine.py                  # Graph walks, DAG sort, and limitation searches
    └── run_query.py               # Reasoning walk coordinator
```

---

## 2. Configuration & Environment Variables

This project requires a **Groq API Key** to perform LLM extraction and confirmatory matches.

1.  Copy the example environment file to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` and fill in your Groq credentials:
    ```text
    GROQ_API_KEY=gsk_your_groq_api_key_here
    ```

---

## 3. Installation & Setup

1.  **Initialize the Virtual Environment** (Python 3.12 is recommended):
    ```bash
    py -3.12 -m venv .venv
    ```
2.  **Activate the Environment**:
    *   **PowerShell (Windows)**:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   **Command Prompt (Windows)**:
        ```cmd
        .venv\Scripts\activate.bat
        ```
    *   **Bash/zsh (Mac/Linux)**:
        ```bash
        source .venv/bin/activate
        ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. How to Load or Regenerate the Knowledge State

To run the complete data pipeline (pull papers $\rightarrow$ extract semantic entities $\rightarrow$ build the graph), run the `rebuild` command:
```bash
python cli.py rebuild
```

### Ingestion Details
*   **Cache-First**: Raw API queries to Semantic Scholar are cached in a local SQLite database `paper_cache.db`.
*   **Curated Fallback**: Since the anonymous Semantic Scholar API rate-limits aggressively (HTTP 429), the ingest client will automatically load a high-quality pre-defined set of 6 foundational papers on Multi-Agent LLM Orchestration (CAMEL, ChatDev, AutoGen, MetaGPT, Dynabar, GPTSwarm) if the API blocks the request.

---

## 5. How to Run the Query Interface

Use the `query` command to feed a new abstract or research question into the system. The reasoning engine will extract methods, match them to existing graph nodes using confirmatory LLM synonym logic, and output a structured reading path.

### Example 1: Feed a New Paper Abstract
```bash
python cli.py query --input "We introduce AutoSOP, a method that improves Standard Operating Procedures in agent systems. While MetaGPT relies on manual SOP definition, AutoSOP automatically refines SOP rules. We build on the MetaGPT framework's Shared Blackboard Memory, but extend it by adding dynamic reinforcement feedback. Our approach competes with manual AutoGen configurations on complex tasks."
```

### Example 2: Feed a Research Question
```bash
python cli.py query --input "What are the limitations of MetaGPT's static blackboard memory and what competing approaches exist?"
```

### Output Schema format
The CLI returns a pretty-printed JSON response matching exactly the required schema:
```json
{
  "closest_prior_work": [
    "camel_2023",
    "chatdev_2023",
    "metagpt_2023",
    "dynabar_2024"
  ],
  "suggested_reading_order": [
    "camel_2023",
    "chatdev_2023",
    "metagpt_2023",
    "dynabar_2024"
  ],
  "known_limitations_in_this_space": [
    "dynabar_2024 solves limitation of metagpt_2023: 'MetaGPT's static shared blackboard memory'"
  ],
  "competing_approaches": [
    "no competing approaches recorded in current dataset"
  ]
}
```
*Note: If no competing methods exist in the graph database for the queried concepts, `competing_approaches` returns a descriptive fallback warning instead of an empty list.*
