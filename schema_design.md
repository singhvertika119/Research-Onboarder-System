# Knowledge Schema Design — Research Paper Onboarding System
**Domain:** B — Research Paper Onboarding
**Topic:** Multi-Agent LLM Orchestration Systems
**Author:** Vertika Singh

This document is the authoritative schema. Antigravity should implement exactly this — no additional entity types, relationship types, or auto-extraction libraries (spaCy, NER pipelines, LangChain graph extractors, etc.) are permitted. All extraction logic must call an LLM using the prompt template defined here, with outputs validated against this schema in code.

---

## 1. Entities

### 1.1 Paper
```json
{
  "id": "string — Semantic Scholar paperId",
  "title": "string",
  "year": "int",
  "abstract": "string",
  "venue": "string",
  "core_contribution": "string — one sentence, what is NEW in this paper (not a summary of topic)"
}
```

### 1.2 Author
```json
{
  "id": "string — normalized name, lowercase, no punctuation",
  "name": "string — display name"
}
```

### 1.3 Method
A Method is a specific, nameable technique, architecture, or design pattern for multi-agent LLM systems. Generic terms ("LLM", "agents", "prompting") do NOT qualify as Methods.
```json
{
  "id": "string — slug, e.g. 'shared-blackboard-memory'",
  "name": "string",
  "description": "string",
  "first_introduced_by": "paper_id — set only when relationship = INTRODUCES_METHOD"
}
```

### 1.4 Venue
```json
{
  "id": "string",
  "name": "string"
}
```

---

## 2. Relationships

| Relationship | From → To | Meaning | Extraction rule |
|---|---|---|---|
| `CITES` | Paper → Paper | Direct citation, pulled from Semantic Scholar reference data | Structural, not LLM-extracted |
| `AUTHORED_BY` | Paper → Author | Authorship | Structural, not LLM-extracted |
| `PUBLISHED_IN` | Paper → Venue | Publication venue | Structural, not LLM-extracted |
| `INTRODUCES_METHOD` | Paper → Method | This paper originates the method | LLM-extracted per paper |
| `APPLIES_METHOD` | Paper → Method | Uses a method without originating it | LLM-extracted per paper |
| `EXTENDS_METHOD` | Paper → Method | Modifies/improves a named prior method | LLM-extracted per paper |
| `ADDRESSES_LIMITATION_OF` | Paper → Paper | Explicitly states it solves a stated weakness of prior work | LLM-extracted; only if paper says this explicitly (intro/related work) — never inferred |
| `COMPETES_WITH` | Method ↔ Method | Two methods solve the same problem via different approaches | LLM-extracted; only if paper directly compares them |

`ADDRESSES_LIMITATION_OF` and `COMPETES_WITH` are the load-bearing relationships — they are what let the system reason ("why was this built," "what are the alternatives") rather than just retrieve citations.

---

## 3. Extraction Pipeline (per paper)

**Step 1 — Structural relationships** (no LLM call, pulled directly from Semantic Scholar API response):
`CITES`, `AUTHORED_BY`, `PUBLISHED_IN`

**Step 2 — LLM extraction call** (see Section 4 for exact prompt) produces:
`core_contribution`, candidate `Method` entities, `INTRODUCES_METHOD` / `APPLIES_METHOD` / `EXTENDS_METHOD`, `ADDRESSES_LIMITATION_OF`, `COMPETES_WITH`

**Step 3 — Method deduplication pass** (manual/semi-manual, run once over the full method list after all papers are processed):
LLMs will name the same underlying idea differently across papers (e.g. "shared blackboard memory" vs "central shared memory store"). This pass is a deliberate modeling decision, not automated — merge near-duplicate Method entities by hand or with a single confirmatory LLM call per candidate pair ("are these the same method: X vs Y?"), and log every merge decision for `approach.md`.

**Step 4 — Graph build:** load all entities/relationships into a `networkx.MultiDiGraph`, typed nodes and typed edges.

**Step 5 — Export:** serialize to `knowledge_state.json` (human-readable) + a `networkx` pickle for internal use.

---

## 4. LLM Extraction Prompt Template

Used once per paper in Step 2. Must be called with a fixed system-style instruction — do not let the model freelance on schema.

```
You are extracting structured knowledge from a single research paper according to a FIXED schema. Do not invent entity or relationship types outside this schema. If something doesn't fit, omit it rather than forcing it.

ENTITY TYPES:
- Method: a named technique, architecture, or design pattern for multi-agent LLM systems (must be a specific, nameable approach — not a vague topic like "LLM" or "agents")

RELATIONSHIP TYPES (only extract these, from THIS paper's perspective):
- INTRODUCES_METHOD: this paper originates a method
- APPLIES_METHOD: this paper uses a method it did not originate
- EXTENDS_METHOD: this paper modifies/improves a named prior method
- ADDRESSES_LIMITATION_OF: this paper explicitly states it solves a weakness in prior work (only extract if the paper says this explicitly, e.g. in intro/related work — do not infer)
- COMPETES_WITH: this paper's method and another named method solve the same problem via different approaches (only if the paper directly compares them)

INPUT:
Title: {title}
Abstract: {abstract}

TASK:
1. Identify 0-3 Methods discussed in this paper (only genuinely named/definable techniques)
2. For each Method, state the relationship type from the list above
3. If ADDRESSES_LIMITATION_OF applies, name which specific prior work/approach (by description if not exact title) it targets
4. Write core_contribution as ONE sentence: what is new here, not what the paper is about generally

OUTPUT (valid JSON only, no prose, no markdown fences):
{
  "core_contribution": "string",
  "methods": [
    {
      "name": "string",
      "description": "string",
      "relationship": "INTRODUCES_METHOD | APPLIES_METHOD | EXTENDS_METHOD",
      "targets_limitation_of": "string or null"
    }
  ],
  "competes_with": ["method_name, if explicitly compared"]
}
```

Validate every response against this JSON shape in code before writing to `structured.json`. Log and skip papers that fail validation twice.

---

## 5. Reasoning Over New Input

When a user submits a new abstract or research question (not in the original dataset):

1. Run the **same** extraction prompt template on the new input to identify candidate Methods.
2. Match candidate Methods against existing `Method` nodes:
   - Exact/near-exact name match first.
   - For non-exact matches, issue a single confirmatory LLM call: "is this the same method as X?" — log the decision.
3. Walk the graph from matched Method nodes:
   - Pull Papers connected via `INTRODUCES_METHOD` / `EXTENDS_METHOD` → build a reading order (origin → extensions, chronological).
   - Pull `ADDRESSES_LIMITATION_OF` edges touching those papers → surface known limitations in this space.
   - Pull `COMPETES_WITH` edges → surface competing approaches.
4. Produce structured output:
```json
{
  "closest_prior_work": ["paper_id", "..."],
  "suggested_reading_order": ["paper_id", "..."],
  "known_limitations_in_this_space": ["string", "..."],
  "competing_approaches": ["method_name", "..."]
}
```

This step is what distinguishes the system as reasoning over structure rather than retrieving from storage — it must work on input that was never part of the original dataset.

---

## 6. Explicit Non-Goals (per assignment constraints)

- No NER libraries, spaCy, or automated knowledge-graph-construction tools anywhere in the pipeline.
- No LLM call is permitted to decide the schema itself — the LLM only fills in a schema that is fixed in this document.
- Do not attempt full-field coverage — this is scoped to ~50–100 papers on multi-agent LLM orchestration only.
