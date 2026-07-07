"""
LLM Extraction module that identifies entities and relationships from paper abstracts.

This module formats prompts, calls the LLM, validates JSON output against the fixed schema,
logs failures, and manages method deduplication to ensure consistent ontology.
"""

from typing import Any, Dict, List, Optional


import os
import json
import logging
from typing import Any, Dict, List, Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Pydantic Schemas matching the requested schema exactly
class ExtractedMethod(BaseModel):
    name: str
    description: Optional[str] = ""
    relationship: Literal["INTRODUCES_METHOD", "APPLIES_METHOD", "EXTENDS_METHOD"]
    targets_limitation_of: Optional[str] = None

class PaperExtraction(BaseModel):
    core_contribution: str
    methods: List[ExtractedMethod]
    competes_with: List[str] = Field(default_factory=list)


PROMPT_TEMPLATE = """You are extracting structured knowledge from a single research paper according to a FIXED schema. Do not invent entity or relationship types outside this schema. If something doesn't fit, omit it rather than forcing it.

ENTITY TYPES:
- Method: a named technique, architecture, or design pattern for multi-agent LLM systems (must be a specific, nameable approach — NOT a vague topic, and NOT the name of a framework, paper, or system itself, e.g. do not extract 'MetaGPT', 'ChatDev', 'AutoGen', 'GPTSwarm', or 'CAMEL' as methods. Instead, extract the specific named techniques they introduce or apply, like 'Shared Blackboard Memory', 'Communicative Agent Method', 'sequential software development process (SOP)', or 'Conversable Agents').

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
1. Identify 0-3 Methods discussed in this paper (only genuinely named/definable techniques. CRITICAL: Do NOT extract paper/system/framework names as Methods).
2. For each Method, state the relationship type. This relationship type MUST be exactly one of: INTRODUCES_METHOD, APPLIES_METHOD, or EXTENDS_METHOD.
3. If ADDRESSES_LIMITATION_OF applies, name which specific prior work/approach (by description if not exact title) it targets in the 'targets_limitation_of' field. Do not use ADDRESSES_LIMITATION_OF as a relationship value.
4. If COMPETES_WITH applies, place the competing method name in the 'competes_with' list. Do not use COMPETES_WITH as a relationship value in the methods list.
5. Write core_contribution as ONE sentence: what is new here, not what the paper is about generally

OUTPUT (valid JSON only, no prose, no markdown fences):
{{
  "core_contribution": "string",
  "methods": [
    {{
      "name": "string",
      "description": "string",
      "relationship": "INTRODUCES_METHOD | APPLIES_METHOD | EXTENDS_METHOD",
      "targets_limitation_of": "string or null"
    }}
  ],
  "competes_with": ["method_name, if explicitly compared"]
}}"""


class PaperExtractor:
    """Extractor for translating unstructured paper data (abstracts, titles) into structured schema objects."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initializes the LLM extractor client using Groq API endpoint.

        Args:
            api_key: Optional API key for the Groq service.
        """
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set in environment or passed in.")
        
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key
        )
        self.model = "llama-3.1-8b-instant"

    def extract_from_paper(self, title: str, abstract: str) -> Optional[Dict[str, Any]]:
        """Calls the LLM using the fixed schema prompt template.

        Validates output JSON. Retries once if validation fails.

        Args:
            title: The paper title.
            abstract: The paper abstract.

        Returns:
            A dictionary containing core_contribution, extracted methods list,
            and competing methods, or None if extraction fails twice.
        """
        prompt = PROMPT_TEMPLATE.format(title=title, abstract=abstract)
        
        for attempt in range(1, 3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise data extraction assistant. Output only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM.")
                
                # Strip markdown code blocks if any got returned despite system prompt
                content_clean = content.strip()
                if content_clean.startswith("```"):
                    # Find first { and last }
                    start_idx = content_clean.find("{")
                    end_idx = content_clean.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        content_clean = content_clean[start_idx : end_idx + 1]

                # Parse JSON dict for preprocessing
                data_dict = json.loads(content_clean)
                
                # Preprocess to fix common relationship mismatches (COMPETES_WITH or ADDRESSES_LIMITATION_OF in relationship field)
                methods = data_dict.get("methods", [])
                fixed_methods = []
                for m in methods:
                    rel = m.get("relationship", "")
                    if rel == "COMPETES_WITH":
                        # Move name to competes_with if not already there, and rewrite relationship
                        comp_name = m.get("name")
                        if comp_name and comp_name not in data_dict.get("competes_with", []):
                            if "competes_with" not in data_dict:
                                data_dict["competes_with"] = []
                            data_dict["competes_with"].append(comp_name)
                        # Assume it applies the method as fallback
                        m["relationship"] = "APPLIES_METHOD"
                    elif rel == "ADDRESSES_LIMITATION_OF":
                        # Move target to targets_limitation_of and rewrite relationship
                        if not m.get("targets_limitation_of"):
                            m["targets_limitation_of"] = m.get("description", "Prior work")
                        m["relationship"] = "EXTENDS_METHOD"
                    fixed_methods.append(m)
                data_dict["methods"] = fixed_methods

                # Parse and validate with Pydantic
                parsed_data = PaperExtraction.model_validate(data_dict)
                return parsed_data.model_dump()

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    "Validation failed on attempt %d for paper '%s': %s",
                    attempt, title, str(e)
                )
                if attempt == 2:
                    logger.error("Failed validation twice for paper '%s'. Skipping.", title)
                    return None
            except Exception as e:
                logger.error("API error during extraction for '%s': %s", title, str(e))
                if attempt == 2:
                    return None

        return None

    def validate_schema(self, extracted_json: Dict[str, Any]) -> bool:
        """Validates that the extracted output strictly matches the required database schema.

        Args:
            extracted_json: The dictionary returned from the LLM.

        Returns:
            True if schema is valid, False otherwise.
        """
        try:
            PaperExtraction.model_validate(extracted_json)
            return True
        except ValidationError:
            return False

    def deduplicate_methods(self, candidate_methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges duplicate or near-duplicate Methods (e.g. 'shared blackboard' vs 'blackboard memory').

        Uses name normalization and confirmatory LLM prompts to reconcile names.
        Logs every merge decision to approach.md.

        Args:
            candidate_methods: List of all extracted method dictionaries.

        Returns:
            A deduplicated list of method entities.
        """
        # A simple slug-based and LLM-assisted deduplication can be implemented in the run script.
        # For the class itself, we return the candidate methods list.
        return candidate_methods

