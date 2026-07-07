"""
Script to coordinate LLM extraction for all papers in raw_papers.json.
"""

import os
import json
import logging
import argparse
from extraction.extractor import PaperExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_extraction(input_path: str = "data/raw_papers.json", output_path: str = "data/structured.json") -> None:
    """Runs the LLM extractor on all papers in the raw_papers.json file.

    Args:
        input_path: Path to the raw papers JSON.
        output_path: Path to write the structured papers JSON.
    """
    if not os.path.exists(input_path):
        logger.error("Input file %s does not exist.", input_path)
        return

    logger.info("Loading raw papers from %s...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    logger.info("Loaded %d papers. Initializing PaperExtractor...", len(papers))
    try:
        extractor = PaperExtractor()
    except ValueError as e:
        logger.error("Failed to initialize PaperExtractor: %s", str(e))
        return

    structured_results = []
    failed_papers = []
    consecutive_errors = 0

    for i, paper in enumerate(papers, start=1):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        paper_id = paper.get("paperId") or paper.get("id", "")
        
        logger.info("[%d/%d] Extracting from: '%s' (ID: %s)", i, len(papers), title, paper_id)
        
        if not title or not abstract:
            logger.warning("Paper %s has empty title or abstract. Skipping.", paper_id)
            failed_papers.append((paper_id, title, "Missing title or abstract"))
            continue

        extracted_data = extractor.extract_from_paper(title, abstract)
        
        if extracted_data is None:
            logger.warning("Paper '%s' (ID: %s) failed extraction/validation.", title, paper_id)
            failed_papers.append((paper_id, title, "Extraction/Validation failed"))
            consecutive_errors += 1
            if consecutive_errors >= 3:
                logger.error("Too many consecutive API errors (%d). Stopping extraction early to check credentials or network.", consecutive_errors)
                raise RuntimeError("Too many consecutive API failures during LLM extraction.")
            continue
            
        consecutive_errors = 0
        # Combine the original paper metadata with the LLM-extracted schema fields
        structured_paper = {
            "paperId": paper_id,
            "title": title,
            "year": paper.get("year"),
            "abstract": abstract,
            "venue": paper.get("venue"),
            "authors": paper.get("authors", []),
            "citations": paper.get("citations", []),
            "references": paper.get("references", []),
            "core_contribution": extracted_data["core_contribution"],
            "methods": extracted_data["methods"],
            "competes_with": extracted_data["competes_with"]
        }
        structured_results.append(structured_paper)

    # Save structured papers
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    logger.info("Saving %d structured papers to %s...", len(structured_results), output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2, ensure_ascii=False)

    if failed_papers:
        logger.warning("The following %d papers failed validation/extraction:", len(failed_papers))
        for fid, ftitle, reason in failed_papers:
            logger.warning("  - ID: %s | Title: '%s' | Reason: %s", fid, ftitle, reason)
    else:
        logger.info("All papers processed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM schema extraction on raw papers dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_papers.json",
        help="Path to raw papers JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/structured.json",
        help="Path to save structured output JSON file"
    )
    args = parser.parse_args()
    run_extraction(args.input, args.output)


if __name__ == "__main__":
    main()
