"""
ArXiv Paper Collector
====================
Downloads AI/ML research papers from ArXiv using the official API.
Saves PDFs and metadata for the RAG pipeline.

Usage:
    python src/collect_papers.py                           # Default: 75 papers
    python src/collect_papers.py --max-papers 50           # Custom count
    python src/collect_papers.py --query "transformer"     # Custom query
"""

import json
import time
import argparse
import logging
from pathlib import Path

import arxiv
from tqdm import tqdm

# Allow running as standalone script or as part of package
try:
    from config import (
        PAPERS_DIR, METADATA_FILE, ARXIV_QUERY, MAX_PAPERS, ARXIV_SORT_BY,
    )
except ImportError:
    from src.config import (
        PAPERS_DIR, METADATA_FILE, ARXIV_QUERY, MAX_PAPERS, ARXIV_SORT_BY,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def collect_papers(
    query: str = ARXIV_QUERY,
    max_papers: int = MAX_PAPERS,
    output_dir: Path = PAPERS_DIR,
    metadata_file: Path = METADATA_FILE,
) -> list[dict]:
    """
    Download papers from ArXiv and save PDFs + metadata.

    Parameters
    ----------
    query : str
        ArXiv search query (supports categories, titles, abstracts).
    max_papers : int
        Maximum number of papers to download.
    output_dir : Path
        Directory to save downloaded PDFs.
    metadata_file : Path
        Path to save the metadata JSON file.

    Returns
    -------
    list[dict]
        List of paper metadata dictionaries.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching ArXiv for: {query}")
    logger.info(f"Target: {max_papers} papers")

    # Configure ArXiv search
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,       # Respect ArXiv rate limits
        num_retries=3,
    )

    papers_metadata = []
    seen_ids = set()
    downloaded = 0
    errors = 0

    logger.info("Starting download...")

    for result in tqdm(client.results(search), total=max_papers, desc="Downloading"):
        # ── Deduplication ──
        paper_id = result.entry_id.split("/")[-1]
        if paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)

        # ── Sanitize filename ──
        safe_title = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_"
            for c in result.title[:80]
        ).strip()
        filename = f"{paper_id}_{safe_title}.pdf"
        filepath = output_dir / filename

        # ── Skip already-downloaded ──
        if filepath.exists():
            logger.debug(f"Already exists: {filename}")
        else:
            try:
                result.download_pdf(dirpath=str(output_dir), filename=filename)
                time.sleep(0.5)  # Extra courtesy delay
            except Exception as e:
                logger.warning(f"Failed to download {paper_id}: {e}")
                errors += 1
                continue

        # ── Collect metadata ──
        metadata = {
            "id": paper_id,
            "title": result.title,
            "authors": [str(a) for a in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": result.categories,
            "primary_category": result.primary_category,
            "pdf_url": result.pdf_url,
            "local_path": str(filepath),
            "filename": filename,
        }
        papers_metadata.append(metadata)
        downloaded += 1

        if downloaded >= max_papers:
            break

    # ── Save metadata ──
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(papers_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*50}")
    logger.info(f"Download complete!")
    logger.info(f"  Papers downloaded : {downloaded}")
    logger.info(f"  Errors            : {errors}")
    logger.info(f"  PDFs saved to     : {output_dir}")
    logger.info(f"  Metadata saved to : {metadata_file}")
    logger.info(f"{'='*50}")

    return papers_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Download AI/ML papers from ArXiv for RAG pipeline"
    )
    parser.add_argument(
        "--query", type=str, default=ARXIV_QUERY,
        help=f"ArXiv search query (default: '{ARXIV_QUERY}')",
    )
    parser.add_argument(
        "--max-papers", type=int, default=MAX_PAPERS,
        help=f"Maximum papers to download (default: {MAX_PAPERS})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(PAPERS_DIR),
        help=f"Directory for PDFs (default: {PAPERS_DIR})",
    )
    args = parser.parse_args()

    collect_papers(
        query=args.query,
        max_papers=args.max_papers,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
