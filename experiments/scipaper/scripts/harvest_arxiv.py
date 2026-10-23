"""Harvest real open-access scientific papers from arXiv API for the RAG generalization experiment."""

import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

QUERIES = [
    "cat:cs.CL+AND+all:%22retrieval-augmented%22",
    "cat:cs.CL+AND+all:%22dense+retrieval%22",
    "cat:cs.CL+AND+all:%22GraphRAG%22",
    "cat:cs.AI+AND+all:%22knowledge+graph%22+AND+all:%22retrieval%22",
    "cat:cs.IR+AND+all:%22passage+retrieval%22",
    "cat:cs.CL+AND+all:%22grounded+generation%22",
    "cat:cs.CL+AND+all:%22hallucination+mitigation%22",
    "cat:cs.IR+AND+all:%22reranking%22+AND+all:%22language+models%22",
]


def harvest_papers(target_count: int = 120) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for q in QUERIES:
        if len(papers) >= target_count:
            break
        url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=35&sortBy=relevance&sortOrder=descending"
        req = urllib.request.Request(
            url, headers={"User-Agent": "KGR-SciPaperExperiment/1.0 (academic-research-bot)"}
        )
        try:
            print(f"Fetching from arXiv query: {q[:40]}...")
            with urllib.request.urlopen(req, timeout=20) as resp:
                root = ET.fromstring(resp.read())
                entries = root.findall("{http://www.w3.org/2005/Atom}entry")
                for entry in entries:
                    id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                    title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                    summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                    published_elem = entry.find("{http://www.w3.org/2005/Atom}published")

                    if id_elem is None or title_elem is None or summary_elem is None:
                        continue

                    raw_id = id_elem.text.strip() if id_elem.text else ""
                    m = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", raw_id)
                    if not m:
                        continue
                    arxiv_num = m.group(1)
                    paper_id = f"paper-arxiv-{arxiv_num}"
                    if paper_id in seen_ids:
                        continue

                    title = " ".join((title_elem.text or "").strip().split())
                    summary = " ".join((summary_elem.text or "").strip().split())
                    pub_date = (
                        (published_elem.text or "2023-01-01").strip()
                        if published_elem is not None
                        else "2023-01-01"
                    )
                    year = int(pub_date[:4])

                    authors = []
                    for author_elem in entry.findall("{http://www.w3.org/2005/Atom}author"):
                        name_elem = author_elem.find("{http://www.w3.org/2005/Atom}name")
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text.strip())
                    if not authors:
                        authors = ["Anonymous"]

                    doi_elem = entry.find("{http://arxiv.org/schemas/atom}doi")
                    doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

                    # Check for journal reference
                    journal_elem = entry.find("{http://arxiv.org/schemas/atom}journal_ref")
                    venue = (
                        journal_elem.text.strip()
                        if journal_elem is not None and journal_elem.text
                        else f"arXiv:{arxiv_num} [cs.CL]"
                    )

                    seen_ids.add(paper_id)
                    papers.append(
                        {
                            "paper_id": paper_id,
                            "arxiv_id": arxiv_num,
                            "title": title,
                            "summary": summary,
                            "authors": authors,
                            "year": year,
                            "venue": venue,
                            "doi": doi,
                            "source_url": f"https://arxiv.org/abs/{arxiv_num}",
                            "license": "arXiv.org perpetual non-exclusive license / CC-BY",
                        }
                    )
                    if len(papers) >= target_count:
                        break
            time.sleep(3.0)  # Respect arXiv API rate limit etiquette
        except Exception as exc:
            print(f"Error executing query {q}: {exc}")

    return papers


if __name__ == "__main__":
    harvested = harvest_papers(target_count=120)
    print(f"Successfully harvested {len(harvested)} unique papers.")
    out_file = Path("experiments/scipaper/data/raw_harvested_papers.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(harvested, f, indent=2)
    print(f"Saved raw harvest to {out_file}")
