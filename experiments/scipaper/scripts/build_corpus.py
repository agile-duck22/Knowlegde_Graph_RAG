"""Build canonical manifest.jsonl and corpus_scipaper.jsonl from raw harvested arXiv papers."""

import hashlib
import json
from pathlib import Path

raw_file = Path("experiments/scipaper/data/raw_harvested_papers.json")
manifest_file = Path("experiments/scipaper/data/manifest.jsonl")
corpus_file = Path("experiments/scipaper/data/corpus_scipaper.jsonl")

with raw_file.open("r", encoding="utf-8") as f:
    papers = json.load(f)

manifest_lines = []
corpus_lines = []

for p in papers:
    content = f"# Title: {p['title']}\n\n# Authors: {', '.join(p['authors'])}\n\n# Venue: {p['venue']} ({p['year']})\n\n# Abstract\n\n{p['summary']}"
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    manifest_entry = {
        "paper_id": p["paper_id"],
        "title": p["title"],
        "authors": p["authors"],
        "year": p["year"],
        "venue": p["venue"],
        "source_url": p["source_url"],
        "license": p["license"],
        "content_type": "abstract",
        "sha256": content_sha256,
        "doi": p.get("doi"),
        "arxiv_id": p["arxiv_id"],
    }
    manifest_lines.append(json.dumps(manifest_entry))

    corpus_entry = {
        "document_id": p["paper_id"],
        "source_uri": p["source_url"],
        "title": p["title"],
        "doc_type": "scientific_paper",
        "content": content,
        "metadata": {
            "authors": p["authors"],
            "year": p["year"],
            "venue": p["venue"],
            "license": p["license"],
            "content_type": "abstract",
            "arxiv_id": p["arxiv_id"],
            "doi": p.get("doi"),
        },
    }
    corpus_lines.append(json.dumps(corpus_entry))

with manifest_file.open("w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines) + "\n")

with corpus_file.open("w", encoding="utf-8") as f:
    f.write("\n".join(corpus_lines) + "\n")

print(
    f"Successfully generated {len(manifest_lines)} manifest entries and {len(corpus_lines)} corpus documents."
)
