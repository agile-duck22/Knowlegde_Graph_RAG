"""Phase F Offline Scientific Extraction Evaluation Runner.

Evaluates OfflineScientificExtractor on a deterministic evaluation set of 20 papers
and computes precision, recall, and F1 against the 10-paper gold subset.
Checks provenance validity, duplicate rates, and produces machine-readable JSON metrics.

Constraints:
- ZERO external API calls, ZERO model downloads, ZERO database writes.
- Deterministic and offline execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.scipaper.schemas.extractor import OfflineScientificExtractor
from src.ingest.chunker import DocumentChunker
from src.ingest.loader import RawDocument

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "corpus_scipaper.jsonl"
GOLD_PATH = BASE_DIR / "eval" / "data" / "extraction_gold_subset.json"
RESULTS_PATH = BASE_DIR / "eval" / "results" / "extraction_evaluation.json"


def run_evaluation() -> dict[str, Any]:
    print("=" * 60)
    print("Phase F: Offline Scientific Extraction Evaluation")
    print("=" * 60)

    # 1. Load gold data
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_papers = {p["document_id"]: p for p in gold_data["papers"]}
    print(f"Loaded {len(gold_papers)} gold annotated papers.")

    # 2. Load corpus and select 20 evaluation papers (first 20 papers deterministically)
    eval_docs: list[RawDocument] = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 20:
                break
            record = json.loads(line)
            doc = RawDocument(
                document_id=record["document_id"],
                source_uri=record["source_uri"],
                title=record["title"],
                doc_type=record["doc_type"],
                content=record["content"],
                metadata=record.get("metadata", {}),
            )
            eval_docs.append(doc)

    print(f"Loaded {len(eval_docs)} evaluation papers from corpus.")

    # 3. Instantiate offline extractor
    extractor = OfflineScientificExtractor()

    # Metrics accumulators
    total_chunks = 0
    total_extracted_entities = 0
    total_extracted_relationships = 0
    valid_provenance_count = 0
    invalid_records_count = 0

    gold_eval_results: list[dict[str, Any]] = []

    # Per-type entity counters
    entity_tp: dict[str, int] = {}
    entity_fp: dict[str, int] = {}
    entity_fn: dict[str, int] = {}

    # Per-type relationship counters
    rel_tp: dict[str, int] = {}
    rel_fp: dict[str, int] = {}
    rel_fn: dict[str, int] = {}

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
    for doc in eval_docs:
        chunks = chunker.chunk(doc)
        total_chunks += len(chunks)

        doc_entities: dict[str, Any] = {}
        doc_relationships: set[tuple[str, str, str]] = set()

        for chunk in chunks:
            result = extractor.extract(chunk)

            # Provenance and schema checks
            for ent in result.entities:
                if (
                    ent.canonical_id
                    and ent.entity_type
                    and ent.provenance.chunk_id == chunk.chunk_id
                ):
                    valid_provenance_count += 1
                else:
                    invalid_records_count += 1
                doc_entities[ent.canonical_id] = ent

            for rel in result.relationships:
                if (
                    rel.source_canonical_id
                    and rel.target_canonical_id
                    and chunk.chunk_id in rel.source_chunk_ids
                ):
                    valid_provenance_count += 1
                else:
                    invalid_records_count += 1
                doc_relationships.add(
                    (rel.source_canonical_id, rel.relation_type, rel.target_canonical_id)
                )

        total_extracted_entities += len(doc_entities)
        total_extracted_relationships += len(doc_relationships)

        # Compare with Gold if paper in gold subset
        if doc.document_id in gold_papers:
            gold = gold_papers[doc.document_id]
            g_entities = {e["canonical_id"]: e for e in gold["gold_entities"]}
            g_rels = {
                (r["source_canonical_id"], r["relation_type"], r["target_canonical_id"]): r
                for r in gold["gold_relationships"]
            }

            # Entity precision/recall
            pred_ent_cids = set(doc_entities.keys())
            gold_ent_cids = set(g_entities.keys())

            for cid in pred_ent_cids:
                etype = doc_entities[cid].entity_type
                if cid in gold_ent_cids:
                    entity_tp[etype] = entity_tp.get(etype, 0) + 1
                else:
                    entity_fp[etype] = entity_fp.get(etype, 0) + 1

            for cid in gold_ent_cids:
                etype = g_entities[cid]["entity_type"]
                if cid not in pred_ent_cids:
                    entity_fn[etype] = entity_fn.get(etype, 0) + 1

            # Relationship precision/recall
            pred_rel_keys = doc_relationships
            gold_rel_keys = set(g_rels.keys())

            for rkey in pred_rel_keys:
                rtype = rkey[1]
                if rkey in gold_rel_keys:
                    rel_tp[rtype] = rel_tp.get(rtype, 0) + 1
                else:
                    rel_fp[rtype] = rel_fp.get(rtype, 0) + 1

            for rkey in gold_rel_keys:
                rtype = rkey[1]
                if rkey not in pred_rel_keys:
                    rel_fn[rtype] = rel_fn.get(rtype, 0) + 1

            gold_eval_results.append(
                {
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "gold_entities_count": len(gold_ent_cids),
                    "pred_entities_count": len(pred_ent_cids),
                    "gold_relationships_count": len(gold_rel_keys),
                    "pred_relationships_count": len(pred_rel_keys),
                }
            )

    # Compute overall Entity P, R, F1
    total_ent_tp = sum(entity_tp.values())
    total_ent_fp = sum(entity_fp.values())
    total_ent_fn = sum(entity_fn.values())

    ent_p = (
        total_ent_tp / (total_ent_tp + total_ent_fp) if (total_ent_tp + total_ent_fp) > 0 else 0.0
    )
    ent_r = (
        total_ent_tp / (total_ent_tp + total_ent_fn) if (total_ent_tp + total_ent_fn) > 0 else 0.0
    )
    ent_f1 = 2 * ent_p * ent_r / (ent_p + ent_r) if (ent_p + ent_r) > 0 else 0.0

    # Per-type entity metrics
    entity_metrics: dict[str, dict[str, float]] = {}
    all_etypes = set(entity_tp.keys()) | set(entity_fp.keys()) | set(entity_fn.keys())
    for etype in sorted(all_etypes):
        tp = entity_tp.get(etype, 0)
        fp = entity_fp.get(etype, 0)
        fn = entity_fn.get(etype, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        entity_metrics[etype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Compute overall Relationship P, R, F1
    total_rel_tp = sum(rel_tp.values())
    total_rel_fp = sum(rel_fp.values())
    total_rel_fn = sum(rel_fn.values())

    rel_p = (
        total_rel_tp / (total_rel_tp + total_rel_fp) if (total_rel_tp + total_rel_fp) > 0 else 0.0
    )
    rel_r = (
        total_rel_tp / (total_rel_tp + total_rel_fn) if (total_rel_tp + total_rel_fn) > 0 else 0.0
    )
    rel_f1 = 2 * rel_p * rel_r / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

    # Per-type relationship metrics
    rel_metrics: dict[str, dict[str, float]] = {}
    all_rtypes = set(rel_tp.keys()) | set(rel_fp.keys()) | set(rel_fn.keys())
    for rtype in sorted(all_rtypes):
        tp = rel_tp.get(rtype, 0)
        fp = rel_fp.get(rtype, 0)
        fn = rel_fn.get(rtype, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rel_metrics[rtype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_claims = valid_provenance_count + invalid_records_count
    prov_validity_rate = valid_provenance_count / total_claims if total_claims > 0 else 1.0

    output_payload = {
        "evaluation_summary": {
            "eval_corpus_papers_count": len(eval_docs),
            "eval_total_chunks_processed": total_chunks,
            "total_extracted_unique_entities": total_extracted_entities,
            "total_extracted_unique_relationships": total_extracted_relationships,
            "provenance_validity_rate": round(prov_validity_rate, 4),
            "invalid_records_count": invalid_records_count,
            "gold_subset_evaluated_papers": len(gold_eval_results),
        },
        "overall_metrics": {
            "entity_precision": round(ent_p, 4),
            "entity_recall": round(ent_r, 4),
            "entity_f1": round(ent_f1, 4),
            "relationship_precision": round(rel_p, 4),
            "relationship_recall": round(rel_r, 4),
            "relationship_f1": round(rel_f1, 4),
        },
        "entity_metrics_by_type": entity_metrics,
        "relationship_metrics_by_type": rel_metrics,
        "gold_papers_detail": gold_eval_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nEvaluation complete. Results written to: {RESULTS_PATH}")
    print(f"Entity F1: {ent_f1:.4f} (P: {ent_p:.4f}, R: {ent_r:.4f})")
    print(f"Relationship F1: {rel_f1:.4f} (P: {rel_p:.4f}, R: {rel_r:.4f})")
    print(f"Provenance Validity Rate: {prov_validity_rate * 100:.2f}%")
    return output_payload


if __name__ == "__main__":
    run_evaluation()
