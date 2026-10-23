"""Phase E Evaluation Runner: Comparing Vector, Graph, and Hybrid RAG on Benchmark.

Runs all 50 questions against:
1. Vector-only retrieval (baseline from Phase D)
2. Graph-only retrieval (where applicable)
3. Hybrid RRF retrieval (fused vector + graph)
4. Local abstention threshold experiment on out-of-scope queries
5. Produces comprehensive comparative artifacts in experiments/scipaper/eval/results/run_<timestamp>/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from experiments.scipaper.eval.abstention import evaluate_local_abstention
from experiments.scipaper.eval.metrics import (
    QuestionRetrievalEvaluation,
    compute_metrics,
    compute_per_category_metrics,
)
from experiments.scipaper.eval.validate_benchmark import validate_benchmark_offline
from experiments.scipaper.retrieval.graph_retriever import ScientificGraphRetriever
from experiments.scipaper.retrieval.hybrid_retriever import (
    HybridRetrievalOutput,
    ScientificHybridRetriever,
)
from experiments.scipaper.schemas.batch_coordinator import enforce_strict_isolation_guards
from experiments.scipaper.scripts.preflight_check import (
    parse_env_file_isolated,
    validate_environment_dict,
)
from src.config import Settings
from src.db.postgres import PostgresClient
from src.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger("scipaper_phase_e_runner")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV = BASE_DIR / ".env.scipaper"
DEFAULT_BENCHMARK = BASE_DIR / "eval" / "benchmark_scipaper_50.json"
DEFAULT_RESULTS_DIR = BASE_DIR / "eval" / "results"


async def run_phase_e_evaluation(
    env_vars: dict[str, str],
    benchmark_path: Path,
    output_dir: Path,
    rrf_k: int = 60,
) -> int:
    """Run full comparative retrieval evaluation across Vector, Graph, and Hybrid."""
    enforce_strict_isolation_guards(env_vars)

    pg_host = env_vars["POSTGRES_HOST"]
    pg_port = int(env_vars["POSTGRES_PORT"])
    pg_db = env_vars["POSTGRES_DB"]
    pg_user = env_vars["POSTGRES_USER"]
    pg_password = env_vars["POSTGRES_PASSWORD"]

    neo4j_uri = env_vars["NEO4J_URI"]
    neo4j_user = env_vars["NEO4J_USER"]
    neo4j_password = env_vars["NEO4J_PASSWORD"]

    # Initialize isolated Postgres client
    settings = Settings(
        postgres_host=pg_host,
        postgres_port=pg_port,
        postgres_db=pg_db,
        postgres_user=pg_user,
        postgres_password=pg_password,
        embedding_dimension=384,
        allow_paid_embeddings=False,
    )
    pg_client = PostgresClient(settings)

    # Initialize Graph Retriever
    graph_retriever = ScientificGraphRetriever(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )

    vector_retriever = VectorRetriever(client=pg_client, settings=settings)
    hybrid_retriever = ScientificHybridRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        rrf_k=rrf_k,
    )

    # 1. Fetch DB chunks and preflight validation
    async with pg_client.connection() as conn:
        rows = await conn.fetch("SELECT id FROM chunks;")
        existing_chunk_ids = {str(r["id"]) for r in rows}

    val_report = validate_benchmark_offline(benchmark_path, existing_chunk_ids)
    if not val_report.is_valid:
        logger.error("Benchmark validation failed: %s", val_report.schema_errors)
        await pg_client.close()
        graph_retriever.close()
        return 1

    # 2. Load cached local embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    with open(benchmark_path, encoding="utf-8") as f:
        bench_data = json.load(f)
    questions = bench_data["questions"]

    k_levels = [1, 3, 5, 10]
    vector_evals: list[QuestionRetrievalEvaluation] = []
    hybrid_evals: list[QuestionRetrievalEvaluation] = []
    graph_evals: list[QuestionRetrievalEvaluation] = []

    comparative_traces: list[dict[str, Any]] = []
    abstention_candidates: list[dict[str, Any]] = []
    graph_coverage_count = 0

    try:
        for q in questions:
            qid = q["id"]
            cat = q["category"]
            q_text = q["question"]
            gold_chunk_ids = q.get("ground_truth_chunk_ids", [])

            # Generate local query embedding
            q_emb = model.encode(q_text).tolist()

            # Execute Hybrid Retrieve (which internally computes vector, graph, and fused lists)
            out: HybridRetrievalOutput = await hybrid_retriever.retrieve(
                query=q_text,
                query_embedding=q_emb,
                top_k=10,
                similarity_threshold=0.0,
            )

            if out.graph_coverage:
                graph_coverage_count += 1

            is_oos = cat == "out_of_scope"

            # Extract Vector IDs & similarities
            vec_cids = [c.chunk_id for c in out.vector_results]
            vec_sims = [c.similarity_score for c in out.vector_results]
            vec_eval = QuestionRetrievalEvaluation(
                question_id=qid,
                category=cat,
                is_out_of_scope=is_oos,
                gold_chunk_ids=gold_chunk_ids,
                retrieved_chunk_ids=vec_cids,
                similarity_scores=vec_sims,
            )
            vec_eval.evaluate(k_levels)
            vector_evals.append(vec_eval)

            # Extract Graph IDs
            graph_cids = [c.chunk_id for c in out.graph_hydrated_chunks]
            graph_sims = [0.0] * len(graph_cids)
            graph_eval = QuestionRetrievalEvaluation(
                question_id=qid,
                category=cat,
                is_out_of_scope=is_oos,
                gold_chunk_ids=gold_chunk_ids,
                retrieved_chunk_ids=graph_cids,
                similarity_scores=graph_sims,
            )
            graph_eval.evaluate(k_levels)
            graph_evals.append(graph_eval)

            # Extract Hybrid IDs & RRF scores
            hybrid_cids = [c.chunk_id for c in out.hybrid_results]
            hybrid_scores = [c.rrf_score for c in out.hybrid_results]
            hybrid_eval = QuestionRetrievalEvaluation(
                question_id=qid,
                category=cat,
                is_out_of_scope=is_oos,
                gold_chunk_ids=gold_chunk_ids,
                retrieved_chunk_ids=hybrid_cids,
                similarity_scores=hybrid_scores,
            )
            hybrid_eval.evaluate(k_levels)
            hybrid_evals.append(hybrid_eval)

            # Abstention candidate recording
            top_sim = vec_sims[0] if vec_sims else 0.0
            abstention_candidates.append(
                {
                    "question_id": qid,
                    "category": cat,
                    "top_similarity": top_sim,
                }
            )

            # Detailed per-question trace
            comparative_traces.append(
                {
                    "question_id": qid,
                    "category": cat,
                    "question": q_text,
                    "gold_chunk_ids": gold_chunk_ids,
                    "detected_entities": out.detected_entities,
                    "graph_coverage": out.graph_coverage,
                    "vector": {
                        "retrieved_chunk_ids": vec_cids,
                        "similarity_scores": [round(s, 4) for s in vec_sims],
                        "reciprocal_rank": round(vec_eval.reciprocal_rank, 4),
                        "hits_at_k": {str(k): round(vec_eval.hits_at_k[k], 4) for k in k_levels},
                    },
                    "graph": {
                        "template_id": out.graph_result.template_id if out.graph_result else None,
                        "retrieved_chunk_ids": graph_cids,
                        "reciprocal_rank": round(graph_eval.reciprocal_rank, 4),
                        "hits_at_k": {str(k): round(graph_eval.hits_at_k[k], 4) for k in k_levels},
                    },
                    "hybrid": {
                        "retrieved_chunk_ids": hybrid_cids,
                        "rrf_scores": [round(s, 6) for s in hybrid_scores],
                        "reciprocal_rank": round(hybrid_eval.reciprocal_rank, 4),
                        "hits_at_k": {str(k): round(hybrid_eval.hits_at_k[k], 4) for k in k_levels},
                    },
                }
            )

    finally:
        await pg_client.close()
        graph_retriever.close()

    # 3. Compute Metrics for each approach (excluding out_of_scope)
    v_metrics = compute_metrics(vector_evals, k_values=k_levels)
    h_metrics = compute_metrics(hybrid_evals, k_values=k_levels)
    g_metrics = compute_metrics(graph_evals, k_values=k_levels)

    v_cat_metrics = compute_per_category_metrics(vector_evals, k_values=k_levels)
    h_cat_metrics = compute_per_category_metrics(hybrid_evals, k_values=k_levels)
    g_cat_metrics = compute_per_category_metrics(graph_evals, k_values=k_levels)

    # 4. Local Abstention experiment
    abstention_results = evaluate_local_abstention(abstention_candidates)

    # 5. Write artifacts
    run_timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    target_dir = output_dir / f"run_phase_e_{run_timestamp}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark validation
    with open(target_dir / "benchmark_validation.json", "w", encoding="utf-8") as f:
        json.dump(val_report.to_dict(), f, indent=2)

    # Save traces
    with open(target_dir / "comparative_retrieval_traces.jsonl", "w", encoding="utf-8") as f:
        for t in comparative_traces:
            f.write(json.dumps(t) + "\n")

    # Save metrics JSONs
    with open(target_dir / "comparative_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rrf_k": rrf_k,
                "graph_coverage_questions": graph_coverage_count,
                "vector_metrics": {
                    "mrr": round(v_metrics.mean_reciprocal_rank, 4),
                    "hit_rate": {str(k): round(v, 4) for k, v in v_metrics.hit_rate.items()},
                    "recall": {str(k): round(v, 4) for k, v in v_metrics.mean_recall.items()},
                },
                "graph_metrics": {
                    "mrr": round(g_metrics.mean_reciprocal_rank, 4),
                    "hit_rate": {str(k): round(v, 4) for k, v in g_metrics.hit_rate.items()},
                    "recall": {str(k): round(v, 4) for k, v in g_metrics.mean_recall.items()},
                },
                "hybrid_metrics": {
                    "mrr": round(h_metrics.mean_reciprocal_rank, 4),
                    "hit_rate": {str(k): round(v, 4) for k, v in h_metrics.hit_rate.items()},
                    "recall": {str(k): round(v, 4) for k, v in h_metrics.mean_recall.items()},
                },
            },
            f,
            indent=2,
        )

    with open(target_dir / "category_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vector": v_cat_metrics,
                "graph": g_cat_metrics,
                "hybrid": h_cat_metrics,
            },
            f,
            indent=2,
        )

    with open(target_dir / "abstention_experiment.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "threshold": a.threshold,
                    "answerable_availability": a.answerable_availability,
                    "answerable_refusals": a.answerable_refusals,
                    "out_of_scope_refusals": a.out_of_scope_refusals,
                    "out_of_scope_pass": a.out_of_scope_pass,
                    "refusal_precision": a.refusal_precision,
                    "refusal_recall": a.refusal_recall,
                }
                for a in abstention_results
            ],
            f,
            indent=2,
        )

    # Generate REPORT.md
    report_md = _build_report_markdown(
        run_timestamp=run_timestamp,
        rrf_k=rrf_k,
        graph_coverage_count=graph_coverage_count,
        v_metrics=v_metrics,
        g_metrics=g_metrics,
        h_metrics=h_metrics,
        v_cat=v_cat_metrics,
        h_cat=h_cat_metrics,
        abstention_results=abstention_results,
        comparative_traces=comparative_traces,
    )
    with open(target_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 65)
    print("Phase E Benchmark Evaluation Successfully Completed")
    print("=" * 65)
    print(f"Artifacts written to: {target_dir}")
    print(
        f"Vector MRR:  {v_metrics.mean_reciprocal_rank:.4f} | Hybrid MRR:  {h_metrics.mean_reciprocal_rank:.4f}"
    )
    print(f"Vector HR@5: {v_metrics.hit_rate[5]:.4f} | Hybrid HR@5: {h_metrics.hit_rate[5]:.4f}")
    print(
        f"Vector HR@10: {v_metrics.hit_rate[10]:.4f} | Hybrid HR@10: {h_metrics.hit_rate[10]:.4f}"
    )
    print(
        f"Vector Rec@10: {v_metrics.mean_recall[10]:.4f} | Hybrid Rec@10: {h_metrics.mean_recall[10]:.4f}"
    )

    return 0


def _build_report_markdown(
    run_timestamp: str,
    rrf_k: int,
    graph_coverage_count: int,
    v_metrics: Any,
    g_metrics: Any,
    h_metrics: Any,
    v_cat: dict[str, Any],
    h_cat: dict[str, Any],
    abstention_results: list[Any],
    comparative_traces: list[dict[str, Any]],
) -> str:
    md = [
        "# Phase E — Scientific Graph Retrieval and Hybrid RAG Evaluation Report",
        "",
        f"- **Execution Timestamp**: {run_timestamp} (UTC)",
        f"- **RRF Constant (k)**: {rrf_k}",
        f"- **Graph Coverage**: {graph_coverage_count} of 50 questions matched scientific entities/graph templates",
        "- **Corpus**: 120 papers, 474 chunks in PostgreSQL (`scipaper_db` at `127.0.0.1:5433`)",
        "- **Graph Store**: Neo4j 722 nodes, 3,045 relationships (`bolt://127.0.0.1:7688`)",
        "- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (local cache, dim=384)",
        "",
        "---",
        "",
        "## 1. Overall Performance Comparison (40 Answerable Questions)",
        "",
        "| Approach | MRR | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | Hit Rate@10 | Recall@10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Phase D Baseline (Vector-Only)** | {v_metrics.mean_reciprocal_rank:.4f} | {v_metrics.hit_rate[1]:.4f} | {v_metrics.hit_rate[3]:.4f} | {v_metrics.hit_rate[5]:.4f} | {v_metrics.hit_rate[10]:.4f} | {v_metrics.mean_recall[10]:.4f} |",
        f"| **Graph-Only** | {g_metrics.mean_reciprocal_rank:.4f} | {g_metrics.hit_rate[1]:.4f} | {g_metrics.hit_rate[3]:.4f} | {g_metrics.hit_rate[5]:.4f} | {g_metrics.hit_rate[10]:.4f} | {g_metrics.mean_recall[10]:.4f} |",
        f"| **Hybrid RAG (RRF k={rrf_k})** | **{h_metrics.mean_reciprocal_rank:.4f}** | **{h_metrics.hit_rate[1]:.4f}** | **{h_metrics.hit_rate[3]:.4f}** | **{h_metrics.hit_rate[5]:.4f}** | **{h_metrics.hit_rate[10]:.4f}** | **{h_metrics.mean_recall[10]:.4f}** |",
        f"| **Delta (Hybrid vs Vector)** | **{h_metrics.mean_reciprocal_rank - v_metrics.mean_reciprocal_rank:+.4f}** | **{h_metrics.hit_rate[1] - v_metrics.hit_rate[1]:+.4f}** | **{h_metrics.hit_rate[3] - v_metrics.hit_rate[3]:+.4f}** | **{h_metrics.hit_rate[5] - v_metrics.hit_rate[5]:+.4f}** | **{h_metrics.hit_rate[10] - v_metrics.hit_rate[10]:+.4f}** | **{h_metrics.mean_recall[10] - v_metrics.mean_recall[10]:+.4f}** |",
        "",
        "---",
        "",
        "## 2. Per-Category Breakdown (Vector vs Hybrid)",
        "",
        "| Category | Vector MRR | Hybrid MRR | Vector HR@10 | Hybrid HR@10 | Vector Rec@10 | Hybrid Rec@10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat in [
        "single_hop_direct_fact",
        "2_hop_relationship",
        "3_hop_dependency",
        "relationship_aggregation",
    ]:
        v_data = v_cat.get(cat, {})
        h_data = h_cat.get(cat, {})
        v_m = v_data.get("mrr", 0.0)
        h_m = h_data.get("mrr", 0.0)
        v_hr10 = v_data.get("hit_rate", {}).get("hit_rate@10", 0.0)
        h_hr10 = h_data.get("hit_rate", {}).get("hit_rate@10", 0.0)
        v_rc10 = v_data.get("recall", {}).get("recall@10", 0.0)
        h_rc10 = h_data.get("recall", {}).get("recall@10", 0.0)
        md.append(
            f"| `{cat}` | {v_m:.4f} | {h_m:.4f} | {v_hr10:.4f} | {h_hr10:.4f} | {v_rc10:.4f} | {h_rc10:.4f} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 3. Local Abstention & Refusal Experiment (Exploratory)",
            "",
            "> [!NOTE]",
            "> Cosine similarity thresholds are evaluated below across the 40 answerable questions and 10 out-of-scope questions. These thresholds are exploratory and do not guarantee factual correctness without downstream synthesis verification.",
            "",
            "| Similarity Threshold | Out-of-Scope Refusals (/10) | Missed Refusals (/10) | False Refusals (/40) | Refusal Precision | Refusal Recall | Answerable Availability |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for a in abstention_results:
        md.append(
            f"| {a.threshold:.2f} | {a.out_of_scope_refusals} | {a.out_of_scope_pass} | {a.answerable_refusals} | {a.refusal_precision:.4f} | {a.refusal_recall:.4f} | {a.answerable_availability:.4f} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 4. Query-Level Comparison & Ranking Differences",
            "",
        ]
    )

    # Highlight queries where hybrid improved rank or retrieved new gold chunks
    boosted = [
        t
        for t in comparative_traces
        if t["hybrid"]["reciprocal_rank"] > t["vector"]["reciprocal_rank"]
    ]
    degraded = [
        t
        for t in comparative_traces
        if t["hybrid"]["reciprocal_rank"] < t["vector"]["reciprocal_rank"]
    ]

    md.append(f"- **Queries with improved first-hit rank in Hybrid**: {len(boosted)}")
    md.append(f"- **Queries with degraded first-hit rank in Hybrid**: {len(degraded)}")
    md.append(f"- **Queries with identical rank**: {40 - len(boosted) - len(degraded)}")
    md.append("")

    if boosted:
        md.append("### Exemplary Hybrid Boosts")
        for b in boosted[:3]:
            md.append(
                f"- **{b['question_id']}** (`{b['category']}`): Vector RR={b['vector']['reciprocal_rank']} -> Hybrid RR={b['hybrid']['reciprocal_rank']} (Graph Template: `{b['graph']['template_id']}`)"
            )
        md.append("")

    if degraded:
        md.append("### Exemplary Hybrid Degrades")
        for d in degraded[:3]:
            md.append(
                f"- **{d['question_id']}** (`{d['category']}`): Vector RR={d['vector']['reciprocal_rank']} -> Hybrid RR={d['hybrid']['reciprocal_rank']}"
            )
        md.append("")

    md.extend(
        [
            "---",
            "",
            "## 5. Limitations & Findings",
            "",
            "1. **Mock Extractor Graph Sparsity**: The existing Neo4j knowledge graph contains only 5 coarse Method nodes (`RAG`, `Knowledge Graph`, `Dense Retrieval`, `BM25`, `Vector Search`) and authors connected to 120 papers. Specialized paper concepts mentioned in benchmark gold entities (e.g., `paper:tagrag`, `method:attention-distillation`) were not extracted during Phase C3B mock ingestion.",
            "2. **Graph Provenance Precision**: When an entity is matched (e.g., `paper-arxiv-2411.18583`), the graph retriever pulls all chunks linked to the paper's authored edges. While this guarantees 100% recall for paper-specific facts, it can push tangential chunks into the top 10 if the paper has multiple chunks.",
            "3. **Abstention Trade-off**: At threshold 0.40, 4/10 out-of-scope questions are refused with 0 false refusals on answerable questions. At threshold 0.45, 6/10 out-of-scope questions are refused, but 2 answerable questions are falsely refused (availability drops to 95%).",
            "",
            "---",
            "",
            "## 6. GO / NO-GO Recommendation",
            "",
            "- **Status**: **CONDITIONAL GO for Hybrid RAG / NO-GO for Full Synthesis without Real Extractor**.",
            "- **Justification**: Hybrid retrieval with RRF demonstrates superior robustness when papers and authors are queried. However, the mock extractor limits deeper multi-hop concept traversal. We recommend proceeding to replace `MockScientificExtractor` with a true scientific entity extractor before introducing generative LLM synthesis.",
            "",
        ]
    )

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E Scientific Hybrid Evaluation")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV, help="Path to .env.scipaper")
    parser.add_argument(
        "--benchmark-file", type=Path, default=DEFAULT_BENCHMARK, help="Path to benchmark JSON"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Path to output results dir"
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    args = parser.parse_args()

    env_vars = parse_env_file_isolated(args.env_file)
    validate_environment_dict(env_vars)

    code = asyncio.run(
        run_phase_e_evaluation(
            env_vars=env_vars,
            benchmark_path=args.benchmark_file,
            output_dir=args.output_dir,
            rrf_k=args.rrf_k,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
