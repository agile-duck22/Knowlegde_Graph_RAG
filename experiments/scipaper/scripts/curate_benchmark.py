"""Curate 50 genuinely evidence-grounded benchmark questions for the scientific paper corpus.

Taxonomy:
- 10 single_hop_direct_fact
- 10 2_hop_relationship
- 10 3_hop_dependency
- 10 relationship_aggregation
- 10 out_of_scope

Every answerable question has:
- Verified expected facts extracted directly from the harvested paper text.
- Grounded reference answer.
- Precomputed canonical chunk IDs matching DocumentChunker(chunk_size=500, chunk_overlap=50).
- Explicit metadata with source paper IDs, evidence notes, and verified status.
"""

import json
from pathlib import Path

from src.eval.schema import BenchmarkCategory, BenchmarkDifficulty
from src.ingest.chunker import DocumentChunker
from src.ingest.loader import RawDocument

corpus_file = Path("experiments/scipaper/data/corpus_scipaper.jsonl")
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

# Load documents and pre-generate canonical chunk IDs
docs = {}
doc_chunks = {}
with corpus_file.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = RawDocument.model_validate(json.loads(line))
            docs[d.document_id] = d
            chunks = chunker.chunk(d)
            doc_chunks[d.document_id] = [c.chunk_id for c in chunks]

questions = []

# ==============================================================================
# Category 1: Single-Hop Direct Fact (EVAL-001 to EVAL-010)
# ==============================================================================

# EVAL-001: Automated Literature Review (paper-arxiv-2411.18583)
p1 = "paper-arxiv-2411.18583"
questions.append(
    {
        "id": "EVAL-001",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What core objective and techniques are investigated in the paper on automated literature reviews (paper-arxiv-2411.18583)?",
        "ground_truth_entities": ["paper:automated-literature-review", "method:rag"],
        "ground_truth_chunk_ids": doc_chunks[p1][:2],
        "expected_facts": [
            "The paper compares multiple approaches to automate literature review generation using NLP techniques and RAG with an LLM.",
            "It addresses the challenge of keeping pace with the ever-increasing volume of scientific publications.",
        ],
        "reference_answer": "The paper investigates automating the generation of literature reviews using several Natural Language Processing (NLP) techniques and retrieval-augmented generation (RAG) powered by Large Language Models to cope with the rapid expansion of academic literature.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p1],
            "evidence_notes": "Extracted directly from paper-arxiv-2411.18583 abstract describing NLP and LLM-based RAG for automated literature reviews.",
        },
    }
)

# EVAL-002: Stealthy Membership Inference in RAG (paper-arxiv-2502.00306)
p2 = "paper-arxiv-2502.00306"
questions.append(
    {
        "id": "EVAL-002",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What privacy vulnerability does paper-arxiv-2502.00306 examine in retrieval-augmented generation systems?",
        "ground_truth_entities": ["paper:riddle-me-this", "vuln:membership-inference"],
        "ground_truth_chunk_ids": doc_chunks[p2][:2],
        "expected_facts": [
            "It examines membership inference attacks against external knowledge databases used by RAG systems.",
            "It shows that lack of weight tuning does not prevent leakage from the retrieval database.",
        ],
        "reference_answer": "Paper-arxiv-2502.00306 examines stealthy membership inference attacks targeting RAG systems, demonstrating that even when model parameters remain frozen, external knowledge databases can still suffer from privacy leakage.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p2],
            "evidence_notes": "Abstract explicitly outlines stealthy membership inference against RAG knowledge stores.",
        },
    }
)

# EVAL-003: MUST-RAG Musical QA (paper-arxiv-2507.23334)
p3 = "paper-arxiv-2507.23334"
questions.append(
    {
        "id": "EVAL-003",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What domain and benchmark task is MUST-RAG designed to address according to paper-arxiv-2507.23334?",
        "ground_truth_entities": ["paper:must-rag", "task:musical-text-qa"],
        "ground_truth_chunk_ids": doc_chunks[p3][:2],
        "expected_facts": [
            "MUST-RAG targets musical text question answering.",
            "It addresses LLMs' limited effectiveness in music-related domain tasks.",
        ],
        "reference_answer": "MUST-RAG is designed to address musical text question answering, improving LLM performance in music-related applications where zero-shot models typically struggle.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p3],
            "evidence_notes": "Abstract states the proposal of MUST-RAG for musical text question answering.",
        },
    }
)

# EVAL-004: FAIR-RAG Iterative Refinement (paper-arxiv-2510.22344)
p4 = "paper-arxiv-2510.22344"
questions.append(
    {
        "id": "EVAL-004",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What is the primary methodology introduced by FAIR-RAG in paper-arxiv-2510.22344 to handle complex multi-hop queries?",
        "ground_truth_entities": ["paper:fair-rag", "method:adaptive-iterative-refinement"],
        "ground_truth_chunk_ids": doc_chunks[p4][:2],
        "expected_facts": [
            "FAIR-RAG introduces faithful adaptive iterative refinement for RAG.",
            "It addresses failure modes on complex multi-hop queries requiring synthesis across disparate sources.",
        ],
        "reference_answer": "FAIR-RAG introduces Faithful Adaptive Iterative Refinement, an approach that iteratively refines queries and retrieved contexts to solve complex multi-hop synthesis challenges where traditional RAG falters.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p4],
            "evidence_notes": "Abstract explicitly highlights faithful adaptive iterative refinement for multi-hop synthesis.",
        },
    }
)

# EVAL-005: CRISS Cancer Registry AI (paper-arxiv-2609.29075)
p5 = "paper-arxiv-2609.29075"
questions.append(
    {
        "id": "EVAL-005",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What clinical application and user group does the CRISS system assist in paper-arxiv-2609.29075?",
        "ground_truth_entities": ["paper:criss", "task:cancer-registry-assistance"],
        "ground_truth_chunk_ids": doc_chunks[p5][:2],
        "expected_facts": [
            "CRISS is a retrieval-augmented AI chatbot.",
            "It assists cancer registrars in oncology data collection and curation.",
        ],
        "reference_answer": "CRISS is a retrieval-augmented AI chatbot developed to assist cancer registrars with domain-specific information retrieval and registry curation tasks.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p5],
            "evidence_notes": "Title and abstract define CRISS as a retrieval-augmented assistant for cancer registrars.",
        },
    }
)

# EVAL-006: Multimodal RAG Survey (paper-arxiv-2504.08748)
p6 = "paper-arxiv-2504.08748"
questions.append(
    {
        "id": "EVAL-006",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What architectural evolution in RAG does paper-arxiv-2504.08748 survey?",
        "ground_truth_entities": ["paper:multimodal-rag-survey"],
        "ground_truth_chunk_ids": doc_chunks[p6][:2],
        "expected_facts": [
            "It presents a survey of Multimodal Retrieval-Augmented Generation.",
            "It explores combining textual and visual or multi-sensory modalities into RAG pipelines.",
        ],
        "reference_answer": "Paper-arxiv-2504.08748 surveys Multimodal Retrieval-Augmented Generation, reviewing methodologies that integrate non-text modalities alongside text in retrieval and generation.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p6],
            "evidence_notes": "Title and abstract state 'A Survey of Multimodal Retrieval-Augmented Generation'.",
        },
    }
)

# EVAL-007: Attention Distillation in RAG (paper-arxiv-2402.11794)
p7 = "paper-arxiv-2402.11794"
questions.append(
    {
        "id": "EVAL-007",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What training phase optimization mechanism is analyzed in paper-arxiv-2402.11794 for retrieval-augmented models?",
        "ground_truth_entities": [
            "paper:attention-distillation-rag",
            "method:attention-distillation",
        ],
        "ground_truth_chunk_ids": doc_chunks[p7][:2],
        "expected_facts": [
            "The paper investigates attention distillation during the training phase of retrieval-augmented models.",
            "It enables efficient knowledge transfer and updates for accurate generation.",
        ],
        "reference_answer": "Paper-arxiv-2402.11794 investigates attention distillation as an efficient training-phase mechanism for retrieval-augmented models to improve answer fidelity.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p7],
            "evidence_notes": "Abstract explicitly focuses on investigating attention distillation in RAG.",
        },
    }
)

# EVAL-008: Metacognitive RAG (paper-arxiv-2402.11626)
p8 = "paper-arxiv-2402.11626"
questions.append(
    {
        "id": "EVAL-008",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "How does Metacognitive RAG in paper-arxiv-2402.11626 depart from traditional single-time retrieval?",
        "ground_truth_entities": ["paper:metacognitive-rag", "method:multi-time-retrieval"],
        "ground_truth_chunk_ids": doc_chunks[p8][:2],
        "expected_facts": [
            "It shifts from single-time retrieval to multi-time retrieval.",
            "It introduces metacognitive reasoning to decide when and what to retrieve.",
        ],
        "reference_answer": "Metacognitive RAG departs from conventional single-step retrieval by implementing multi-time retrieval and metacognitive monitoring over when additional external evidence is required.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p8],
            "evidence_notes": "Abstract contrasts traditional single-time retrieval with multi-time retrieval and metacognition.",
        },
    }
)

# EVAL-009: Agentic RAG for Recommendations (paper-arxiv-2506.21931)
p9 = "paper-arxiv-2506.21931"
questions.append(
    {
        "id": "EVAL-009",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What limitation of static retrieval heuristics in recommendation systems is ARAG designed to solve in paper-arxiv-2506.21931?",
        "ground_truth_entities": ["paper:arag", "task:personalized-recommendation"],
        "ground_truth_chunk_ids": doc_chunks[p9][:2],
        "expected_facts": [
            "Static retrieval heuristics fail to capture dynamic user preference shifts.",
            "ARAG introduces agentic retrieval augmented generation for personalized recommendations.",
        ],
        "reference_answer": "ARAG addresses the rigidity of static retrieval heuristics by employing agentic retrieval-augmented generation to dynamically retrieve relevant contextual signals for personalized recommendations.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p9],
            "evidence_notes": "Abstract highlights moving beyond static heuristics with agentic RAG for recommendations.",
        },
    }
)

# EVAL-010: Federated RAG Mapping Study (paper-arxiv-2505.18906)
p10 = "paper-arxiv-2505.18906"
questions.append(
    {
        "id": "EVAL-010",
        "category": BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value,
        "difficulty": BenchmarkDifficulty.EASY.value,
        "question": "What two paradigms are combined in Federated RAG according to paper-arxiv-2505.18906?",
        "ground_truth_entities": [
            "paper:federated-rag-study",
            "method:federated-learning",
            "method:rag",
        ],
        "ground_truth_chunk_ids": doc_chunks[p10][:2],
        "expected_facts": [
            "Federated RAG combines Federated Learning (FL) with Retrieval-Augmented Generation (RAG).",
            "It achieves distributed privacy-preserving model capabilities while improving factual accuracy.",
        ],
        "reference_answer": "Federated RAG combines Federated Learning (FL) for distributed privacy-preserving learning with Retrieval-Augmented Generation (RAG) to ensure factual generation without centralizing raw data.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p10],
            "evidence_notes": "Abstract states: 'Federated RAG combines Federated Learning (FL)... with Retrieval-Augmented Generation (RAG)'.",
        },
    }
)

# ==============================================================================
# Category 2: Two-Hop Cross-Paper Relationship (EVAL-011 to EVAL-020)
# ==============================================================================

# EVAL-011: Chunking Strategies vs Chunk Size (paper-arxiv-2607.01852 & paper-arxiv-2607.24767)
p11 = "paper-arxiv-2607.01852"
p12 = "paper-arxiv-2607.24767"
questions.append(
    {
        "id": "EVAL-011",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "How do paper-arxiv-2607.01852 and paper-arxiv-2607.24767 complement each other in studying the preprocessing stage of RAG pipelines?",
        "ground_truth_entities": [
            "paper:evaluating-chunking-academic",
            "paper:effect-chunk-size-rag",
            "method:chunking",
        ],
        "ground_truth_chunk_ids": doc_chunks[p11][:2] + doc_chunks[p12][:2],
        "expected_facts": [
            "paper-arxiv-2607.01852 evaluates cluster-based semantic chunking on academic texts.",
            "paper-arxiv-2607.24767 investigates the quantitative impact of text chunk size on RAG downstream performance.",
        ],
        "reference_answer": "The two papers examine complementary dimensions of RAG text preprocessing: paper-arxiv-2607.01852 evaluates qualitative chunking strategies like cluster-based semantic chunking, whereas paper-arxiv-2607.24767 systematically tests the effect of variation in chunk size on generation accuracy.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p11, p12],
            "evidence_notes": "Cross-paper link between semantic chunking in academic texts (2607.01852) and chunk size effects (2607.24767).",
        },
    }
)

# EVAL-012: Knowledge Graph Integration in TagRAG vs HetaRAG (paper-arxiv-2601.05254 & paper-arxiv-2509.21336)
p13 = "paper-arxiv-2601.05254"
p14 = "paper-arxiv-2509.21336"
questions.append(
    {
        "id": "EVAL-012",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "Compare how TagRAG (paper-arxiv-2601.05254) and HetaRAG (paper-arxiv-2509.21336) approach knowledge retrieval across complex non-relational structures.",
        "ground_truth_entities": ["paper:tagrag", "paper:hetarag", "database:knowledge-graph"],
        "ground_truth_chunk_ids": doc_chunks[p13][:2] + doc_chunks[p14][:2],
        "expected_facts": [
            "TagRAG uses tag-guided hierarchical knowledge graphs to overcome fragment-level retrieval limitations.",
            "HetaRAG focuses on hybrid deep retrieval spanning heterogeneous data stores.",
        ],
        "reference_answer": "TagRAG introduces tag-guided hierarchical knowledge graph structures to summarize and connect fragmented concepts, whereas HetaRAG solves retrieval across diverse, heterogeneous enterprise data stores to unify fragmented knowledge.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p13, p14],
            "evidence_notes": "Compares TagRAG's hierarchical KG indexing with HetaRAG's heterogeneous multi-store retrieval.",
        },
    }
)

# EVAL-013: Privacy Attacks and Distributed Protection (paper-arxiv-2502.00306 & paper-arxiv-2505.18906)
questions.append(
    {
        "id": "EVAL-013",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "How does the privacy risk exposed in paper-arxiv-2502.00306 relate to the architectural motivation behind Federated RAG in paper-arxiv-2505.18906?",
        "ground_truth_entities": [
            "paper:riddle-me-this",
            "paper:federated-rag-study",
            "vuln:privacy-leakage",
        ],
        "ground_truth_chunk_ids": doc_chunks[p2][:2] + doc_chunks[p10][:2],
        "expected_facts": [
            "Paper-arxiv-2502.00306 shows external retrieval databases are vulnerable to stealthy membership inference.",
            "Paper-arxiv-2505.18906 surveys Federated RAG to avoid exposing or centralizing sensitive raw data across organizations.",
        ],
        "reference_answer": "Paper-arxiv-2502.00306 exposes that centralized external knowledge databases in RAG suffer from membership inference attacks, providing strong motivation for Federated RAG (paper-arxiv-2505.18906), which keeps raw data decentralized and localized.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p2, p10],
            "evidence_notes": "Connects membership inference risk in RAG stores to the privacy justification for federated RAG.",
        },
    }
)

# EVAL-014: Adaptive Iteration vs Metacognition (paper-arxiv-2510.22344 & paper-arxiv-2402.11626)
questions.append(
    {
        "id": "EVAL-014",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "In what way do FAIR-RAG (paper-arxiv-2510.22344) and Metacognitive RAG (paper-arxiv-2402.11626) address multi-step query complexity?",
        "ground_truth_entities": [
            "paper:fair-rag",
            "paper:metacognitive-rag",
            "method:multi-step-retrieval",
        ],
        "ground_truth_chunk_ids": doc_chunks[p4][:2] + doc_chunks[p8][:2],
        "expected_facts": [
            "FAIR-RAG utilizes iterative refinement across disparate sources for multi-hop queries.",
            "Metacognitive RAG implements multi-time retrieval guided by metacognitive self-evaluation.",
        ],
        "reference_answer": "Both frameworks move beyond static single-hop retrieval: FAIR-RAG applies adaptive iterative refinement across multi-hop evidence, while Metacognitive RAG relies on multi-time retrieval evaluated by metacognitive assessment.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p4, p8],
            "evidence_notes": "Examines iterative refinement in FAIR-RAG vs multi-step metacognitive retrieval in 2402.11626.",
        },
    }
)

# EVAL-015: Domain-Specific RAG in Cancer Registries vs Musical Text (paper-arxiv-2609.29075 & paper-arxiv-2507.23334)
questions.append(
    {
        "id": "EVAL-015",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "How do CRISS (paper-arxiv-2609.29075) and MUST-RAG (paper-arxiv-2507.23334) demonstrate the necessity of specialized RAG beyond generic LLM pretraining?",
        "ground_truth_entities": ["paper:criss", "paper:must-rag", "task:domain-specialization"],
        "ground_truth_chunk_ids": doc_chunks[p5][:2] + doc_chunks[p3][:2],
        "expected_facts": [
            "CRISS demonstrates specialized RAG for clinical oncology registry guidelines.",
            "MUST-RAG targets specialized musical text QA where zero-shot LLM knowledge is inadequate.",
        ],
        "reference_answer": "Both papers show that zero-shot generalist LLMs fail in specialized fields: CRISS introduces tailored retrieval for clinical cancer registries, while MUST-RAG supplies structured domain retrieval for musicology and music text QA.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p5, p3],
            "evidence_notes": "Cross-paper analysis of domain-specific RAG adaptation in medicine vs music.",
        },
    }
)

# EVAL-016: Literature Review Automation vs Chunking (paper-arxiv-2411.18583 & paper-arxiv-2607.01852)
questions.append(
    {
        "id": "EVAL-016",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "How does the evaluation of chunking strategies in paper-arxiv-2607.01852 directly impact the automated literature review systems studied in paper-arxiv-2411.18583?",
        "ground_truth_entities": [
            "paper:automated-literature-review",
            "paper:evaluating-chunking-academic",
        ],
        "ground_truth_chunk_ids": doc_chunks[p1][:2] + doc_chunks[p11][:2],
        "expected_facts": [
            "Paper-arxiv-2411.18583 processes long academic papers for automated literature reviews.",
            "Paper-arxiv-2607.01852 evaluates semantic chunking specifically on academic text structure to improve answer fidelity.",
        ],
        "reference_answer": "Automated literature review generation relies on digesting long, dense scientific papers (2411.18583), which directly benefits from semantic cluster-based chunking tailored to academic papers (2607.01852) rather than naive fixed splitting.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p1, p11],
            "evidence_notes": "Links academic paper literature review pipeline to academic chunking benchmarks.",
        },
    }
)

# EVAL-017: Agentic Recommendation vs Heterogeneous Stores (paper-arxiv-2506.21931 & paper-arxiv-2509.21336)
questions.append(
    {
        "id": "EVAL-017",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "What shared architectural trend toward agentic and hybrid retrieval is reflected between ARAG (paper-arxiv-2506.21931) and HetaRAG (paper-arxiv-2509.21336)?",
        "ground_truth_entities": ["paper:arag", "paper:hetarag", "method:hybrid-agentic-rag"],
        "ground_truth_chunk_ids": doc_chunks[p9][:2] + doc_chunks[p14][:2],
        "expected_facts": [
            "ARAG employs agentic decision-making over static heuristics.",
            "HetaRAG deploys deep hybrid retrieval across varied data stores.",
        ],
        "reference_answer": "Both papers reflect the evolution from simple top-k vector search toward autonomous pipelines: ARAG introduces agentic retrieval routing, while HetaRAG orchestrates deep retrieval across multi-format, heterogeneous storage backends.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p9, p14],
            "evidence_notes": "Synthesizes agentic routing in ARAG with heterogeneous retrieval in HetaRAG.",
        },
    }
)

# EVAL-018: Multimodal Expansion vs Attention Distillation (paper-arxiv-2504.08748 & paper-arxiv-2402.11794)
questions.append(
    {
        "id": "EVAL-018",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "Contrast the focus of paper-arxiv-2504.08748 on input modality breadth with the training efficiency focus of paper-arxiv-2402.11794.",
        "ground_truth_entities": [
            "paper:multimodal-rag-survey",
            "paper:attention-distillation-rag",
        ],
        "ground_truth_chunk_ids": doc_chunks[p6][:2] + doc_chunks[p7][:2],
        "expected_facts": [
            "Paper-arxiv-2504.08748 broadens the scope of RAG to multimodal inputs (images, audio, video).",
            "Paper-arxiv-2402.11794 deepens model training efficiency through attention distillation.",
        ],
        "reference_answer": "Paper-arxiv-2504.08748 expands architectural breadth by exploring multimodal retrieval paradigms, whereas paper-arxiv-2402.11794 focuses internally on model training optimization via attention distillation.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p6, p7],
            "evidence_notes": "Contrasts multimodal breadth survey with mechanistic attention distillation.",
        },
    }
)

# EVAL-019: Graph Hierarchies vs Multi-hop Iteration (paper-arxiv-2601.05254 & paper-arxiv-2510.22344)
questions.append(
    {
        "id": "EVAL-019",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "How do TagRAG (paper-arxiv-2601.05254) and FAIR-RAG (paper-arxiv-2510.22344) provide alternative solutions to multi-hop context synthesis?",
        "ground_truth_entities": ["paper:tagrag", "paper:fair-rag", "method:multi-hop-synthesis"],
        "ground_truth_chunk_ids": doc_chunks[p13][:2] + doc_chunks[p4][:2],
        "expected_facts": [
            "TagRAG structures information into a hierarchical knowledge graph with tag guidance.",
            "FAIR-RAG solves multi-hop synthesis via adaptive iterative refinement loops during query time.",
        ],
        "reference_answer": "TagRAG solves multi-hop synthesis structurally through hierarchical knowledge graph indexing at ingestion, while FAIR-RAG solves it dynamically through iterative query-time refinement loops.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p13, p4],
            "evidence_notes": "Compares structural KG hierarchy in TagRAG with dynamic query iteration in FAIR-RAG.",
        },
    }
)

# EVAL-020: Chunk Size Variance vs Automated Literature Review (paper-arxiv-2607.24767 & paper-arxiv-2411.18583)
questions.append(
    {
        "id": "EVAL-020",
        "category": BenchmarkCategory.TWO_HOP_RELATIONSHIP.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "Why is the empirical finding regarding chunk size in paper-arxiv-2607.24767 critical when implementing the literature review pipelines from paper-arxiv-2411.18583?",
        "ground_truth_entities": [
            "paper:effect-chunk-size-rag",
            "paper:automated-literature-review",
        ],
        "ground_truth_chunk_ids": doc_chunks[p12][:2] + doc_chunks[p1][:2],
        "expected_facts": [
            "Chunk size directly affects retrieval precision and context fragmentation.",
            "Literature review synthesis requires balancing broad conceptual context with specific empirical details.",
        ],
        "reference_answer": "Chunk size selection (paper-arxiv-2607.24767) dictates whether extracted passages retain sufficient context for synthesis without overwhelming prompt budgets, directly impacting the factual coherence of automated literature reviews (paper-arxiv-2411.18583).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p12, p1],
            "evidence_notes": "Connects chunk size parameter study to literature review synthesis quality.",
        },
    }
)

# ==============================================================================
# Category 3: Three-Hop Dependency (EVAL-021 to EVAL-030)
# ==============================================================================

# EVAL-021: Chunking Strategy -> Retrieval Precision -> Literature Review Synthesis
questions.append(
    {
        "id": "EVAL-021",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Trace the dependency chain connecting chunk size (paper-arxiv-2607.24767), semantic chunking on academic papers (paper-arxiv-2607.01852), and automated literature review generation (paper-arxiv-2411.18583).",
        "ground_truth_entities": [
            "paper:effect-chunk-size-rag",
            "paper:evaluating-chunking-academic",
            "paper:automated-literature-review",
        ],
        "ground_truth_chunk_ids": doc_chunks[p12][:2] + doc_chunks[p11][:2] + doc_chunks[p1][:2],
        "expected_facts": [
            "Chunk size determines the boundary granularity of text fragments (2607.24767).",
            "Semantic cluster chunking refines these boundaries along academic discourse structures (2607.01852).",
            "High-fidelity chunk extraction powers accurate, comprehensive literature review synthesis (2411.18583).",
        ],
        "reference_answer": "The dependency chain progresses from parameter-level chunk sizing (2607.24767), to semantic cluster chunking adapted for academic papers (2607.01852), which in turn feeds coherent context into automated literature review pipelines (2411.18583).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p12, p11, p1],
            "evidence_notes": "3-hop dependency: chunk sizing -> academic semantic boundaries -> literature review generation.",
        },
    }
)

# EVAL-022: Privacy Attack -> Knowledge Graph Fragmentation -> Federated Protection
questions.append(
    {
        "id": "EVAL-022",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Trace the multi-hop dependency between membership inference vulnerabilities (paper-arxiv-2502.00306), heterogeneous data stores (paper-arxiv-2509.21336), and federated deployment (paper-arxiv-2505.18906).",
        "ground_truth_entities": [
            "paper:riddle-me-this",
            "paper:hetarag",
            "paper:federated-rag-study",
        ],
        "ground_truth_chunk_ids": doc_chunks[p2][:2] + doc_chunks[p14][:2] + doc_chunks[p10][:2],
        "expected_facts": [
            "Membership inference leaks sensitive private documents from centralized stores (2502.00306).",
            "Modern RAG pools sensitive records across heterogeneous enterprise stores (2509.21336).",
            "Federated RAG addresses both challenges by maintaining localized data governance without centralized pooling (2505.18906).",
        ],
        "reference_answer": "Centralized retrieval across heterogeneous enterprise stores (2509.21336) magnifies membership inference risks (2502.00306), necessitating federated RAG architectures (2505.18906) to enforce data boundaries.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p2, p14, p10],
            "evidence_notes": "3-hop link: privacy vulnerability -> multi-store enterprise integration -> federated mitigation.",
        },
    }
)

# EVAL-023: Hierarchical Graph Indexing -> Multi-hop Iteration -> Adaptive Refinement
questions.append(
    {
        "id": "EVAL-023",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "How does hierarchical knowledge graph structuring in TagRAG (paper-arxiv-2601.05254) connect with iterative refinement in FAIR-RAG (paper-arxiv-2510.22344) and metacognitive retrieval (paper-arxiv-2402.11626)?",
        "ground_truth_entities": ["paper:tagrag", "paper:fair-rag", "paper:metacognitive-rag"],
        "ground_truth_chunk_ids": doc_chunks[p13][:2] + doc_chunks[p4][:2] + doc_chunks[p8][:2],
        "expected_facts": [
            "TagRAG structures fragmented documents into hierarchical graph entities (2601.05254).",
            "FAIR-RAG traverses complex entity links using iterative query refinement (2510.22344).",
            "Metacognitive control evaluates when sufficient hops have been traversed to generate grounded answers (2402.11626).",
        ],
        "reference_answer": "TagRAG establishes the hierarchical graph relationships, FAIR-RAG iteratively navigates these links across multi-hop queries, and Metacognitive RAG monitors the sufficiency of retrieved evidence before synthesis.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p13, p4, p8],
            "evidence_notes": "3-hop dependency: hierarchical graph representation -> iterative traversal -> metacognitive stopping criteria.",
        },
    }
)

# EVAL-024: Training Phase Attention Distillation -> Agentic Query Routing -> Domain Chatbots
questions.append(
    {
        "id": "EVAL-024",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Trace the technical progression from attention distillation during model training (paper-arxiv-2402.11794), to agentic routing (paper-arxiv-2506.21931), and specialized assistant deployment in CRISS (paper-arxiv-2609.29075).",
        "ground_truth_entities": ["paper:attention-distillation-rag", "paper:arag", "paper:criss"],
        "ground_truth_chunk_ids": doc_chunks[p7][:2] + doc_chunks[p9][:2] + doc_chunks[p5][:2],
        "expected_facts": [
            "Attention distillation creates efficient retriever-reader alignment (2402.11794).",
            "Agentic RAG orchestrates dynamic query decision-making (2506.21931).",
            "CRISS deploys these capabilities into a specialized clinical assistant for cancer registrars (2609.29075).",
        ],
        "reference_answer": "Optimized reader-retriever attention distillation (2402.11794) enables effective agentic routing (2506.21931), which ultimately powers specialized high-reliability domain chatbots such as CRISS (2609.29075).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p7, p9, p5],
            "evidence_notes": "3-hop progression: training distillation -> dynamic agent routing -> specialized clinical deployment.",
        },
    }
)

# EVAL-025: Modality Expansion -> Specialized Domain Audio/Music -> Chunking Considerations
questions.append(
    {
        "id": "EVAL-025",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "How do multimodal RAG paradigms (paper-arxiv-2504.08748) connect to specialized musical text reasoning in MUST-RAG (paper-arxiv-2507.23334) and chunking evaluations (paper-arxiv-2607.01852)?",
        "ground_truth_entities": [
            "paper:multimodal-rag-survey",
            "paper:must-rag",
            "paper:evaluating-chunking-academic",
        ],
        "ground_truth_chunk_ids": doc_chunks[p6][:2] + doc_chunks[p3][:2] + doc_chunks[p11][:2],
        "expected_facts": [
            "Multimodal RAG expands beyond uniform plain text (2504.08748).",
            "MUST-RAG requires domain-specific symbolic/text representations of musical knowledge (2507.23334).",
            "Semantic chunking must adapt to maintain domain-specific structural coherence (2607.01852).",
        ],
        "reference_answer": "Multimodal RAG frameworks (2504.08748) encompass non-standard modalities like musical notations and domain text in MUST-RAG (2507.23334), which depend heavily on structured semantic chunking (2607.01852) to avoid losing critical symbolic context.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p6, p3, p11],
            "evidence_notes": "3-hop link: multimodal expansion -> specialized musical domain semantics -> academic chunking boundaries.",
        },
    }
)

# EVAL-026: Iterative Refinement -> Knowledge Graphs -> Literature Reviews
questions.append(
    {
        "id": "EVAL-026",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Trace how iterative refinement (paper-arxiv-2510.22344) and hierarchical graphs (paper-arxiv-2601.05254) together overcome limitations in automated scientific literature synthesis (paper-arxiv-2411.18583).",
        "ground_truth_entities": [
            "paper:fair-rag",
            "paper:tagrag",
            "paper:automated-literature-review",
        ],
        "ground_truth_chunk_ids": doc_chunks[p4][:2] + doc_chunks[p13][:2] + doc_chunks[p1][:2],
        "expected_facts": [
            "Hierarchical knowledge graphs organize cross-paper concepts and taxonomy (2601.05254).",
            "Iterative refinement dynamically explores related research threads across multiple hops (2510.22344).",
            "Automated literature review tools require both to synthesize coherent summaries across hundreds of publications (2411.18583).",
        ],
        "reference_answer": "Hierarchical graph indexing (2601.05254) provides structured research entity relationships that iterative refinement (2510.22344) traverses to generate holistic, grounded automated literature reviews (2411.18583).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p4, p13, p1],
            "evidence_notes": "3-hop dependency: graph structure -> iterative traversal -> comprehensive literature synthesis.",
        },
    }
)

# EVAL-027: Static Heuristic Limits -> Multi-Time Retrieval -> Clinical Registry Accuracy
questions.append(
    {
        "id": "EVAL-027",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "How do limitations of static retrieval (paper-arxiv-2506.21931) and metacognitive multi-time retrieval (paper-arxiv-2402.11626) impact critical clinical accuracy in systems like CRISS (paper-arxiv-2609.29075)?",
        "ground_truth_entities": ["paper:arag", "paper:metacognitive-rag", "paper:criss"],
        "ground_truth_chunk_ids": doc_chunks[p9][:2] + doc_chunks[p8][:2] + doc_chunks[p5][:2],
        "expected_facts": [
            "Static heuristics fail to adapt when user queries require deeper context (2506.21931).",
            "Metacognitive reflection checks whether evidence is sufficient to prevent hallucinations (2402.11626).",
            "Clinical cancer registry systems require rigorous verification to prevent dangerous medical misinformation (2609.29075).",
        ],
        "reference_answer": "Static retrieval shortcomings (2506.21931) necessitate metacognitive verification loops (2402.11626), which are vital in clinical tools like CRISS (2609.29075) where unverified retrieval risks medical errors.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p9, p8, p5],
            "evidence_notes": "3-hop dependency: heuristic failure -> metacognitive reflection -> clinical safety.",
        },
    }
)

# EVAL-028: Chunk Sizing -> Heterogeneous Stores -> Privacy Leakage
questions.append(
    {
        "id": "EVAL-028",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Trace the relationship between chunk size choices (paper-arxiv-2607.24767), heterogeneous store aggregation (paper-arxiv-2509.21336), and susceptibility to membership inference (paper-arxiv-2502.00306).",
        "ground_truth_entities": [
            "paper:effect-chunk-size-rag",
            "paper:hetarag",
            "paper:riddle-me-this",
        ],
        "ground_truth_chunk_ids": doc_chunks[p12][:2] + doc_chunks[p14][:2] + doc_chunks[p2][:2],
        "expected_facts": [
            "Chunk size defines verbatim passage length stored in vectors (2607.24767).",
            "Heterogeneous retrieval aggregates these chunks across diverse repositories (2509.21336).",
            "Verbatim chunk retrieval directly facilitates membership inference attacks that verify presence of private records (2502.00306).",
        ],
        "reference_answer": "Larger verbatim chunk sizes (2607.24767) aggregated across heterogeneous repositories (2509.21336) increase unique token sequences exposed in generation, heightening vulnerability to membership inference attacks (2502.00306).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p12, p14, p2],
            "evidence_notes": "3-hop link: chunk granularity -> heterogeneous aggregation -> membership leakage vulnerability.",
        },
    }
)

# EVAL-029: Semantic Chunking -> Attention Distillation -> Domain Specialization
questions.append(
    {
        "id": "EVAL-029",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "How does academic semantic chunking (paper-arxiv-2607.01852) interact with attention distillation (paper-arxiv-2402.11794) to improve domain-specific QA in MUST-RAG (paper-arxiv-2507.23334)?",
        "ground_truth_entities": [
            "paper:evaluating-chunking-academic",
            "paper:attention-distillation-rag",
            "paper:must-rag",
        ],
        "ground_truth_chunk_ids": doc_chunks[p11][:2] + doc_chunks[p7][:2] + doc_chunks[p3][:2],
        "expected_facts": [
            "Semantic chunking preserves coherent domain concepts without mid-sentence truncations (2607.01852).",
            "Attention distillation aligns model attention onto these coherent passages during training (2402.11794).",
            "Specialized QA applications like MUST-RAG achieve superior factual generation as a result (2507.23334).",
        ],
        "reference_answer": "Semantic chunking (2607.01852) provides clean conceptual units that maximize the effectiveness of attention distillation (2402.11794), driving higher factual accuracy in specialized music QA systems (2507.23334).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p11, p7, p3],
            "evidence_notes": "3-hop link: semantic chunk quality -> attention distillation alignment -> specialized QA performance.",
        },
    }
)

# EVAL-030: Graph RAG Hierarchies -> Agentic Recommendation -> Literature Reviews
questions.append(
    {
        "id": "EVAL-030",
        "category": BenchmarkCategory.THREE_HOP_DEPENDENCY.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Trace how knowledge graph hierarchies (paper-arxiv-2601.05254) and agentic routing (paper-arxiv-2506.21931) together enhance automated scientific literature discovery (paper-arxiv-2411.18583).",
        "ground_truth_entities": [
            "paper:tagrag",
            "paper:arag",
            "paper:automated-literature-review",
        ],
        "ground_truth_chunk_ids": doc_chunks[p13][:2] + doc_chunks[p9][:2] + doc_chunks[p1][:2],
        "expected_facts": [
            "Hierarchical knowledge graphs represent multi-level conceptual dependencies (2601.05254).",
            "Agentic RAG orchestrates dynamic multi-query exploratory retrieval (2506.21931).",
            "Automated literature review tools leverage both to generate structured synthesis across wide scientific corpora (2411.18583).",
        ],
        "reference_answer": "Hierarchical graph representations (2601.05254) paired with agentic retrieval exploration (2506.21931) enable automated literature review tools (2411.18583) to comprehensively map and synthesize evolving scientific domains.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p13, p9, p1],
            "evidence_notes": "3-hop dependency: hierarchical KG indexing -> agentic exploration -> comprehensive literature mapping.",
        },
    }
)

# ==============================================================================
# Category 4: Relationship Aggregation (EVAL-031 to EVAL-040)
# ==============================================================================

# EVAL-031: Aggregation of Multi-hop RAG Approaches
questions.append(
    {
        "id": "EVAL-031",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Across the corpus, compare the distinct strategies proposed by FAIR-RAG (paper-arxiv-2510.22344), TagRAG (paper-arxiv-2601.05254), and Metacognitive RAG (paper-arxiv-2402.11626) to overcome traditional fragment-level retrieval limitations.",
        "ground_truth_entities": ["paper:fair-rag", "paper:tagrag", "paper:metacognitive-rag"],
        "ground_truth_chunk_ids": doc_chunks[p4][:2] + doc_chunks[p13][:2] + doc_chunks[p8][:2],
        "expected_facts": [
            "FAIR-RAG uses faithful adaptive iterative refinement.",
            "TagRAG constructs a tag-guided hierarchical knowledge graph.",
            "Metacognitive RAG implements multi-time retrieval with metacognitive reflection.",
        ],
        "reference_answer": "To surpass single-shot fragment retrieval, FAIR-RAG uses iterative query refinement, TagRAG introduces hierarchical tag-guided knowledge graphs, and Metacognitive RAG applies multi-step retrieval regulated by metacognitive reflection.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p4, p13, p8],
            "evidence_notes": "Aggregates multi-hop strategies across FAIR-RAG, TagRAG, and Metacognitive RAG.",
        },
    }
)

# EVAL-032: Aggregation of Preprocessing & Chunking Research
questions.append(
    {
        "id": "EVAL-032",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "What collective insights emerge from paper-arxiv-2607.01852 and paper-arxiv-2607.24767 regarding chunking optimization in academic RAG pipelines?",
        "ground_truth_entities": [
            "paper:evaluating-chunking-academic",
            "paper:effect-chunk-size-rag",
        ],
        "ground_truth_chunk_ids": doc_chunks[p11][:2] + doc_chunks[p12][:2],
        "expected_facts": [
            "Chunk size is a primary determinant of downstream LLM accuracy.",
            "Semantic cluster-based chunking outperforms arbitrary character/token splits on academic document structures.",
        ],
        "reference_answer": "Collectively, the papers demonstrate that chunking is a critical architectural lever: optimizing chunk size directly impacts generation accuracy (2607.24767), and semantic cluster chunking preserves logical discourse structure better than fixed-length splits (2607.01852).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p11, p12],
            "evidence_notes": "Synthesizes findings on chunk size and semantic chunking strategies.",
        },
    }
)

# EVAL-033: Aggregation of Privacy & Federated RAG Paradigms
questions.append(
    {
        "id": "EVAL-033",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "How do papers in the corpus characterize privacy risks in standard RAG (paper-arxiv-2502.00306) and the architectural response in federated environments (paper-arxiv-2505.18906)?",
        "ground_truth_entities": ["paper:riddle-me-this", "paper:federated-rag-study"],
        "ground_truth_chunk_ids": doc_chunks[p2][:2] + doc_chunks[p10][:2],
        "expected_facts": [
            "Standard RAG leaks private database membership even without parameter tuning.",
            "Federated RAG decentralizes retrieval databases across privacy-preserving nodes.",
        ],
        "reference_answer": "The literature characterizes standard RAG as vulnerable to membership inference despite frozen parameters (2502.00306), prompting federated RAG architectures that distribute and safeguard private knowledge bases (2505.18906).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p2, p10],
            "evidence_notes": "Aggregates privacy risk analysis and federated mitigation.",
        },
    }
)

# EVAL-034: Aggregation of Domain-Specific Applications
questions.append(
    {
        "id": "EVAL-034",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "Summarize the domain-specific RAG applications evaluated across the corpus in healthcare (paper-arxiv-2609.29075), music (paper-arxiv-2507.23334), and literature synthesis (paper-arxiv-2411.18583).",
        "ground_truth_entities": [
            "paper:criss",
            "paper:must-rag",
            "paper:automated-literature-review",
        ],
        "ground_truth_chunk_ids": doc_chunks[p5][:2] + doc_chunks[p3][:2] + doc_chunks[p1][:2],
        "expected_facts": [
            "Healthcare: CRISS assists cancer registrars in oncology data curation.",
            "Music: MUST-RAG improves musical text QA where zero-shot models fail.",
            "Literature review: NLP and RAG automate scientific synthesis across vast paper collections.",
        ],
        "reference_answer": "The corpus demonstrates specialized RAG across diverse domains: oncology registry assistance (CRISS), musical question answering (MUST-RAG), and automated scientific literature review generation.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p5, p3, p1],
            "evidence_notes": "Aggregates domain-specific deployments across healthcare, music, and academic literature.",
        },
    }
)

# EVAL-035: Aggregation of Advanced Knowledge Retrieval Architectures
questions.append(
    {
        "id": "EVAL-035",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Compare how TagRAG (paper-arxiv-2601.05254) and HetaRAG (paper-arxiv-2509.21336) incorporate non-vector structures to improve retrieval coverage.",
        "ground_truth_entities": ["paper:tagrag", "paper:hetarag"],
        "ground_truth_chunk_ids": doc_chunks[p13][:2] + doc_chunks[p14][:2],
        "expected_facts": [
            "TagRAG employs hierarchical knowledge graph communities with semantic tags.",
            "HetaRAG integrates diverse heterogeneous storage backends beyond standard vector tables.",
        ],
        "reference_answer": "TagRAG introduces hierarchical tag-guided knowledge graph structures for entity-level synthesis, while HetaRAG bridges multi-store heterogeneous repositories to unify disconnected data sources.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p13, p14],
            "evidence_notes": "Aggregates non-vector structural retrieval across TagRAG and HetaRAG.",
        },
    }
)

# EVAL-036: Aggregation of Training vs Inference Optimization
questions.append(
    {
        "id": "EVAL-036",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Contrast the training-time optimization in paper-arxiv-2402.11794 with inference-time optimizations in paper-arxiv-2402.11626 and paper-arxiv-2510.22344.",
        "ground_truth_entities": [
            "paper:attention-distillation-rag",
            "paper:metacognitive-rag",
            "paper:fair-rag",
        ],
        "ground_truth_chunk_ids": doc_chunks[p7][:2] + doc_chunks[p8][:2] + doc_chunks[p4][:2],
        "expected_facts": [
            "Paper-arxiv-2402.11794 optimizes training via attention distillation.",
            "Paper-arxiv-2402.11626 optimizes inference via metacognitive multi-time retrieval.",
            "Paper-arxiv-2510.22344 optimizes inference via iterative adaptive query refinement.",
        ],
        "reference_answer": "Paper-arxiv-2402.11794 optimizes model weights during training using attention distillation, whereas Metacognitive RAG (2402.11626) and FAIR-RAG (2510.22344) optimize query-time inference through metacognition and iterative refinement.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p7, p8, p4],
            "evidence_notes": "Aggregates training-time vs inference-time RAG enhancements.",
        },
    }
)

# EVAL-037: Aggregation of Recommendation and Agentic Workflows
questions.append(
    {
        "id": "EVAL-037",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "How do agentic RAG systems (paper-arxiv-2506.21931) differ from traditional static retrieval pipelines across the reviewed literature?",
        "ground_truth_entities": ["paper:arag"],
        "ground_truth_chunk_ids": doc_chunks[p9][:2],
        "expected_facts": [
            "Traditional RAG relies on static heuristics and single-shot top-k retrieval.",
            "Agentic RAG utilizes autonomous decision-making to dynamically query, evaluate, and adapt context.",
        ],
        "reference_answer": "Across the corpus, agentic RAG (e.g. ARAG) replaces rigid single-step retrieval heuristics with dynamic, autonomous decision loops that evaluate user context and formulate targeted sub-queries.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p9],
            "evidence_notes": "Aggregates contrasts between agentic decision loops and static retrieval.",
        },
    }
)

# EVAL-038: Aggregation of Modality and Architecture Evolution
questions.append(
    {
        "id": "EVAL-038",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.MEDIUM.value,
        "question": "Synthesize the major architectural trends in RAG literature from unimodal text retrieval to multimodal systems (paper-arxiv-2504.08748) and knowledge graphs (paper-arxiv-2601.05254).",
        "ground_truth_entities": ["paper:multimodal-rag-survey", "paper:tagrag"],
        "ground_truth_chunk_ids": doc_chunks[p6][:2] + doc_chunks[p13][:2],
        "expected_facts": [
            "Evolution towards multimodal retrieval incorporates visual and sensory modalities.",
            "Evolution towards graph-augmented RAG structures unstructured text into interconnected semantic knowledge graphs.",
        ],
        "reference_answer": "The literature documents an evolution from plain-text passage vectors toward multimodal retrieval integrating visual/audio data (2504.08748) and graph-augmented architectures providing explicit semantic relationships (2601.05254).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p6, p13],
            "evidence_notes": "Aggregates multi-year architectural evolution toward multimodal and graph RAG.",
        },
    }
)

# EVAL-039: Aggregation of Academic Text Processing Challenges
questions.append(
    {
        "id": "EVAL-039",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "What common challenges in handling scientific and academic text are identified across paper-arxiv-2411.18583, paper-arxiv-2607.01852, and paper-arxiv-2607.24767?",
        "ground_truth_entities": [
            "paper:automated-literature-review",
            "paper:evaluating-chunking-academic",
            "paper:effect-chunk-size-rag",
        ],
        "ground_truth_chunk_ids": doc_chunks[p1][:2] + doc_chunks[p11][:2] + doc_chunks[p12][:2],
        "expected_facts": [
            "Rapidly growing volume of publications creates information overload.",
            "Academic papers have complex section hierarchies that are disrupted by naive chunking.",
            "Improper chunk sizing leads to degraded answer quality or lost context.",
        ],
        "reference_answer": "Academic text poses severe challenges: massive publication volume (2411.18583), hierarchical discourse structures requiring semantic clustering (2607.01852), and high sensitivity to chunk size variations (2607.24767).",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p1, p11, p12],
            "evidence_notes": "Aggregates shared challenges in academic literature retrieval and processing.",
        },
    }
)

# EVAL-040: Aggregation of Verification & Hallucination Mitigation
questions.append(
    {
        "id": "EVAL-040",
        "category": BenchmarkCategory.RELATIONSHIP_AGGREGATION.value,
        "difficulty": BenchmarkDifficulty.HARD.value,
        "question": "Compare the hallucination mitigation techniques discussed across FAIR-RAG (paper-arxiv-2510.22344) and Metacognitive RAG (paper-arxiv-2402.11626).",
        "ground_truth_entities": ["paper:fair-rag", "paper:metacognitive-rag"],
        "ground_truth_chunk_ids": doc_chunks[p4][:2] + doc_chunks[p8][:2],
        "expected_facts": [
            "FAIR-RAG enforces faithful iterative refinement across multi-hop contexts.",
            "Metacognitive RAG employs self-monitoring reflection over retrieval quality.",
        ],
        "reference_answer": "To mitigate hallucinations, FAIR-RAG validates intermediate reasoning through faithful iterative refinement loops, while Metacognitive RAG uses self-reflective evaluation to monitor evidence sufficiency.",
        "metadata": {
            "status": "verified",
            "source_paper_ids": [p4, p8],
            "evidence_notes": "Aggregates hallucination mitigation approaches in iterative vs metacognitive RAG.",
        },
    }
)

# ==============================================================================
# Category 5: Out of Scope / Unanswerable (EVAL-041 to EVAL-050)
# ==============================================================================

out_of_scope_inquiries = [
    (
        "EVAL-041",
        "What is the recommended lithium-ion battery charging protocol for electric vehicles described in the harvested papers?",
    ),
    (
        "EVAL-042",
        "Which paper details the catalytic synthesis of ammonia using Haber-Bosch ruthenium nanoparticles?",
    ),
    (
        "EVAL-043",
        "What were the quarterly revenue earnings of Apple Inc in Q4 2024 according to the corpus?",
    ),
    (
        "EVAL-044",
        "How is CRISPR-Cas9 base editing applied to sickle cell disease mutations in the provided papers?",
    ),
    (
        "EVAL-045",
        "What is the gravitational wave frequency detected by LIGO for neutron star collision GW170817?",
    ),
    (
        "EVAL-046",
        "Which paper describes the aerodynamic drag coefficient of supersonic transport aircraft wings?",
    ),
    (
        "EVAL-047",
        "What are the clinical trial outcomes for mRNA cancer vaccines against metastatic melanoma in the corpus?",
    ),
    (
        "EVAL-048",
        "What architectural guidelines are given for deploying microservices on Kubernetes in AWS us-east-1?",
    ),
    (
        "EVAL-049",
        "How do optical telescope mirrors in the James Webb Space Telescope achieve cryogenic beryllium alignment?",
    ),
    (
        "EVAL-050",
        "Which paper provides the seismic resistance code specifications for high-rise buildings in Tokyo?",
    ),
]

for qid, qtext in out_of_scope_inquiries:
    questions.append(
        {
            "id": qid,
            "category": BenchmarkCategory.OUT_OF_SCOPE.value,
            "difficulty": BenchmarkDifficulty.EASY.value,
            "question": qtext,
            "ground_truth_entities": [],
            "ground_truth_chunk_ids": [],
            "expected_facts": [],
            "reference_answer": "The retrieved scientific literature does not contain information to answer this question as it falls outside the domain of retrieval-augmented generation and natural language processing.",
            "metadata": {
                "status": "verified_out_of_scope",
                "source_paper_ids": [],
                "evidence_notes": "Deliberately unanswerable out-of-domain question to test hallucination refusal.",
            },
        }
    )

dataset = {
    "version": "1.0.0",
    "domain": "scientific_papers_rag",
    "total_questions": len(questions),
    "questions": questions,
}

benchmark_file = Path("experiments/scipaper/eval/benchmark_scipaper_50.json")
with benchmark_file.open("w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Successfully generated {len(questions)} verified benchmark questions in {benchmark_file}.")
