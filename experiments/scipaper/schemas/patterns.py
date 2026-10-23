"""Deterministic Pattern Catalog and Gazetteers for Offline Scientific Extraction (Phase F).

Contains curated, high-precision domain dictionaries and regex patterns for:
- Method / Technique detection
- Model / Architecture detection
- Dataset / Benchmark detection
- Metric detection
- Task / Problem detection
- Research Organization detection

Pure offline, zero remote dependencies, zero model weights.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# 1. Gazetteers: Known entities with (display_name, canonical_slug, aliases)
# ---------------------------------------------------------------------------

GAZETTEER_METHODS: Final[list[tuple[str, str, list[str]]]] = [
    # Core RAG & Retrieval Methods
    (
        "Retrieval-Augmented Generation",
        "retrieval-augmented-generation",
        ["rag", "retrieval augmented generation", "retrieval-augmented text generation"],
    ),
    (
        "Dense Retrieval",
        "dense-retrieval",
        ["dense retrieval", "dense passage retrieval", "dpr", "dense vector retrieval"],
    ),
    ("Sparse Retrieval", "sparse-retrieval", ["sparse retrieval", "lexical retrieval"]),
    ("BM25", "bm25", ["bm25", "okapi bm25"]),
    (
        "Knowledge Graph",
        "knowledge-graph",
        ["knowledge graph", "kg", "knowledge graphs", "graphrag", "knowledge-graph"],
    ),
    (
        "Vector Search",
        "vector-search",
        ["vector search", "vector indexing", "vector similarity search"],
    ),
    (
        "Semantic Chunking",
        "semantic-chunking",
        ["semantic chunking", "cluster-based semantic chunking", "semantic chunk"],
    ),
    ("Attention Distillation", "attention-distillation", ["attention distillation"]),
    (
        "Iterative Refinement",
        "iterative-refinement",
        ["iterative refinement", "iterative refinement cycle", "adaptive iterative refinement"],
    ),
    ("Federated Learning", "federated-learning", ["federated learning", "federated rag"]),
    ("Reciprocal Rank Fusion", "reciprocal-rank-fusion", ["reciprocal rank fusion", "rrf"]),
    (
        "Cross-Encoder Reranking",
        "cross-encoder-reranking",
        ["cross-encoder", "cross encoder reranking", "reranker"],
    ),
    ("Contrastive Learning", "contrastive-learning", ["contrastive learning", "contrastive loss"]),
    ("LoRA", "lora", ["lora", "low-rank adaptation", "lora adapters"]),
    ("Natural Language Processing", "nlp", ["nlp", "natural language processing"]),
    (
        "Membership Inference Attack",
        "membership-inference-attack",
        ["membership inference", "membership inference attack", "interrogation attack"],
    ),
    ("Personalized PageRank", "personalized-pagerank", ["personalized pagerank", "ppr"]),
    (
        "Reinforcement Learning from Human Feedback",
        "rlhf",
        ["rlhf", "reinforcement learning from human feedback"],
    ),
]

GAZETTEER_MODELS: Final[list[tuple[str, str, list[str]]]] = [
    ("GPT-4", "gpt-4", ["gpt-4", "gpt-4o", "gpt4"]),
    ("GPT-3.5", "gpt-3.5", ["gpt-3.5", "gpt-3.5-turbo", "gpt3.5"]),
    ("BERT", "bert", ["bert", "bert-base", "bert-large"]),
    ("RoBERTa", "roberta", ["roberta"]),
    ("T5", "t5", ["t5", "simple t5"]),
    (
        "Llama 3",
        "llama-3",
        ["llama3", "llama-3", "llama 3", "llama3:70b", "llama3:8b", "llama3.2:3b", "llama3.2"],
    ),
    ("Gemma 2", "gemma-2", ["gemma2", "gemma-2", "gemma 2", "gemma2:27b", "gemma2:9b"]),
    ("Mistral", "mistral", ["mistral", "mixtral"]),
    ("Phi-3", "phi-3", ["phi3", "phi-3", "phi3:3.8b"]),
    ("BGE", "bge", ["bge", "bge-large", "bge-small"]),
    ("MiniLM", "minilm", ["minilm", "all-minilm-l6-v2"]),
    ("Qwen", "qwen", ["qwen", "qwq", "qwq:32b"]),
]

GAZETTEER_DATASETS: Final[list[tuple[str, str, list[str]]]] = [
    ("SciTLDR", "scitldr", ["scitldr", "scitldr dataset"]),
    ("HotpotQA", "hotpotqa", ["hotpotqa", "hotpot qa"]),
    ("2WikiMultiHopQA", "2wikimultihopqa", ["2wikimultihopqa", "2wiki"]),
    ("MusiQue", "musique", ["musique", "musique benchmark"]),
    ("MS MARCO", "ms-marco", ["ms marco", "msmarco"]),
    ("Natural Questions", "natural-questions", ["natural questions", "nq"]),
    ("TriviaQA", "triviaqa", ["triviaqa", "trivia qa"]),
    ("KILT", "kilt", ["kilt", "kilt benchmark"]),
    ("LaMP", "lamp", ["lamp", "lamp benchmark"]),
    ("UltraDomain", "ultradomain", ["ultradomain", "ultradomain datasets"]),
    ("MusWikiDB", "muswikidb", ["muswikidb"]),
]

GAZETTEER_METRICS: Final[list[tuple[str, str, list[str]]]] = [
    ("ROUGE", "rouge", ["rouge", "rouge-1", "rouge-2", "rouge-l", "rouge scores"]),
    ("BLEU", "bleu", ["bleu", "bleu score", "bleu-4"]),
    ("F1 Score", "f1-score", ["f1-score", "f1 score", "f1"]),
    ("Exact Match", "exact-match", ["exact match", "em score"]),
    ("Hit Rate", "hit-rate", ["hit rate", "hit@5", "hit@10", "hit@1"]),
    ("Recall", "recall", ["recall", "recall@5", "recall@10", "recall@1"]),
    ("Precision", "precision", ["precision", "precision@5", "precision@10"]),
    ("Mean Reciprocal Rank", "mrr", ["mrr", "mean reciprocal rank"]),
    ("NDCG", "ndcg", ["ndcg", "ndcg@5", "ndcg@10"]),
    ("Perplexity", "perplexity", ["perplexity"]),
    ("Faithfulness", "faithfulness", ["faithfulness", "grounding score", "answer faithfulness"]),
    ("Relevance", "relevance", ["context relevance", "answer relevance"]),
]

GAZETTEER_TASKS: Final[list[tuple[str, str, list[str]]]] = [
    (
        "Literature Review Generation",
        "literature-review-generation",
        [
            "literature review generation",
            "automated literature review",
            "generating literature reviews",
        ],
    ),
    (
        "Question Answering",
        "question-answering",
        ["question answering", "multi-hop qa", "document question answering", "qa tasks", "mqa"],
    ),
    (
        "Information Retrieval",
        "information-retrieval",
        ["information retrieval", "passage retrieval", "document search"],
    ),
    (
        "Personalized Recommendation",
        "personalized-recommendation",
        ["personalized recommendation", "recommendation systems"],
    ),
    (
        "Text Generation",
        "text-generation",
        ["text generation", "text-only music question answering"],
    ),
    ("Data Imputation", "data-imputation", ["data imputation"]),
    ("Membership Inference", "membership-inference", ["membership inference", "privacy leakage"]),
    (
        "Debate Evaluation",
        "debate-evaluation",
        ["retrieval-augmented debate", "evaluating debate responses"],
    ),
]

GAZETTEER_ORGS: Final[list[tuple[str, str, list[str]]]] = [
    ("arXiv", "arxiv", ["arxiv", "arxiv.org"]),
    ("OpenAI", "openai", ["openai"]),
    ("Google", "google", ["google"]),
    ("Meta", "meta", ["meta", "facebook ai"]),
    ("Anthropic", "anthropic", ["anthropic"]),
    ("Hugging Face", "huggingface", ["hugging face", "huggingface"]),
]


# ---------------------------------------------------------------------------
# 2. Structural & Relational Linguistic Patterns
# ---------------------------------------------------------------------------

# Patterns for Paper PROPOSES Method / Framework / Architecture
# e.g., "We propose TagRAG, a tag-guided hierarchical knowledge graph"
# e.g., "This paper introduces FAIR-RAG, a novel agentic framework"
PROPOSES_PATTERNS = [
    re.compile(
        r"(?:we|this paper|this work)\s+(?:propose|proposes|introduce|introduces|present|presents)\s+([A-Z][A-Za-z0-9\-]+)(?:[,\s]+(?:a|an|the)\s+([a-z\s\-]+?))?(?:[,\.]|\s+for|\s+designed)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9\-]{2,})\s+is\s+(?:proposed|introduced|presented)\s+(?:in this paper|in this work)?",
        re.IGNORECASE,
    ),
]

# Patterns for Paper EVALUATED_ON Dataset / Benchmark
# e.g., "Extensive experiments on UltraDomain datasets"
# e.g., "We conducted experiments on challenging multi-hop QA benchmarks, including HotpotQA"
EVALUATED_ON_PATTERNS = [
    re.compile(
        r"(?:experiments|evaluated|testing|benchmarked|evaluate)\s+(?:on|across|using)\s+(?:the\s+)?([A-Za-z0-9\-\s,]+?)(?:benchmark|dataset|benchmarks|datasets)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:benchmarks|datasets)[,\s]+including\s+([A-Za-z0-9\-\s,]+?)(?:\.|\sand\s|;)",
        re.IGNORECASE,
    ),
]

# Patterns for Paper REPORTS_METRIC Metric
# e.g., "The ROUGE scores are used for the evaluation"
# e.g., "achieves an F1-score of 0.453"
REPORTS_METRIC_PATTERNS = [
    re.compile(
        r"(?:achieves|achieved|reaches|reaching|yields|reports|improved|improvement of)\s+(?:an?\s+)?(?:average\s+)?([0-9\.]+%?)\s+(?:in\s+|on\s+)?([A-Za-z0-9\-]+)(?:\s+score)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Za-z0-9\-]+)\s+(?:score|scores|metric|metrics)\s+(?:are|is|were)?\s+(?:used|evaluated|measured|reported)",
        re.IGNORECASE,
    ),
]

# Patterns for Paper ADDRESSES_TASK Task
# e.g., "for personalized recommendation"
# e.g., "designed for question answering"
ADDRESSES_TASK_PATTERNS = [
    re.compile(
        r"(?:applied to|designed for|adapts? to|aims to automate|focuses on|addressed for)\s+([a-z\s\-]+?)(?:tasks?|benchmarks?|\.|$)",
        re.IGNORECASE,
    ),
]
