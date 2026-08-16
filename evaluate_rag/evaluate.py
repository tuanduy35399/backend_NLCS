"""Evaluate the project's RAG retrieval and answer generation with LlamaIndex.

Run this module from ``backend_NLCS``. Examples::

    python -m evaluate_rag.evaluate inspect --query "Ngành Kỹ thuật phần mềm học gì?"
    python -m evaluate_rag.evaluate run --dataset evaluate_rag/test_cases.json
    python -m evaluate_rag.evaluate run --dataset evaluate_rag/test_cases.json --config full

The evaluator is intentionally separate from the FastAPI application. It reuses the
same indexes, retrieval components, reranker, LLM and answer generator without
changing the production request flow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.core.evaluation.retrieval.metrics import (
    HitRate,
    MRR,
    NDCG,
    Precision,
    Recall,
)
from llama_index.core.indices.property_graph import PropertyGraphIndex

from graphRAG.indexing.embedding.bge_m3 import BGE_M3_Embedding
from graphRAG.indexing.graph.graph_store import GraphStore
from graphRAG.indexing.vector.chroma import ChromaStore
from graphRAG.llm.answer_generator import AnswerGenerator
from graphRAG.llm.custom import CustomLLM
from graphRAG.retrieval.reranker.bge_reranker import BGEReranker
from graphRAG.retrieval.rewrite.hyde import HyDE
from graphRAG.retrieval.rewrite.query_rewrite import QueryRewrite
from graphRAG.retrieval.search.graph_search import GraphSearch
from graphRAG.retrieval.search.hybrid_search import HybridSearch
from graphRAG.retrieval.search.vector_search import VectorSearch


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "results"
CONFIG_NAMES = ("vector", "graph", "hybrid", "full")


@dataclass(frozen=True)
class TestCase:
    """One labelled RAG evaluation example."""

    case_id: str
    query: str
    expected_ids: tuple[str, ...]
    expected_urls: tuple[str, ...] = ()
    group_major: str = ""
    reference_answer: str = ""
    difficulty: str = ""
    category: str = ""
    unanswerable: bool = False

    @property
    def retrieval_query(self) -> str:
        if not self.group_major:
            return self.query
        return (
            f"NHÓM NGÀNH NGƯỜI DÙNG ĐÃ CHỌN: {self.group_major}\n"
            f"NỘI DUNG NGƯỜI DÙNG: {self.query}"
        )


@dataclass
class RagComponents:
    """Runtime components shared by all evaluated configurations."""

    llm: Any
    vector_search: VectorSearch | None
    graph_search: GraphSearch | None
    hybrid_search: HybridSearch | None
    reranker: BGEReranker | None
    rewriter: QueryRewrite | None
    hyde: HyDE | None
    answer_generator: AnswerGenerator | None


def load_test_cases(path: Path) -> list[TestCase]:
    """Load and validate the labelled JSON evaluation dataset."""

    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Không tìm thấy tập đánh giá: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON không hợp lệ trong {path}: {exc}") from exc

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Tập đánh giá phải là một JSON array không rỗng.")

    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Test case thứ {index} phải là một JSON object.")

        case_id = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        expected_ids = tuple(
            str(node_id).strip()
            for node_id in item.get("expected_ids", [])
            if str(node_id).strip()
        )
        expected_urls = tuple(
            str(url).strip()
            for url in item.get("expected_urls", [])
            if str(url).strip()
        )

        if not case_id or not query:
            raise ValueError(f"Test case thứ {index} thiếu 'id' hoặc 'query'.")
        if case_id in seen_ids:
            raise ValueError(f"ID test case bị trùng: {case_id}")
        unanswerable = bool(item.get("unanswerable", False))
        if not unanswerable and not expected_ids and not expected_urls:
            raise ValueError(
                f"Test case '{case_id}' cần expected_ids hoặc expected_urls."
            )
        if any(node_id.startswith("REPLACE_WITH_") for node_id in expected_ids):
            raise ValueError(
                f"Test case '{case_id}' vẫn chứa node ID mẫu. "
                "Hãy chạy lệnh inspect và gán node ID đúng trước khi đánh giá."
            )

        seen_ids.add(case_id)
        cases.append(
            TestCase(
                case_id=case_id,
                query=query,
                expected_ids=expected_ids,
                expected_urls=expected_urls,
                group_major=str(item.get("group_major", "")).strip(),
                reference_answer=str(item.get("reference_answer", "")).strip(),
                difficulty=str(item.get("difficulty", "")).strip(),
                category=str(item.get("category", "")).strip(),
                unanswerable=unanswerable,
            )
        )

    return cases


def build_components(
    top_k: int,
    configs: Sequence[str],
    evaluate_answers: bool,
) -> RagComponents:
    """Build only the dependencies required by the selected configurations."""

    llm = CustomLLM().get_llm()
    embed_model = BGE_M3_Embedding().get_model()
    needs_vector = any(name in {"vector", "hybrid", "full"} for name in configs)
    needs_graph = any(name in {"graph", "hybrid", "full"} for name in configs)
    needs_hybrid = any(name in {"hybrid", "full"} for name in configs)
    needs_full = "full" in configs

    vector_search = None
    if needs_vector:
        chroma_store = ChromaStore()
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=chroma_store.vector_store,
            embed_model=embed_model,
        )
        vector_search = VectorSearch(vector_index=vector_index, top_k=top_k)

    graph_search = None
    if needs_graph:
        graph_store = GraphStore().get_store()
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm,
            embed_model=embed_model,
        )
        graph_search = GraphSearch(graph_index=graph_index, top_k=top_k)

    return RagComponents(
        llm=llm,
        vector_search=vector_search,
        graph_search=graph_search,
        hybrid_search=HybridSearch() if needs_hybrid else None,
        reranker=BGEReranker(top_k=top_k) if needs_full else None,
        rewriter=QueryRewrite(llm) if needs_full else None,
        hyde=HyDE(llm) if needs_full else None,
        answer_generator=AnswerGenerator(llm=llm) if evaluate_answers else None,
    )


def require_component(component: Any, name: str) -> Any:
    if component is None:
        raise RuntimeError(f"Thành phần '{name}' chưa được khởi tạo cho cấu hình này.")
    return component


def retrieve_for_config(
    components: RagComponents,
    config_name: str,
    query: str,
) -> list[Any]:
    """Run one retrieval configuration for an ablation comparison."""

    if config_name == "vector":
        vector_search = require_component(components.vector_search, "vector_search")
        return list(vector_search.search(query))

    if config_name == "graph":
        graph_search = require_component(components.graph_search, "graph_search")
        return list(graph_search.search(query))

    if config_name == "hybrid":
        vector_search = require_component(components.vector_search, "vector_search")
        graph_search = require_component(components.graph_search, "graph_search")
        hybrid_search = require_component(components.hybrid_search, "hybrid_search")
        vector_results = vector_search.search(query)
        graph_results = graph_search.search(query)
        merged_results = hybrid_search.merge(
            vector_results,
            graph_results,
        )
        return list(merged_results[: vector_search.top_k])

    if config_name == "full":
        vector_search = require_component(components.vector_search, "vector_search")
        graph_search = require_component(components.graph_search, "graph_search")
        hybrid_search = require_component(components.hybrid_search, "hybrid_search")
        rewriter = require_component(components.rewriter, "rewriter")
        hyde = require_component(components.hyde, "hyde")
        reranker = require_component(components.reranker, "reranker")
        rewritten_query = rewriter.rewrite(query)
        hyde_query = hyde.generate(rewritten_query)
        vector_results = vector_search.search(hyde_query)
        graph_results = graph_search.search(rewritten_query)
        merged_results = hybrid_search.merge(
            vector_results,
            graph_results,
        )
        return list(reranker.rerank(rewritten_query, merged_results[:20]))

    raise ValueError(f"Cấu hình không hợp lệ: {config_name}")


def node_id(item: Any) -> str:
    node = getattr(item, "node", item)
    return str(getattr(node, "node_id", getattr(node, "id_", "")))


def node_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if text:
        return str(text)
    node = getattr(item, "node", item)
    get_content = getattr(node, "get_content", None)
    return str(get_content() if callable(get_content) else node)


def node_metadata(item: Any) -> dict[str, Any]:
    node = getattr(item, "node", item)
    metadata = getattr(node, "metadata", {}) or {}
    return dict(metadata)


def retrieval_identity(item: Any, use_urls: bool) -> str:
    """Return a stable URL identity when the test case is labelled by source."""

    if use_urls:
        return str(node_metadata(item).get("url", "")).strip()
    return node_id(item)


def finite_score(value: Any) -> float | None:
    """Normalize evaluator outputs so they can be serialized safely."""

    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def compute_retrieval_metrics(
    query: str,
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    retrieved_texts: Sequence[str],
) -> dict[str, float | None]:
    """Compute LlamaIndex retrieval metrics over the returned Top-K list."""

    metrics = {
        "hit_rate": HitRate(),
        "mrr": MRR(),
        "precision": Precision(),
        "recall": Recall(),
        "ndcg": NDCG(),
    }
    results: dict[str, float | None] = {}
    for name, metric in metrics.items():
        result = metric.compute(
            query=query,
            expected_ids=list(expected_ids),
            retrieved_ids=list(retrieved_ids),
            retrieved_texts=list(retrieved_texts),
        )
        results[name] = finite_score(result.score)
    return results


def answer_as_text(answer: Any) -> str:
    """Convert the structured application answer into evaluator-friendly text."""

    if not isinstance(answer, dict):
        return str(answer)

    preferred_fields = (
        "ten_nganh",
        "mo_ta_nganh",
        "ly_do_phu_hop",
        "thong_bao_dinh_huong",
        "goi_y_tiep_theo",
        "noi_dung_tra_loi",
    )
    parts = [str(answer.get(field, "")).strip() for field in preferred_fields]
    return "\n\n".join(part for part in parts if part)


def evaluate_generation(
    components: RagComponents,
    test_case: TestCase,
    retrieved_nodes: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, float | None], dict[str, str]]:
    """Generate one answer and score it with LlamaIndex LLM evaluators."""

    answer_generator = require_component(
        components.answer_generator,
        "answer_generator",
    )
    answer = answer_generator.generate(
        test_case.retrieval_query,
        retrieved_nodes,
        history=[],
    )
    response_text = answer_as_text(answer)
    contexts = [node_text(item) for item in retrieved_nodes]

    evaluators: list[tuple[str, Any, dict[str, Any]]] = [
        (
            "faithfulness",
            FaithfulnessEvaluator(llm=components.llm),
            {
                "query": test_case.query,
                "response": response_text,
                "contexts": contexts,
            },
        ),
        (
            "answer_relevancy",
            AnswerRelevancyEvaluator(llm=components.llm),
            {
                "query": test_case.query,
                "response": response_text,
                "contexts": contexts,
            },
        ),
    ]

    if test_case.reference_answer:
        evaluators.append(
            (
                "correctness",
                CorrectnessEvaluator(llm=components.llm),
                {
                    "query": test_case.query,
                    "response": response_text,
                    "reference": test_case.reference_answer,
                },
            )
        )

    scores: dict[str, float | None] = {}
    feedback: dict[str, str] = {}
    for name, evaluator, kwargs in evaluators:
        result = evaluator.evaluate(**kwargs)
        scores[name] = finite_score(result.score)
        feedback[name] = str(result.feedback or "")

    return answer, scores, feedback


def evaluate_case(
    components: RagComponents,
    test_case: TestCase,
    config_name: str,
    evaluate_answers: bool,
) -> dict[str, Any]:
    """Evaluate a single test case and return a JSON-serializable record."""

    started = time.perf_counter()
    retrieved_nodes = retrieve_for_config(
        components,
        config_name,
        test_case.retrieval_query,
    )
    retrieval_seconds = time.perf_counter() - started

    use_urls = bool(test_case.expected_urls)
    expected_identities = (
        test_case.expected_urls if use_urls else test_case.expected_ids
    )
    metric_pairs: list[tuple[str, str]] = []
    seen_identities: set[str] = set()
    for item in retrieved_nodes:
        identity = retrieval_identity(item, use_urls=use_urls)
        if not identity or identity in seen_identities:
            continue
        seen_identities.add(identity)
        metric_pairs.append((identity, node_text(item)))
    retrieved_ids = [identity for identity, _ in metric_pairs]
    retrieved_texts = [text for _, text in metric_pairs]
    if test_case.unanswerable:
        retrieval_metrics = {
            "hit_rate": None,
            "mrr": None,
            "precision": None,
            "recall": None,
            "ndcg": None,
        }
    else:
        retrieval_metrics = compute_retrieval_metrics(
            query=test_case.retrieval_query,
            expected_ids=expected_identities,
            retrieved_ids=retrieved_ids,
            retrieved_texts=retrieved_texts,
        )

    answer: dict[str, Any] | None = None
    generation_metrics: dict[str, float | None] = {}
    generation_feedback: dict[str, str] = {}
    generation_seconds = 0.0

    if evaluate_answers:
        generation_started = time.perf_counter()
        answer, generation_metrics, generation_feedback = evaluate_generation(
            components,
            test_case,
            retrieved_nodes,
        )
        generation_seconds = time.perf_counter() - generation_started

    return {
        "case_id": test_case.case_id,
        "config": config_name,
        "query": test_case.query,
        "group_major": test_case.group_major,
        "difficulty": test_case.difficulty,
        "category": test_case.category,
        "unanswerable": test_case.unanswerable,
        "expected_ids": list(test_case.expected_ids),
        "expected_urls": list(test_case.expected_urls),
        "retrieved": [
            {
                "rank": rank,
                "node_id": node_id(item),
                "score": finite_score(getattr(item, "score", None)),
                "metadata": node_metadata(item),
                "text": node_text(item),
            }
            for rank, item in enumerate(retrieved_nodes, start=1)
        ],
        "retrieval_metrics": retrieval_metrics,
        "answer": answer,
        "generation_metrics": generation_metrics,
        "generation_feedback": generation_feedback,
        "latency_seconds": {
            "retrieval": round(retrieval_seconds, 4),
            "generation_and_judging": round(generation_seconds, 4),
            "total": round(retrieval_seconds + generation_seconds, 4),
        },
    }


def mean_present(records: Iterable[dict[str, Any]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for record in records:
        current: Any = record
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            values.append(float(current))
    return statistics.fmean(values) if values else None


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-case results by evaluated configuration."""

    summaries: list[dict[str, Any]] = []
    for config_name in CONFIG_NAMES:
        config_records = [r for r in records if r["config"] == config_name]
        if not config_records:
            continue
        summaries.append(
            {
                "config": config_name,
                "test_cases": len(config_records),
                "answerable_cases": sum(
                    not record.get("unanswerable", False)
                    for record in config_records
                ),
                "unanswerable_cases": sum(
                    record.get("unanswerable", False)
                    for record in config_records
                ),
                "hit_rate": mean_present(
                    config_records, ("retrieval_metrics", "hit_rate")
                ),
                "mrr": mean_present(config_records, ("retrieval_metrics", "mrr")),
                "precision": mean_present(
                    config_records, ("retrieval_metrics", "precision")
                ),
                "recall": mean_present(
                    config_records, ("retrieval_metrics", "recall")
                ),
                "ndcg": mean_present(config_records, ("retrieval_metrics", "ndcg")),
                "faithfulness": mean_present(
                    config_records, ("generation_metrics", "faithfulness")
                ),
                "answer_relevancy": mean_present(
                    config_records, ("generation_metrics", "answer_relevancy")
                ),
                "correctness": mean_present(
                    config_records, ("generation_metrics", "correctness")
                ),
                "retrieval_seconds": mean_present(
                    config_records, ("latency_seconds", "retrieval")
                ),
                "generation_and_judging_seconds": mean_present(
                    config_records,
                    ("latency_seconds", "generation_and_judging"),
                ),
                "total_seconds": mean_present(
                    config_records, ("latency_seconds", "total")
                ),
            }
        )
    return summaries


def write_results(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write detailed JSON plus CSV and Markdown summary tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = output_dir / f"rag_evaluation_{timestamp}.json"
    summary_path = output_dir / f"rag_evaluation_summary_{timestamp}.csv"
    markdown_path = output_dir / f"rag_evaluation_summary_{timestamp}.md"

    details_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summaries = summarize(records)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    table_columns = (
        "config",
        "test_cases",
        "answerable_cases",
        "unanswerable_cases",
        "hit_rate",
        "mrr",
        "precision",
        "recall",
        "ndcg",
        "faithfulness",
        "answer_relevancy",
        "correctness",
        "total_seconds",
    )

    def display(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(table_columns) + " |"
    separator = "| " + " | ".join("---" for _ in table_columns) + " |"
    rows = [
        "| "
        + " | ".join(display(summary.get(column)) for column in table_columns)
        + " |"
        for summary in summaries
    ]
    markdown_path.write_text(
        "# RAG evaluation summary\n\n"
        + "\n".join([header, separator, *rows])
        + "\n",
        encoding="utf-8",
    )

    print("\nRAG EVALUATION SUMMARY")
    print("\n".join([header, separator, *rows]))

    return details_path, summary_path, markdown_path


def inspect_nodes(components: RagComponents, query: str, config_name: str) -> None:
    """Print retrievable node IDs so a human can label expected_ids."""

    results = retrieve_for_config(components, config_name, query)
    if not results:
        print("Không tìm thấy node nào.")
        return

    for rank, item in enumerate(results, start=1):
        metadata = node_metadata(item)
        preview = " ".join(node_text(item).split())[:240]
        print(f"\n[{rank}] node_id={node_id(item)}")
        print(f"score={getattr(item, 'score', None)}")
        print(f"metadata={json.dumps(metadata, ensure_ascii=False)}")
        print(f"text={preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá Hybrid RAG bằng LlamaIndex Evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Xem node ID được retrieve để gán nhãn expected_ids.",
    )
    inspect_parser.add_argument("--query", required=True)
    inspect_parser.add_argument("--config", choices=CONFIG_NAMES, default="full")
    inspect_parser.add_argument("--top-k", type=int, default=5)

    run_parser = subparsers.add_parser("run", help="Chạy bộ đánh giá đã gán nhãn.")
    run_parser.add_argument("--dataset", type=Path, required=True)
    run_parser.add_argument(
        "--config",
        choices=(*CONFIG_NAMES, "all"),
        default="all",
    )
    run_parser.add_argument("--top-k", type=int, default=5)
    run_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Không sinh câu trả lời và không chạy LLM judge.",
    )
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k phải lớn hơn hoặc bằng 1.")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.command == "inspect":
        components = build_components(
            top_k=args.top_k,
            configs=(args.config,),
            evaluate_answers=False,
        )
        inspect_nodes(components, args.query, args.config)
        return

    test_cases = load_test_cases(args.dataset.resolve())
    configs = CONFIG_NAMES if args.config == "all" else (args.config,)
    components = build_components(
        top_k=args.top_k,
        configs=configs,
        evaluate_answers=not args.retrieval_only,
    )
    records: list[dict[str, Any]] = []

    for config_name in configs:
        for index, test_case in enumerate(test_cases, start=1):
            print(
                f"[{config_name}] {index}/{len(test_cases)} "
                f"- {test_case.case_id}"
            )
            try:
                records.append(
                    evaluate_case(
                        components=components,
                        test_case=test_case,
                        config_name=config_name,
                        evaluate_answers=not args.retrieval_only,
                    )
                )
            except Exception as exc:
                print(f"Lỗi tại {config_name}/{test_case.case_id}: {exc}")
                records.append(
                    {
                        "case_id": test_case.case_id,
                        "config": config_name,
                        "query": test_case.query,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    successful_records = [record for record in records if "error" not in record]
    if not successful_records:
        raise RuntimeError("Không có test case nào được đánh giá thành công.")

    details_path, summary_path, markdown_path = write_results(
        successful_records,
        args.output_dir,
    )
    print(f"\nChi tiết: {details_path.resolve()}")
    print(f"Tổng hợp: {summary_path.resolve()}")
    print(f"Bảng Markdown: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
