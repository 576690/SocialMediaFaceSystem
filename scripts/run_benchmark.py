import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark import export_benchmark_results
from core.benchmark import build_face_quality_grid
from core.benchmark import evaluate_face_quality_grid
from core.benchmark import extract_dataset_embeddings
from core.benchmark import run_benchmark_suite
from core.benchmark import summarize_benchmark


def _parse_csv_values(raw_value, caster):
    values = []
    for item in str(raw_value or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(caster(item))
    return values


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run clustering and retrieval benchmarks on a directory-organized face dataset.",
    )
    parser.add_argument("--dataset-path", required=True, help="Root directory where each subdirectory is one identity.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV outputs and cache files.")
    parser.add_argument("--algorithms", default="dbscan,hdbscan,optics")
    parser.add_argument("--metrics", default="cosine,euclidean")
    parser.add_argument("--eps-grid", default="0.3,0.4,0.5")
    parser.add_argument("--min-samples-grid", default="2,3,4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quality-min-face-sizes", default="40,56,72")
    parser.add_argument("--quality-min-laplacian-vars", default="0,50,80,120")
    parser.add_argument("--quality-max-pose-deviations", default="0.25,0.35,0.45")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = _parse_csv_values(args.algorithms, str)
    metrics = _parse_csv_values(args.metrics, str)
    eps_grid = _parse_csv_values(args.eps_grid, float)
    min_samples_grid = _parse_csv_values(args.min_samples_grid, int)
    min_face_sizes = _parse_csv_values(args.quality_min_face_sizes, int)
    min_laplacian_vars = _parse_csv_values(args.quality_min_laplacian_vars, float)
    max_pose_deviations = _parse_csv_values(args.quality_max_pose_deviations, float)

    cache_path = output_dir / "embedding_cache.npz"
    failures_path = output_dir / "failed_samples.csv"
    quality_cache_dir = output_dir / "quality_cache"

    dataset_payload = extract_dataset_embeddings(
        args.dataset_path,
        cache_path=cache_path,
        failures_path=failures_path,
        refresh_cache=args.refresh_cache,
    )
    if len(dataset_payload["embeddings"]) < 2:
        parser.error("Not enough valid face embeddings were extracted from the dataset.")

    benchmark_payload = run_benchmark_suite(
        dataset_payload["embeddings"],
        dataset_payload["label_ids"],
        algorithms=algorithms,
        metrics=metrics,
        eps_grid=eps_grid,
        min_samples_grid=min_samples_grid,
        top_k=max(int(args.top_k), 1),
    )
    quality_payload = evaluate_face_quality_grid(
        args.dataset_path,
        algorithms=algorithms,
        metrics=metrics,
        eps_grid=eps_grid,
        min_samples_grid=min_samples_grid,
        top_k=max(int(args.top_k), 1),
        quality_grid=build_face_quality_grid(
            min_face_size_values=min_face_sizes,
            min_laplacian_var_values=min_laplacian_vars,
            max_pose_deviation_values=max_pose_deviations,
        ),
        cache_dir=quality_cache_dir,
        refresh_cache=args.refresh_cache,
    )
    exported = export_benchmark_results(
        output_dir,
        benchmark_payload["clustering_results"],
        benchmark_payload["retrieval_results"],
        dataset_payload["failures"],
        quality_results=quality_payload["results"],
    )

    print(
        summarize_benchmark(
            benchmark_payload["clustering_results"],
            benchmark_payload["retrieval_results"],
            dataset_payload["failures"],
            benchmark_payload["elapsed_seconds"],
            quality_results=quality_payload["results"],
            recommended_face_quality=quality_payload["recommended_face_quality"],
        )
    )
    print(f"embedding_cache: {cache_path}")
    print(f"clustering_results: {exported['clustering']}")
    print(f"retrieval_results: {exported['retrieval']}")
    print(f"failed_samples: {exported['failures']}")
    print(f"quality_filter_results: {exported['quality']}")


if __name__ == "__main__":
    main()
