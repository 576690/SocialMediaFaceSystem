import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark import export_benchmark_results
from core.benchmark import build_controlled_face_quality_grid
from core.benchmark import build_face_quality_grid
from core.benchmark import evaluate_face_quality_grid
from core.benchmark import ensure_lfw_deepfunneled_dataset
from core.benchmark import extract_dataset_embeddings
from core.benchmark import get_runtime_device_report
from core.benchmark import run_benchmark_suite
from core.benchmark import summarize_benchmark


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)
logging.getLogger("core").setLevel(logging.INFO)
for noisy_logger_name in ("faiss", "faiss.loader", "httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)


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
    parser.add_argument(
        "--dataset-path",
        default=str(REPO_ROOT / "storage" / "datasets" / "lfw-deepfunneled"),
        help="Root directory where each subdirectory is one identity.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for CSV outputs and cache files. Defaults to a timestamped LFW benchmark directory.",
    )
    parser.add_argument("--algorithms", default="dbscan,hdbscan")
    parser.add_argument("--metrics", default="cosine,euclidean")
    parser.add_argument("--eps-grid", default="0.3,0.4,0.5")
    parser.add_argument("--min-samples-grid", default="2,3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-backend", choices=["auto", "torch-cuda", "numpy"], default="auto")
    parser.add_argument("--skip-quality-grid", action="store_true")
    parser.add_argument("--quality-grid-preset", choices=["controlled", "default"], default="controlled")
    parser.add_argument("--quality-min-face-sizes", default="40,56,72")
    parser.add_argument("--quality-min-face-ratios", default="0.02,0.035,0.05")
    parser.add_argument("--quality-min-laplacian-vars", default="0,50,80,120")
    parser.add_argument("--quality-max-pose-deviations", default="0.25,0.35,0.45")
    parser.add_argument("--no-download-dataset", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def _default_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "storage" / "benchmarks" / f"lfw_deepfunneled_full_{timestamp}"


def _write_benchmark_run(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = output_dir / "benchmark_run.json"
    started_at = time.perf_counter()

    algorithms = _parse_csv_values(args.algorithms, str)
    metrics = _parse_csv_values(args.metrics, str)
    eps_grid = _parse_csv_values(args.eps_grid, float)
    min_samples_grid = _parse_csv_values(args.min_samples_grid, int)
    min_face_sizes = _parse_csv_values(args.quality_min_face_sizes, int)
    min_face_ratios = _parse_csv_values(args.quality_min_face_ratios, float)
    min_laplacian_vars = _parse_csv_values(args.quality_min_laplacian_vars, float)
    max_pose_deviations = _parse_csv_values(args.quality_max_pose_deviations, float)

    dataset_status = ensure_lfw_deepfunneled_dataset(
        args.dataset_path,
        download_if_missing=not args.no_download_dataset,
    )
    device_report = get_runtime_device_report()
    run_metadata = {
        "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": algorithms,
        "metrics": metrics,
        "eps_grid": eps_grid,
        "min_samples_grid": min_samples_grid,
        "top_k": max(int(args.top_k), 1),
        "retrieval_backend": args.retrieval_backend,
        "quality_grid_preset": args.quality_grid_preset,
        "skip_quality_grid": bool(args.skip_quality_grid),
        "dataset_status": dataset_status,
        "device_report": device_report,
    }
    _write_benchmark_run(run_json_path, run_metadata)

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
        parser.error("数据集中有效人脸向量不足，无法运行基准测试。")

    benchmark_payload = run_benchmark_suite(
        dataset_payload["embeddings"],
        dataset_payload["label_ids"],
        algorithms=algorithms,
        metrics=metrics,
        eps_grid=eps_grid,
        min_samples_grid=min_samples_grid,
        top_k=max(int(args.top_k), 1),
        retrieval_backend=args.retrieval_backend,
    )
    quality_payload = {"results": None, "recommended_face_quality": None}
    if not args.skip_quality_grid:
        if args.quality_grid_preset == "controlled":
            quality_grid = build_controlled_face_quality_grid()
        else:
            quality_grid = build_face_quality_grid(
                min_face_size_values=min_face_sizes,
                min_face_ratio_values=min_face_ratios,
                min_laplacian_var_values=min_laplacian_vars,
                max_pose_deviation_values=max_pose_deviations,
            )
        quality_payload = evaluate_face_quality_grid(
            args.dataset_path,
            algorithms=algorithms,
            metrics=metrics,
            eps_grid=eps_grid,
            min_samples_grid=min_samples_grid,
            top_k=max(int(args.top_k), 1),
            quality_grid=quality_grid,
            cache_dir=quality_cache_dir,
            refresh_cache=args.refresh_cache,
            retrieval_backend=args.retrieval_backend,
        )
    exported = export_benchmark_results(
        output_dir,
        benchmark_payload["clustering_results"],
        benchmark_payload["retrieval_results"],
        dataset_payload["failures"],
        quality_results=quality_payload["results"],
    )

    summary = summarize_benchmark(
        benchmark_payload["clustering_results"],
        benchmark_payload["retrieval_results"],
        dataset_payload["failures"],
        benchmark_payload["elapsed_seconds"],
        quality_results=quality_payload["results"],
        recommended_face_quality=quality_payload["recommended_face_quality"],
    )
    logger.info("基准测试完成：\n%s", summary)
    run_metadata.update(
        {
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "embedding_cache": str(cache_path),
            "outputs": exported,
            "samples_kept": int(len(dataset_payload["embeddings"])),
            "failed_samples": int(len(dataset_payload["failures"])),
            "from_cache": bool(dataset_payload.get("from_cache")),
            "recommended_face_quality": quality_payload["recommended_face_quality"],
            "retrieval_results": benchmark_payload["retrieval_results"],
        }
    )
    _write_benchmark_run(run_json_path, run_metadata)
    logger.info(
        "结果文件：embedding_cache=%s clustering_results=%s retrieval_results=%s failed_samples=%s quality_filter_results=%s benchmark_run=%s",
        cache_path,
        exported["clustering"],
        exported["retrieval"],
        exported["failures"],
        exported["quality"],
        run_json_path,
    )


if __name__ == "__main__":
    main()
