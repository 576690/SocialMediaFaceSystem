import csv
import inspect
import json
import time
from itertools import product
from pathlib import Path

import numpy as np

from core.clustering import evaluate_embedding_clusters, evaluate_embedding_retrieval


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_FACE_QUALITY_GRID = {
    "min_face_size": [40, 56, 72],
    "min_face_ratio": [0.02, 0.035, 0.05],
    "min_laplacian_var": [0, 50, 80, 120],
    "max_pose_deviation": [0.25, 0.35, 0.45],
    "blur_eval_size": [96],
}


def discover_identity_dataset(dataset_path):
    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_root}")

    identities = []
    for identity_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        image_paths = sorted(
            path for path in identity_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        identities.append(
            {
                "identity": identity_dir.name,
                "identity_dir": str(identity_dir),
                "image_paths": [str(path.resolve()) for path in image_paths],
            }
        )
    return identities


def normalize_face_quality_config(face_quality_config=None):
    raw = face_quality_config or {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "min_face_size": int(raw.get("min_face_size", 56)),
        "min_face_ratio": float(raw.get("min_face_ratio", 0.035)),
        "min_laplacian_var": float(raw.get("min_laplacian_var", 80.0)),
        "max_pose_deviation": float(raw.get("max_pose_deviation", 0.35)),
        "blur_eval_size": int(raw.get("blur_eval_size", 96)),
    }


def build_face_quality_grid(
    min_face_size_values=None,
    min_face_ratio_values=None,
    min_laplacian_var_values=None,
    max_pose_deviation_values=None,
    blur_eval_size_values=None,
):
    sizes = list(min_face_size_values or DEFAULT_FACE_QUALITY_GRID["min_face_size"])
    ratios = list(
        min_face_ratio_values or DEFAULT_FACE_QUALITY_GRID["min_face_ratio"]
    )
    blurs = list(
        min_laplacian_var_values or DEFAULT_FACE_QUALITY_GRID["min_laplacian_var"]
    )
    poses = list(
        max_pose_deviation_values or DEFAULT_FACE_QUALITY_GRID["max_pose_deviation"]
    )
    blur_eval_sizes = list(
        blur_eval_size_values or DEFAULT_FACE_QUALITY_GRID["blur_eval_size"]
    )
    return [
        normalize_face_quality_config(
            {
                "enabled": True,
                "min_face_size": size,
                "min_face_ratio": ratio,
                "min_laplacian_var": blur,
                "max_pose_deviation": pose,
                "blur_eval_size": blur_eval_size,
            }
        )
        for size, ratio, blur, pose, blur_eval_size in product(
            sizes,
            ratios,
            blurs,
            poses,
            blur_eval_sizes,
        )
    ]


def _default_embedding_extractor(image_path, face_quality_config=None):
    from core.analyzer import AIProcessor

    extractor = _default_embedding_extractor._instance
    if extractor is None:
        extractor = AIProcessor()
        _default_embedding_extractor._instance = extractor

    result = extractor.get_face_embedding_result_from_path(
        image_path,
        face_quality_config=normalize_face_quality_config(face_quality_config),
    )
    if result["embedding"] is None:
        return None, result["failure_reason"] or "embedding_failed"
    return np.asarray(result["embedding"], dtype=np.float32), None


_default_embedding_extractor._instance = None


def load_embedding_cache(cache_path):
    cache_path = Path(cache_path)
    payload = np.load(cache_path, allow_pickle=True)
    meta_path = cache_path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
    return {
        "embeddings": np.asarray(payload["embeddings"], dtype=np.float32),
        "label_ids": np.asarray(payload["label_ids"], dtype=np.int32),
        "label_names": payload["label_names"].tolist(),
        "image_paths": payload["image_paths"].tolist(),
        "meta": meta,
    }


def save_embedding_cache(
    cache_path,
    embeddings,
    label_ids,
    label_names,
    image_paths,
    dataset_path,
    face_quality_config=None,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        label_ids=np.asarray(label_ids, dtype=np.int32),
        label_names=np.asarray(label_names, dtype=object),
        image_paths=np.asarray(image_paths, dtype=object),
    )
    meta_path = cache_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_path": str(Path(dataset_path).resolve()),
                "samples": int(len(image_paths)),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "face_quality_config": normalize_face_quality_config(face_quality_config),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def extract_dataset_embeddings(
    dataset_path,
    cache_path=None,
    failures_path=None,
    refresh_cache=False,
    embedding_extractor=None,
    face_quality_config=None,
):
    cache_path = Path(cache_path) if cache_path else None
    failures_path = Path(failures_path) if failures_path else None
    normalized_face_quality = normalize_face_quality_config(face_quality_config)
    if cache_path and cache_path.exists() and not refresh_cache:
        cached = load_embedding_cache(cache_path)
        cached_meta = cached.get("meta", {})
        if cached_meta.get("dataset_path") == str(Path(dataset_path).resolve()) and cached_meta.get(
            "face_quality_config"
        ) == normalized_face_quality:
            cached["failures"] = []
            cached["from_cache"] = True
            return cached

    if embedding_extractor is None:
        embedding_extractor = lambda image_path: _default_embedding_extractor(
            image_path,
            face_quality_config=normalized_face_quality,
        )
    identities = discover_identity_dataset(dataset_path)
    embeddings = []
    label_ids = []
    label_names = []
    image_paths = []
    failures = []

    for label_id, identity in enumerate(identities):
        for image_path in identity["image_paths"]:
            try:
                if len(inspect.signature(embedding_extractor).parameters) >= 2:
                    embedding, failure_reason = embedding_extractor(
                        image_path,
                        normalized_face_quality,
                    )
                else:
                    embedding, failure_reason = embedding_extractor(image_path)
            except Exception as exc:
                embedding = None
                failure_reason = str(exc)

            if embedding is None:
                failures.append(
                    {
                        "identity": identity["identity"],
                        "image_path": image_path,
                        "reason": failure_reason or "embedding_failed",
                    }
                )
                continue

            embeddings.append(np.asarray(embedding, dtype=np.float32))
            label_ids.append(label_id)
            label_names.append(identity["identity"])
            image_paths.append(image_path)

    payload = {
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "label_ids": np.asarray(label_ids, dtype=np.int32),
        "label_names": label_names,
        "image_paths": image_paths,
        "failures": failures,
        "from_cache": False,
        "face_quality_config": normalized_face_quality,
    }

    if cache_path:
        save_embedding_cache(
            cache_path,
            payload["embeddings"],
            payload["label_ids"],
            payload["label_names"],
            payload["image_paths"],
            dataset_path,
            face_quality_config=normalized_face_quality,
        )
    if failures_path:
        write_csv(
            failures_path,
            failures,
            fieldnames=["identity", "image_path", "reason"],
        )
    return payload


def _best_cluster_metrics(clustering_results):
    valid_results = [row for row in clustering_results if not row.get("error")]
    if not valid_results:
        return {}
    return max(
        valid_results,
        key=lambda row: (
            row.get("ari") or 0.0,
            row.get("nmi") or 0.0,
            row.get("purity") or 0.0,
            -(row.get("noise_ratio") or 0.0),
        ),
    )


def _best_retrieval_metrics(retrieval_results):
    if not retrieval_results:
        return {}
    return max(
        retrieval_results,
        key=lambda row: (
            row.get("top1") or 0.0,
            row.get("top5") or 0.0,
        ),
    )


def _quality_cache_path(cache_dir, config):
    if cache_dir is None:
        return None
    config = normalize_face_quality_config(config)
    return Path(cache_dir) / (
        f"facesize_{config['min_face_size']}_ratio_{config['min_face_ratio']:.3f}"
        f"_blur_{config['min_laplacian_var']:.0f}_eval_{config['blur_eval_size']}"
        f"_pose_{config['max_pose_deviation']:.2f}.npz"
    )


def recommend_face_quality(quality_results):
    valid_rows = [row for row in quality_results if row.get("balanced_score") is not None]
    if not valid_rows:
        return None

    def _sort_key(row):
        return (
            row.get("balanced_score") or 0.0,
            -(row.get("failure_rate") or 1.0),
            -(row.get("min_face_size") or 0),
            -(row.get("min_face_ratio") or 0.0),
            -(row.get("min_laplacian_var") or 0.0),
            row.get("max_pose_deviation") or 0.0,
        )

    best = max(valid_rows, key=_sort_key)
    return {
        "enabled": True,
        "min_face_size": int(best["min_face_size"]),
        "min_face_ratio": float(best.get("min_face_ratio", 0.035)),
        "min_laplacian_var": float(best["min_laplacian_var"]),
        "max_pose_deviation": float(best["max_pose_deviation"]),
        "blur_eval_size": int(best.get("blur_eval_size", 96)),
    }


def evaluate_face_quality_grid(
    dataset_path,
    algorithms,
    metrics,
    eps_grid,
    min_samples_grid,
    top_k=5,
    quality_grid=None,
    cache_dir=None,
    refresh_cache=False,
    embedding_extractor=None,
):
    quality_grid = quality_grid or build_face_quality_grid()
    rows = []

    for config in quality_grid:
        dataset_payload = extract_dataset_embeddings(
            dataset_path,
            cache_path=_quality_cache_path(cache_dir, config),
            refresh_cache=refresh_cache,
            embedding_extractor=embedding_extractor,
            face_quality_config=config,
        )
        embeddings = dataset_payload["embeddings"]
        label_ids = dataset_payload["label_ids"]
        failures = dataset_payload["failures"]
        total_samples = len(embeddings) + len(failures)
        kept_samples = int(len(embeddings))
        failure_rate = (
            round(len(failures) / total_samples, 4) if total_samples else 1.0
        )

        cluster_row = {}
        retrieval_row = {}
        elapsed_seconds = 0.0
        if len(embeddings) >= 2 and len(set(label_ids.tolist())) >= 2:
            benchmark_payload = run_benchmark_suite(
                embeddings,
                label_ids,
                algorithms=algorithms,
                metrics=metrics,
                eps_grid=eps_grid,
                min_samples_grid=min_samples_grid,
                top_k=top_k,
            )
            cluster_row = _best_cluster_metrics(benchmark_payload["clustering_results"])
            retrieval_row = _best_retrieval_metrics(benchmark_payload["retrieval_results"])
            elapsed_seconds = benchmark_payload["elapsed_seconds"]

        balanced_components = [
            retrieval_row.get("top1"),
            retrieval_row.get("top5"),
            cluster_row.get("ari"),
            cluster_row.get("nmi"),
        ]
        if any(value is None for value in balanced_components):
            balanced_score = None
        else:
            balanced_score = round(sum(balanced_components) / 4.0, 4)

        rows.append(
            {
                "enabled": True,
                "min_face_size": int(config["min_face_size"]),
                "min_face_ratio": float(config["min_face_ratio"]),
                "min_laplacian_var": float(config["min_laplacian_var"]),
                "max_pose_deviation": float(config["max_pose_deviation"]),
                "blur_eval_size": int(config["blur_eval_size"]),
                "samples_kept": kept_samples,
                "failed_samples": int(len(failures)),
                "failure_rate": failure_rate,
                "top1": retrieval_row.get("top1"),
                "top5": retrieval_row.get("top5"),
                "purity": cluster_row.get("purity"),
                "nmi": cluster_row.get("nmi"),
                "ari": cluster_row.get("ari"),
                "balanced_score": balanced_score,
                "elapsed_seconds": elapsed_seconds,
            }
        )

    return {
        "results": rows,
        "recommended_face_quality": recommend_face_quality(rows),
    }


def run_benchmark_suite(
    embeddings,
    label_ids,
    algorithms,
    metrics,
    eps_grid,
    min_samples_grid,
    top_k=5,
):
    started_at = time.perf_counter()
    clustering_results = evaluate_embedding_clusters(
        embeddings,
        label_ids,
        algorithms=algorithms,
        metrics=metrics,
        eps_values=eps_grid,
        min_samples_values=min_samples_grid,
    )
    retrieval_results = evaluate_embedding_retrieval(
        embeddings,
        label_ids,
        metrics=metrics,
        top_k=top_k,
    )
    return {
        "clustering_results": clustering_results,
        "retrieval_results": retrieval_results,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field) for field in fieldnames}
            writer.writerow(normalized)


def export_benchmark_results(
    output_dir,
    clustering_results,
    retrieval_results,
    failures,
    quality_results=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clustering_path = output_dir / "clustering_results.csv"
    retrieval_path = output_dir / "retrieval_results.csv"
    failures_path = output_dir / "failed_samples.csv"
    quality_path = output_dir / "quality_filter_results.csv"

    write_csv(
        clustering_path,
        clustering_results,
        fieldnames=[
            "algorithm",
            "metric",
            "eps",
            "min_samples",
            "clusters_count",
            "noise_count",
            "noise_ratio",
            "purity",
            "nmi",
            "ari",
            "error",
        ],
    )
    write_csv(
        retrieval_path,
        retrieval_results,
        fieldnames=["metric", "top1", "top5", "queries"],
    )
    write_csv(
        failures_path,
        failures,
        fieldnames=["identity", "image_path", "reason"],
    )
    if quality_results is not None:
        write_csv(
            quality_path,
            quality_results,
            fieldnames=[
                "enabled",
                "min_face_size",
                "min_face_ratio",
                "min_laplacian_var",
                "max_pose_deviation",
                "blur_eval_size",
                "samples_kept",
                "failed_samples",
                "failure_rate",
                "top1",
                "top5",
                "purity",
                "nmi",
                "ari",
                "balanced_score",
                "elapsed_seconds",
            ],
        )
    return {
        "clustering": str(clustering_path),
        "retrieval": str(retrieval_path),
        "failures": str(failures_path),
        "quality": str(quality_path) if quality_results is not None else "",
    }


def summarize_benchmark(
    clustering_results,
    retrieval_results,
    failures,
    elapsed_seconds,
    quality_results=None,
    recommended_face_quality=None,
):
    valid_cluster_results = [row for row in clustering_results if not row.get("error")]
    best_clusters = valid_cluster_results[:3]
    failed_clusters = [row for row in clustering_results if row.get("error")]
    best_retrieval = sorted(
        retrieval_results,
        key=lambda row: ((row.get("top1") is None), -(row.get("top1") or 0.0), -(row.get("top5") or 0.0)),
    )[:3]

    lines = [
        "Benchmark summary",
        f"- elapsed_seconds: {elapsed_seconds}",
        f"- clustering_runs: {len(clustering_results)}",
        f"- retrieval_runs: {len(retrieval_results)}",
        f"- failed_samples: {len(failures)}",
    ]

    if recommended_face_quality:
        lines.append(
            "- recommended_face_quality: "
            + json.dumps(recommended_face_quality, ensure_ascii=False)
        )

    if quality_results:
        lines.append(f"- face_quality_configs: {len(quality_results)}")

    if best_clusters:
        lines.append("- best clustering:")
        for row in best_clusters:
            lines.append(
                "  "
                + f"{row['algorithm']} / {row['metric']} / eps={row['eps']} / min_samples={row['min_samples']} "
                + f"purity={row.get('purity')} nmi={row.get('nmi')} ari={row.get('ari')} noise={row.get('noise_ratio')}"
            )

    if best_retrieval:
        lines.append("- best retrieval:")
        for row in best_retrieval:
            lines.append(
                "  "
                + f"{row['metric']} top1={row.get('top1')} top5={row.get('top5')} queries={row.get('queries')}"
            )

    if failed_clusters:
        lines.append("- failed clustering configs:")
        for row in failed_clusters[:5]:
            lines.append(
                "  "
                + f"{row['algorithm']} / {row['metric']} / eps={row['eps']} / min_samples={row['min_samples']} -> {row['error']}"
            )

    return "\n".join(lines)
