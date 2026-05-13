import csv
import hashlib
import inspect
import json
import random
import shutil
import tarfile
import time
from itertools import product
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from core.clustering import evaluate_embedding_clusters, evaluate_embedding_retrieval


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LFW_DEEPFUNNELED_DIRNAME = "lfw-deepfunneled"
LFW_DEEPFUNNELED_FILENAME = "lfw-deepfunneled.tgz"
LFW_DEEPFUNNELED_MD5 = "68331da3eb755a505a502b5aacb3c201"
LFW_DEEPFUNNELED_URL = f"http://vis-www.cs.umass.edu/lfw/{LFW_DEEPFUNNELED_FILENAME}"
LFW_EXPECTED_IDENTITIES = 5749
LFW_EXPECTED_IMAGES = 13233
DEFAULT_FACE_QUALITY_GRID = {
    "min_face_size": [40, 56, 72],
    "min_face_ratio": [0.02, 0.035, 0.05],
    "min_laplacian_var": [0, 50, 80, 120],
    "max_pose_deviation": [0.25, 0.35, 0.45],
    "blur_eval_size": [96],
}
CONTROLLED_FACE_QUALITY_GRID = [
    {
        "enabled": True,
        "min_face_size": 56,
        "min_face_ratio": 0.035,
        "min_laplacian_var": 80.0,
        "max_pose_deviation": 0.35,
        "blur_eval_size": 96,
    },
    {
        "enabled": True,
        "min_face_size": 40,
        "min_face_ratio": 0.02,
        "min_laplacian_var": 50.0,
        "max_pose_deviation": 0.45,
        "blur_eval_size": 96,
    },
    {
        "enabled": True,
        "min_face_size": 72,
        "min_face_ratio": 0.05,
        "min_laplacian_var": 120.0,
        "max_pose_deviation": 0.25,
        "blur_eval_size": 96,
    },
]
THESIS_SAMPLE_CONFIG = {
    "sample_identities": 300,
    "min_images_per_identity": 2,
    "max_images_per_identity": 5,
    "seed": 20260511,
}
THESIS_DEFAULT_CLUSTER_CONFIG = {
    "algorithm": "dbscan",
    "metric": "cosine",
    "eps": 0.4,
    "min_samples": 2,
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


def count_identity_dataset(dataset_path):
    identities = discover_identity_dataset(dataset_path)
    return {
        "identities": len(identities),
        "images": sum(len(identity["image_paths"]) for identity in identities),
    }


def normalize_sample_config(sample_config=None):
    if not sample_config:
        return None
    return {
        "sample_identities": max(int(sample_config.get("sample_identities", 0)), 0),
        "min_images_per_identity": max(
            int(sample_config.get("min_images_per_identity", 1)),
            1,
        ),
        "max_images_per_identity": max(
            int(sample_config.get("max_images_per_identity", 0)),
            0,
        ),
        "seed": int(sample_config.get("seed", THESIS_SAMPLE_CONFIG["seed"])),
    }


def sample_identity_dataset(
    dataset_path,
    sample_identities=THESIS_SAMPLE_CONFIG["sample_identities"],
    min_images_per_identity=THESIS_SAMPLE_CONFIG["min_images_per_identity"],
    max_images_per_identity=THESIS_SAMPLE_CONFIG["max_images_per_identity"],
    seed=THESIS_SAMPLE_CONFIG["seed"],
):
    sample_config = normalize_sample_config(
        {
            "sample_identities": sample_identities,
            "min_images_per_identity": min_images_per_identity,
            "max_images_per_identity": max_images_per_identity,
            "seed": seed,
        }
    )
    identities = discover_identity_dataset(dataset_path)
    eligible = [
        identity
        for identity in identities
        if len(identity["image_paths"]) >= sample_config["min_images_per_identity"]
    ]
    rng = random.Random(sample_config["seed"])
    target_count = sample_config["sample_identities"] or len(eligible)
    selected = rng.sample(eligible, min(target_count, len(eligible)))
    selected.sort(key=lambda item: item["identity"])

    sampled_identities = []
    manifest_rows = []
    max_images = sample_config["max_images_per_identity"]
    for label_id, identity in enumerate(selected):
        image_paths = list(identity["image_paths"])
        if max_images and len(image_paths) > max_images:
            image_paths = rng.sample(image_paths, max_images)
        image_paths = sorted(image_paths)
        sampled_identities.append(
            {
                **identity,
                "image_paths": image_paths,
            }
        )
        for image_index, image_path in enumerate(image_paths):
            manifest_rows.append(
                {
                    "label_id": int(label_id),
                    "identity": identity["identity"],
                    "image_index": int(image_index),
                    "image_path": image_path,
                }
            )

    return {
        "identities": sampled_identities,
        "manifest": manifest_rows,
        "sample_config": sample_config,
        "available_identities": int(len(eligible)),
        "sampled_identities": int(len(sampled_identities)),
        "sampled_images": int(sum(len(item["image_paths"]) for item in sampled_identities)),
    }


def _file_md5(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(tar, destination):
    destination = Path(destination).resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination)
        except ValueError:
            raise RuntimeError(f"压缩包成员路径不安全：{member.name}")
    tar.extractall(destination)


def ensure_lfw_deepfunneled_dataset(dataset_path, download_if_missing=True):
    dataset_path = Path(dataset_path).expanduser().resolve()
    status = {
        "dataset_path": str(dataset_path),
        "expected_identities": LFW_EXPECTED_IDENTITIES,
        "expected_images": LFW_EXPECTED_IMAGES,
        "downloaded": False,
        "source": "local",
        "complete": False,
    }

    if dataset_path.exists():
        counts = count_identity_dataset(dataset_path)
        status.update(counts)
        if counts["identities"] == LFW_EXPECTED_IDENTITIES and counts["images"] == LFW_EXPECTED_IMAGES:
            status["complete"] = True
            return status

    if not download_if_missing:
        return status

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_path.parent / LFW_DEEPFUNNELED_FILENAME
    if not archive_path.exists() or _file_md5(archive_path) != LFW_DEEPFUNNELED_MD5:
        urlretrieve(LFW_DEEPFUNNELED_URL, archive_path)
        actual_md5 = _file_md5(archive_path)
        if actual_md5 != LFW_DEEPFUNNELED_MD5:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"LFW 压缩包校验失败：期望 {LFW_DEEPFUNNELED_MD5}，实际 {actual_md5}"
            )

    extract_parent = dataset_path.parent
    extracted_path = extract_parent / LFW_DEEPFUNNELED_DIRNAME
    if extracted_path.exists() and extracted_path.resolve() != dataset_path:
        shutil.rmtree(extracted_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract_tar(tar, extract_parent)
    if extracted_path.resolve() != dataset_path:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        extracted_path.replace(dataset_path)

    counts = count_identity_dataset(dataset_path)
    status.update(counts)
    status["downloaded"] = True
    status["source"] = LFW_DEEPFUNNELED_URL
    status["complete"] = counts["identities"] == LFW_EXPECTED_IDENTITIES and counts["images"] == LFW_EXPECTED_IMAGES
    if not status["complete"]:
        raise RuntimeError(
            "下载的 LFW deepfunneled 数据集不完整："
            f"{counts['identities']} 个身份，{counts['images']} 张图片。"
        )
    return status


def get_runtime_device_report():
    report = {
        "torch_available": False,
        "torch_cuda_available": False,
        "torch_version": "",
        "torch_cuda_version": "",
        "cuda_device_count": 0,
        "cuda_device_name": "",
        "onnxruntime_available": False,
        "onnxruntime_version": "",
        "onnxruntime_providers": [],
        "faiss_available": False,
        "faiss_version": "",
        "faiss_gpu_available": False,
    }
    try:
        import torch

        report["torch_available"] = True
        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = str(torch.version.cuda or "")
        report["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count():
            report["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        report["torch_error"] = str(exc)

    try:
        import onnxruntime as ort

        report["onnxruntime_available"] = True
        report["onnxruntime_version"] = str(ort.__version__)
        report["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception as exc:
        report["onnxruntime_error"] = str(exc)

    try:
        import faiss

        report["faiss_available"] = True
        report["faiss_version"] = str(getattr(faiss, "__version__", "unknown"))
        report["faiss_gpu_available"] = bool(hasattr(faiss, "StandardGpuResources"))
    except Exception as exc:
        report["faiss_error"] = str(exc)

    return report


def build_controlled_face_quality_grid():
    return [normalize_face_quality_config(config) for config in CONTROLLED_FACE_QUALITY_GRID]


def _with_experiment_metadata(config, experiment_name, changed_parameter, is_baseline=False):
    normalized = normalize_face_quality_config(config)
    normalized.update(
        {
            "experiment_name": experiment_name,
            "changed_parameter": changed_parameter,
            "is_baseline": bool(is_baseline),
        }
    )
    return normalized


def build_thesis_face_quality_grid():
    baseline = {
        "enabled": True,
        "min_face_size": 56,
        "min_face_ratio": 0.035,
        "min_laplacian_var": 80.0,
        "max_pose_deviation": 0.35,
        "blur_eval_size": 96,
    }
    configs = [
        (baseline, "default_balanced", "baseline", True),
        (
            {
                **baseline,
                "min_face_size": 40,
                "min_face_ratio": 0.02,
                "min_laplacian_var": 50.0,
                "max_pose_deviation": 0.45,
            },
            "relaxed_all",
            "all_relaxed",
            False,
        ),
        (
            {
                **baseline,
                "min_face_size": 72,
                "min_face_ratio": 0.05,
                "min_laplacian_var": 120.0,
                "max_pose_deviation": 0.25,
            },
            "strict_all",
            "all_strict",
            False,
        ),
        ({**baseline, "min_face_size": 40}, "face_size_low", "min_face_size", False),
        ({**baseline, "min_face_size": 72}, "face_size_high", "min_face_size", False),
        ({**baseline, "min_face_ratio": 0.02}, "face_ratio_low", "min_face_ratio", False),
        ({**baseline, "min_face_ratio": 0.05}, "face_ratio_high", "min_face_ratio", False),
        ({**baseline, "min_laplacian_var": 50.0}, "blur_threshold_low", "min_laplacian_var", False),
        ({**baseline, "min_laplacian_var": 120.0}, "blur_threshold_high", "min_laplacian_var", False),
        ({**baseline, "max_pose_deviation": 0.45}, "pose_threshold_loose", "max_pose_deviation", False),
        ({**baseline, "max_pose_deviation": 0.25}, "pose_threshold_strict", "max_pose_deviation", False),
    ]
    return [
        _with_experiment_metadata(config, experiment_name, changed_parameter, is_baseline)
        for config, experiment_name, changed_parameter, is_baseline in configs
    ]


def build_thesis_cluster_grid():
    baseline = dict(THESIS_DEFAULT_CLUSTER_CONFIG)
    configs = [
        (baseline, "default_dbscan_cosine_eps04_min2", "baseline", True),
        ({**baseline, "eps": 0.3}, "eps_03", "eps", False),
        ({**baseline, "eps": 0.5}, "eps_05", "eps", False),
        ({**baseline, "min_samples": 3}, "min_samples_3", "min_samples", False),
        ({**baseline, "metric": "euclidean"}, "metric_euclidean", "metric", False),
        ({**baseline, "algorithm": "hdbscan"}, "algorithm_hdbscan", "algorithm", False),
        ({**baseline, "algorithm": "optics"}, "algorithm_optics", "algorithm", False),
    ]
    return [
        {
            **config,
            "experiment_name": experiment_name,
            "changed_parameter": changed_parameter,
            "is_baseline": bool(is_baseline),
        }
        for config, experiment_name, changed_parameter, is_baseline in configs
    ]


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
        return None, result["failure_reason"] or "embedding_failed", result.get("metrics")
    return np.asarray(result["embedding"], dtype=np.float32), None, result.get("metrics")


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
        "quality_metrics": payload["quality_metrics"].tolist()
        if "quality_metrics" in payload.files
        else [],
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
    quality_metrics=None,
    sample_config=None,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        label_ids=np.asarray(label_ids, dtype=np.int32),
        label_names=np.asarray(label_names, dtype=object),
        image_paths=np.asarray(image_paths, dtype=object),
        quality_metrics=np.asarray(quality_metrics or [], dtype=object),
    )
    meta_path = cache_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_path": str(Path(dataset_path).resolve()),
                "samples": int(len(image_paths)),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "face_quality_config": normalize_face_quality_config(face_quality_config),
                "sample_config": normalize_sample_config(sample_config),
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
    identities=None,
    sample_config=None,
):
    cache_path = Path(cache_path) if cache_path else None
    failures_path = Path(failures_path) if failures_path else None
    normalized_face_quality = normalize_face_quality_config(face_quality_config)
    normalized_sample_config = normalize_sample_config(sample_config)
    if cache_path and cache_path.exists() and not refresh_cache:
        cached = load_embedding_cache(cache_path)
        cached_meta = cached.get("meta", {})
        if cached_meta.get("dataset_path") == str(Path(dataset_path).resolve()) and cached_meta.get(
            "face_quality_config"
        ) == normalized_face_quality and cached_meta.get("sample_config") == normalized_sample_config:
            cached["failures"] = []
            cached["from_cache"] = True
            return cached

    if embedding_extractor is None:
        embedding_extractor = lambda image_path: _default_embedding_extractor(
            image_path,
            face_quality_config=normalized_face_quality,
        )
    identities = identities if identities is not None else discover_identity_dataset(dataset_path)
    embeddings = []
    label_ids = []
    label_names = []
    image_paths = []
    quality_metrics = []
    failures = []

    for label_id, identity in enumerate(identities):
        for image_path in identity["image_paths"]:
            try:
                if len(inspect.signature(embedding_extractor).parameters) >= 2:
                    extractor_result = embedding_extractor(
                        image_path,
                        normalized_face_quality,
                    )
                else:
                    extractor_result = embedding_extractor(image_path)
                if isinstance(extractor_result, tuple) and len(extractor_result) >= 3:
                    embedding, failure_reason, metrics = extractor_result[:3]
                else:
                    embedding, failure_reason = extractor_result
                    metrics = None
            except Exception as exc:
                embedding = None
                failure_reason = str(exc)
                metrics = None

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
            quality_metrics.append(metrics or {})

    payload = {
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "label_ids": np.asarray(label_ids, dtype=np.int32),
        "label_names": label_names,
        "image_paths": image_paths,
        "quality_metrics": quality_metrics,
        "failures": failures,
        "from_cache": False,
        "face_quality_config": normalized_face_quality,
        "sample_config": normalized_sample_config,
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
            quality_metrics=payload["quality_metrics"],
            sample_config=normalized_sample_config,
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


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_ratio(value, target, lower_is_better=False):
    value = _safe_float(value)
    target = _safe_float(target)
    if value is None or target is None or target <= 0:
        return None
    if lower_is_better:
        return max(0.0, min(1.0, 1.0 - (value / target)))
    return max(0.0, min(1.0, value / target))


def aggregate_quality_metrics(quality_metrics):
    rows = [row for row in (quality_metrics or []) if isinstance(row, dict)]
    if not rows:
        return {
            "avg_min_face_size": None,
            "avg_face_ratio": None,
            "avg_laplacian_var": None,
            "avg_pose_deviation": None,
            "accepted_quality_score": None,
        }

    def _average(key):
        values = [_safe_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    avg_min_face_size = _average("min_face_size")
    avg_face_ratio = _average("face_ratio")
    avg_laplacian_var = _average("laplacian_var")
    avg_pose_deviation = _average("pose_deviation")
    components = [
        _bounded_ratio(avg_min_face_size, 72.0),
        _bounded_ratio(avg_face_ratio, 0.05),
        _bounded_ratio(avg_laplacian_var, 120.0),
        _bounded_ratio(avg_pose_deviation, 0.45, lower_is_better=True),
    ]
    if any(value is None for value in components):
        accepted_quality_score = None
    else:
        accepted_quality_score = round(sum(components) / len(components), 4)
    return {
        "avg_min_face_size": avg_min_face_size,
        "avg_face_ratio": avg_face_ratio,
        "avg_laplacian_var": avg_laplacian_var,
        "avg_pose_deviation": avg_pose_deviation,
        "accepted_quality_score": accepted_quality_score,
    }


def compute_balanced_score_v2(row):
    top1 = _safe_float(row.get("top1"))
    top5 = _safe_float(row.get("top5"))
    nmi = _safe_float(row.get("nmi"))
    ari = _safe_float(row.get("ari"))
    noise_ratio = _safe_float(row.get("noise_ratio"))
    failure_rate = _safe_float(row.get("failure_rate"))
    accepted_quality_score = _safe_float(row.get("accepted_quality_score"))
    required = [top1, top5, nmi, ari, noise_ratio, failure_rate, accepted_quality_score]
    if any(value is None for value in required):
        return {
            "retrieval_score": None,
            "cluster_score": None,
            "retention_score": None,
            "quality_balance_score": None,
            "balanced_score_v2": None,
        }

    retrieval_score = (0.7 * top1) + (0.3 * top5)
    cluster_score = (0.5 * nmi) + (0.3 * ari) + (0.2 * (1.0 - noise_ratio))
    retention_score = 1.0 - failure_rate
    quality_balance_score = (0.5 * retention_score) + (0.5 * accepted_quality_score)
    balanced_score_v2 = (
        (0.45 * retrieval_score)
        + (0.30 * cluster_score)
        + (0.25 * quality_balance_score)
    )
    return {
        "retrieval_score": round(retrieval_score, 4),
        "cluster_score": round(cluster_score, 4),
        "retention_score": round(retention_score, 4),
        "quality_balance_score": round(quality_balance_score, 4),
        "balanced_score_v2": round(balanced_score_v2, 4),
    }


def recommend_face_quality(quality_results):
    score_field = (
        "balanced_score_v2"
        if any(row.get("balanced_score_v2") is not None for row in quality_results)
        else "balanced_score"
    )
    valid_rows = [row for row in quality_results if row.get(score_field) is not None]
    if not valid_rows:
        return None

    def _sort_key(row):
        if score_field == "balanced_score_v2":
            return (
                row.get("balanced_score_v2") or 0.0,
                row.get("top1") or 0.0,
                row.get("nmi") or 0.0,
                -(row.get("failure_rate") or 1.0),
            )
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
    retrieval_backend="numpy",
    identities=None,
    sample_config=None,
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
            identities=identities,
            sample_config=sample_config,
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
                retrieval_backend=retrieval_backend,
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

        quality_metrics = aggregate_quality_metrics(dataset_payload.get("quality_metrics"))
        row = {
            "experiment_name": config.get("experiment_name", ""),
            "changed_parameter": config.get("changed_parameter", ""),
            "is_baseline": bool(config.get("is_baseline", False)),
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
            "noise_ratio": cluster_row.get("noise_ratio"),
            "balanced_score": balanced_score,
            "elapsed_seconds": elapsed_seconds,
        }
        row.update(quality_metrics)
        row.update(compute_balanced_score_v2(row))
        rows.append(row)

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
    retrieval_backend="numpy",
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
        backend=retrieval_backend,
    )
    return {
        "clustering_results": clustering_results,
        "retrieval_results": retrieval_results,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "retrieval_backend": retrieval_backend,
    }


def evaluate_cluster_ablation(embeddings, label_ids, cluster_grid=None):
    cluster_grid = cluster_grid or build_thesis_cluster_grid()
    rows = []
    for config in cluster_grid:
        started_at = time.perf_counter()
        result = evaluate_embedding_clusters(
            embeddings,
            label_ids,
            algorithms=[config["algorithm"]],
            metrics=[config["metric"]],
            eps_values=[config["eps"]],
            min_samples_values=[config["min_samples"]],
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        row = dict(result[0]) if result else {}
        row.update(
            {
                "experiment_name": config.get("experiment_name", ""),
                "changed_parameter": config.get("changed_parameter", ""),
                "is_baseline": bool(config.get("is_baseline", False)),
                "algorithm": config["algorithm"],
                "metric": config["metric"],
                "eps": float(config["eps"]),
                "min_samples": int(config["min_samples"]),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        if row.get("nmi") is not None and row.get("ari") is not None and row.get("noise_ratio") is not None:
            row["cluster_score"] = round(
                (0.5 * float(row["nmi"]))
                + (0.3 * float(row["ari"]))
                + (0.2 * (1.0 - float(row["noise_ratio"]))),
                4,
            )
        else:
            row["cluster_score"] = None
        rows.append(row)
    rows.sort(
        key=lambda item: (
            bool(item.get("error")),
            item.get("cluster_score") is None,
            -(item.get("cluster_score") or 0.0),
            -(item.get("purity") or 0.0),
        )
    )
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field) for field in fieldnames}
            writer.writerow(normalized)


def export_sample_manifest(output_dir, sample_payload):
    if not sample_payload:
        return {"csv": "", "json": ""}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sample_manifest.csv"
    json_path = output_dir / "sample_manifest.json"
    write_csv(
        csv_path,
        sample_payload.get("manifest") or [],
        fieldnames=["label_id", "identity", "image_index", "image_path"],
    )
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "sample_config": sample_payload.get("sample_config"),
                "available_identities": sample_payload.get("available_identities"),
                "sampled_identities": sample_payload.get("sampled_identities"),
                "sampled_images": sample_payload.get("sampled_images"),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return {"csv": str(csv_path), "json": str(json_path)}


def export_benchmark_results(
    output_dir,
    clustering_results,
    retrieval_results,
    failures,
    quality_results=None,
    quality_ablation_results=None,
    cluster_ablation_results=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clustering_path = output_dir / "clustering_results.csv"
    retrieval_path = output_dir / "retrieval_results.csv"
    failures_path = output_dir / "failed_samples.csv"
    quality_path = output_dir / "quality_filter_results.csv"
    quality_ablation_path = output_dir / "quality_ablation_results.csv"
    cluster_ablation_path = output_dir / "cluster_ablation_results.csv"

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
        fieldnames=["metric", "backend", "top1", "top5", "queries", "fallback_reason"],
    )
    write_csv(
        failures_path,
        failures,
        fieldnames=["identity", "image_path", "reason"],
    )
    if quality_results is not None:
        quality_fieldnames = [
            "experiment_name",
            "changed_parameter",
            "is_baseline",
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
            "noise_ratio",
            "avg_min_face_size",
            "avg_face_ratio",
            "avg_laplacian_var",
            "avg_pose_deviation",
            "accepted_quality_score",
            "retrieval_score",
            "cluster_score",
            "retention_score",
            "quality_balance_score",
            "balanced_score",
            "balanced_score_v2",
            "elapsed_seconds",
        ]
        write_csv(quality_path, quality_results, fieldnames=quality_fieldnames)
    if quality_ablation_results is not None:
        write_csv(
            quality_ablation_path,
            quality_ablation_results,
            fieldnames=[
                "experiment_name",
                "changed_parameter",
                "is_baseline",
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
                "noise_ratio",
                "avg_min_face_size",
                "avg_face_ratio",
                "avg_laplacian_var",
                "avg_pose_deviation",
                "accepted_quality_score",
                "retrieval_score",
                "cluster_score",
                "retention_score",
                "quality_balance_score",
                "balanced_score",
                "balanced_score_v2",
                "elapsed_seconds",
            ],
        )
    if cluster_ablation_results is not None:
        write_csv(
            cluster_ablation_path,
            cluster_ablation_results,
            fieldnames=[
                "experiment_name",
                "changed_parameter",
                "is_baseline",
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
                "cluster_score",
                "elapsed_seconds",
                "error",
            ],
        )
    return {
        "clustering": str(clustering_path),
        "retrieval": str(retrieval_path),
        "failures": str(failures_path),
        "quality": str(quality_path) if quality_results is not None else "",
        "quality_ablation": str(quality_ablation_path)
        if quality_ablation_results is not None
        else "",
        "cluster_ablation": str(cluster_ablation_path)
        if cluster_ablation_results is not None
        else "",
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
        "基准测试摘要",
        f"- 耗时秒数：{elapsed_seconds}",
        f"- 聚类运行数：{len(clustering_results)}",
        f"- 检索运行数：{len(retrieval_results)}",
        f"- 失败样本数：{len(failures)}",
    ]

    if recommended_face_quality:
        lines.append(
            "- 推荐人脸过滤参数："
            + json.dumps(recommended_face_quality, ensure_ascii=False)
        )

    if quality_results:
        lines.append(f"- 人脸过滤配置数：{len(quality_results)}")

    if best_clusters:
        lines.append("- 最优聚类配置：")
        for row in best_clusters:
            lines.append(
                "  "
                + f"{row['algorithm']} / {row['metric']} / eps={row['eps']} / min_samples={row['min_samples']} "
                + f"purity={row.get('purity')} nmi={row.get('nmi')} ari={row.get('ari')} noise={row.get('noise_ratio')}"
            )

    if best_retrieval:
        lines.append("- 最优检索配置：")
        for row in best_retrieval:
            lines.append(
                "  "
                + f"{row['metric']} backend={row.get('backend')} "
                + f"top1={row.get('top1')} top5={row.get('top5')} queries={row.get('queries')}"
            )

    if failed_clusters:
        lines.append("- 失败聚类配置：")
        for row in failed_clusters[:5]:
            lines.append(
                "  "
                + f"{row['algorithm']} / {row['metric']} / eps={row['eps']} / min_samples={row['min_samples']} -> {row['error']}"
            )

    return "\n".join(lines)
