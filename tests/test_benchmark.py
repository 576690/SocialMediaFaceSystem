import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import core.clustering as clustering
import scripts.run_benchmark as run_benchmark_script
from core.benchmark import build_controlled_face_quality_grid
from core.benchmark import build_thesis_cluster_grid
from core.benchmark import build_thesis_face_quality_grid
from core.benchmark import compute_balanced_score_v2
from core.benchmark import discover_identity_dataset
from core.benchmark import evaluate_face_quality_grid
from core.benchmark import export_benchmark_results
from core.benchmark import extract_dataset_embeddings
from core.benchmark import get_runtime_device_report
from core.benchmark import recommend_face_quality
from core.benchmark import run_benchmark_suite
from core.benchmark import sample_identity_dataset


class BenchmarkTests(unittest.TestCase):
    def test_discover_identity_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "alice").mkdir()
            (root / "bob").mkdir()
            (root / "alice" / "a1.jpg").write_bytes(b"")
            (root / "bob" / "b1.png").write_bytes(b"")
            (root / "bob" / "notes.txt").write_text("skip", encoding="utf-8")

            identities = discover_identity_dataset(root)

            self.assertEqual([item["identity"] for item in identities], ["alice", "bob"])
            self.assertEqual(len(identities[0]["image_paths"]), 1)
            self.assertEqual(len(identities[1]["image_paths"]), 1)

    def test_sample_identity_dataset_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(6):
                identity_dir = root / f"person_{index}"
                identity_dir.mkdir()
                for image_index in range(4):
                    (identity_dir / f"{image_index}.jpg").write_bytes(b"")

            first = sample_identity_dataset(
                root,
                sample_identities=3,
                min_images_per_identity=2,
                max_images_per_identity=2,
                seed=123,
            )
            second = sample_identity_dataset(
                root,
                sample_identities=3,
                min_images_per_identity=2,
                max_images_per_identity=2,
                seed=123,
            )

            self.assertEqual(first["manifest"], second["manifest"])
            self.assertEqual(first["sampled_identities"], 3)
            self.assertEqual(first["sampled_images"], 6)
            self.assertTrue(
                all(len(identity["image_paths"]) == 2 for identity in first["identities"])
            )

    def test_extract_embeddings_uses_cache_and_logs_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "output"
            (dataset / "alice").mkdir(parents=True)
            (dataset / "bob").mkdir(parents=True)
            (dataset / "alice" / "a1.jpg").write_bytes(b"")
            (dataset / "bob" / "b1.jpg").write_bytes(b"")
            (dataset / "bob" / "b2.jpg").write_bytes(b"")

            def fake_extractor(image_path):
                image_path = Path(image_path)
                if image_path.name == "b2.jpg":
                    return None, "no_face_detected"
                if image_path.parent.name == "alice":
                    return np.array([1.0, 0.0], dtype=np.float32), None
                return np.array([0.0, 1.0], dtype=np.float32), None

            cache_path = output / "embedding_cache.npz"
            failures_path = output / "failed_samples.csv"
            payload = extract_dataset_embeddings(
                dataset,
                cache_path=cache_path,
                failures_path=failures_path,
                embedding_extractor=fake_extractor,
            )

            self.assertFalse(payload["from_cache"])
            self.assertEqual(payload["embeddings"].shape, (2, 2))
            self.assertEqual(len(payload["failures"]), 1)
            self.assertTrue(cache_path.exists())
            self.assertTrue(failures_path.exists())

            cached_payload = extract_dataset_embeddings(
                dataset,
                cache_path=cache_path,
                failures_path=failures_path,
                embedding_extractor=lambda _: (_ for _ in ()).throw(RuntimeError("should not run")),
            )
            self.assertTrue(cached_payload["from_cache"])
            self.assertEqual(cached_payload["embeddings"].shape, (2, 2))

    def test_extract_embeddings_cache_isolated_by_sample_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "output"
            (dataset / "alice").mkdir(parents=True)
            (dataset / "alice" / "a1.jpg").write_bytes(b"")

            calls = {"count": 0}

            def fake_extractor(image_path):
                calls["count"] += 1
                return np.array([1.0, 0.0], dtype=np.float32), None

            cache_path = output / "embedding_cache.npz"
            extract_dataset_embeddings(
                dataset,
                cache_path=cache_path,
                embedding_extractor=fake_extractor,
                sample_config={
                    "sample_identities": 1,
                    "min_images_per_identity": 1,
                    "max_images_per_identity": 1,
                    "seed": 1,
                },
            )
            extract_dataset_embeddings(
                dataset,
                cache_path=cache_path,
                embedding_extractor=fake_extractor,
                sample_config={
                    "sample_identities": 1,
                    "min_images_per_identity": 1,
                    "max_images_per_identity": 1,
                    "seed": 2,
                },
            )

            self.assertEqual(calls["count"], 2)

    def test_run_benchmark_suite_and_export_results(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        label_ids = np.array([0, 0, 1, 1], dtype=np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            benchmark = run_benchmark_suite(
                embeddings,
                label_ids,
                algorithms=["dbscan"],
                metrics=["cosine"],
                eps_grid=[0.1],
                min_samples_grid=[2],
                top_k=2,
                retrieval_backend="numpy",
            )
            exported = export_benchmark_results(
                output_dir,
                benchmark["clustering_results"],
                benchmark["retrieval_results"],
                failures=[],
                quality_results=[
                    {
                        "enabled": True,
                        "min_face_size": 56,
                        "min_face_ratio": 0.035,
                        "min_laplacian_var": 80.0,
                        "max_pose_deviation": 0.35,
                        "blur_eval_size": 96,
                        "samples_kept": 4,
                        "failed_samples": 0,
                        "failure_rate": 0.0,
                        "top1": 1.0,
                        "top5": 1.0,
                        "purity": 1.0,
                        "nmi": 1.0,
                        "ari": 1.0,
                        "balanced_score": 1.0,
                        "elapsed_seconds": 0.1,
                    }
                ],
            )

            self.assertTrue(Path(exported["clustering"]).exists())
            self.assertTrue(Path(exported["retrieval"]).exists())
            self.assertTrue(Path(exported["failures"]).exists())
            self.assertTrue(Path(exported["quality"]).exists())

    def test_retrieval_backend_falls_back_to_numpy(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        label_ids = np.array([0, 0, 1, 1], dtype=np.int32)

        with patch.object(
            clustering,
            "_evaluate_embedding_retrieval_torch_cuda",
            side_effect=RuntimeError("cuda unavailable"),
        ):
            results = clustering.evaluate_embedding_retrieval(
                embeddings,
                label_ids,
                metrics=["cosine"],
                top_k=2,
                backend="auto",
            )

        self.assertEqual(results[0]["backend"], "numpy")
        self.assertIn("cuda unavailable", results[0]["fallback_reason"])
        self.assertEqual(results[0]["top1"], 1.0)

    @unittest.skipUnless(
        get_runtime_device_report()["torch_cuda_available"],
        "CUDA is unavailable in this runtime.",
    )
    def test_torch_cuda_retrieval_matches_numpy(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
        label_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)

        numpy_results = clustering.evaluate_embedding_retrieval(
            embeddings,
            label_ids,
            metrics=["cosine", "euclidean"],
            top_k=2,
            backend="numpy",
        )
        cuda_results = clustering.evaluate_embedding_retrieval(
            embeddings,
            label_ids,
            metrics=["cosine", "euclidean"],
            top_k=2,
            backend="torch-cuda",
        )

        for numpy_row, cuda_row in zip(numpy_results, cuda_results):
            self.assertEqual(cuda_row["backend"], "torch-cuda")
            self.assertEqual(cuda_row["metric"], numpy_row["metric"])
            self.assertEqual(cuda_row["top1"], numpy_row["top1"])
            self.assertEqual(cuda_row["top5"], numpy_row["top5"])
            self.assertEqual(cuda_row["queries"], numpy_row["queries"])

    def test_controlled_quality_grid_has_three_configs(self):
        grid = build_controlled_face_quality_grid()

        self.assertEqual(len(grid), 3)
        self.assertEqual(grid[0]["min_face_size"], 56)
        self.assertEqual(grid[1]["min_face_ratio"], 0.02)
        self.assertEqual(grid[2]["max_pose_deviation"], 0.25)

    def test_thesis_quality_grid_has_baseline_and_single_factor_controls(self):
        grid = build_thesis_face_quality_grid()
        names = {item["experiment_name"] for item in grid}
        changed = {item["changed_parameter"] for item in grid}

        self.assertIn("default_balanced", names)
        self.assertIn("relaxed_all", names)
        self.assertIn("strict_all", names)
        self.assertTrue(any(item["is_baseline"] for item in grid))
        self.assertTrue(
            {"min_face_size", "min_face_ratio", "min_laplacian_var", "max_pose_deviation"}.issubset(changed)
        )

    def test_thesis_cluster_grid_has_default_and_ablation_controls(self):
        grid = build_thesis_cluster_grid()
        baseline = [item for item in grid if item["is_baseline"]][0]
        changed = {item["changed_parameter"] for item in grid}

        self.assertEqual(baseline["algorithm"], "dbscan")
        self.assertEqual(baseline["metric"], "cosine")
        self.assertEqual(baseline["eps"], 0.4)
        self.assertEqual(baseline["min_samples"], 2)
        self.assertTrue({"eps", "min_samples", "metric", "algorithm"}.issubset(changed))

    def test_balanced_score_v2_penalizes_extreme_quality_tradeoffs(self):
        balanced = compute_balanced_score_v2(
            {
                "top1": 0.98,
                "top5": 0.99,
                "nmi": 0.8,
                "ari": 0.2,
                "noise_ratio": 0.3,
                "failure_rate": 0.08,
                "accepted_quality_score": 0.8,
            }
        )
        too_strict = compute_balanced_score_v2(
            {
                "top1": 0.99,
                "top5": 0.995,
                "nmi": 0.82,
                "ari": 0.22,
                "noise_ratio": 0.3,
                "failure_rate": 0.7,
                "accepted_quality_score": 0.95,
            }
        )
        too_relaxed = compute_balanced_score_v2(
            {
                "top1": 0.99,
                "top5": 0.995,
                "nmi": 0.82,
                "ari": 0.22,
                "noise_ratio": 0.3,
                "failure_rate": 0.02,
                "accepted_quality_score": 0.2,
            }
        )

        self.assertGreater(balanced["balanced_score_v2"], too_strict["balanced_score_v2"])
        self.assertGreater(balanced["balanced_score_v2"], too_relaxed["balanced_score_v2"])

    def test_cli_writes_benchmark_run_device_metadata(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        label_ids = np.array([0, 0, 1, 1], dtype=np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "lfw-deepfunneled"
            output = root / "benchmark"
            dataset.mkdir()
            argv = [
                "run_benchmark.py",
                "--dataset-path",
                str(dataset),
                "--output-dir",
                str(output),
                "--skip-quality-grid",
                "--retrieval-backend",
                "numpy",
            ]

            with patch.object(sys, "argv", argv), patch.object(
                run_benchmark_script,
                "ensure_lfw_deepfunneled_dataset",
                return_value={
                    "dataset_path": str(dataset),
                    "identities": 5749,
                    "images": 13233,
                    "complete": True,
                    "source": "local",
                    "downloaded": False,
                },
            ), patch.object(
                run_benchmark_script,
                "get_runtime_device_report",
                return_value={
                    "torch_available": True,
                    "torch_cuda_available": True,
                    "cuda_device_name": "Unit Test GPU",
                    "onnxruntime_providers": ["CUDAExecutionProvider"],
                    "faiss_gpu_available": False,
                },
            ), patch.object(
                run_benchmark_script,
                "extract_dataset_embeddings",
                return_value={
                    "embeddings": embeddings,
                    "label_ids": label_ids,
                    "failures": [],
                    "from_cache": False,
                },
            ):
                run_benchmark_script.main()

            metadata_path = output / "benchmark_run.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["retrieval_backend"], "numpy")
            self.assertEqual(metadata["device_report"]["cuda_device_name"], "Unit Test GPU")
            self.assertEqual(metadata["dataset_status"]["images"], 13233)

    def test_cli_thesis_small_writes_ablation_outputs(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        label_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        quality_metrics = [
            {
                "min_face_size": 64,
                "face_ratio": 0.04,
                "laplacian_var": 100,
                "pose_deviation": 0.2,
            }
            for _ in range(4)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "lfw-deepfunneled"
            output = root / "benchmark"
            for identity in ("alice", "bob"):
                (dataset / identity).mkdir(parents=True)
                (dataset / identity / "1.jpg").write_bytes(b"")
                (dataset / identity / "2.jpg").write_bytes(b"")
            argv = [
                "run_benchmark.py",
                "--dataset-path",
                str(dataset),
                "--output-dir",
                str(output),
                "--experiment-preset",
                "thesis-small",
                "--sample-identities",
                "2",
                "--retrieval-backend",
                "numpy",
            ]

            with patch.object(sys, "argv", argv), patch.object(
                run_benchmark_script,
                "ensure_lfw_deepfunneled_dataset",
                return_value={
                    "dataset_path": str(dataset),
                    "identities": 2,
                    "images": 4,
                    "complete": True,
                    "source": "local",
                    "downloaded": False,
                },
            ), patch.object(
                run_benchmark_script,
                "get_runtime_device_report",
                return_value={"torch_available": False},
            ), patch.object(
                run_benchmark_script,
                "extract_dataset_embeddings",
                return_value={
                    "embeddings": embeddings,
                    "label_ids": label_ids,
                    "failures": [],
                    "from_cache": False,
                    "quality_metrics": quality_metrics,
                },
            ):
                run_benchmark_script.main()

            metadata = json.loads((output / "benchmark_run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["experiment_preset"], "thesis-small")
            self.assertEqual(metadata["quality_grid_preset"], "thesis")
            self.assertTrue((output / "sample_manifest.csv").exists())
            self.assertTrue(Path(metadata["outputs"]["quality_ablation"]).exists())
            self.assertTrue(Path(metadata["outputs"]["cluster_ablation"]).exists())

    def test_extract_embeddings_passes_face_quality_to_custom_extractor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            (dataset / "alice").mkdir(parents=True)
            (dataset / "alice" / "a1.jpg").write_bytes(b"")
            seen = {}

            def fake_extractor(image_path, face_quality_config):
                seen["config"] = face_quality_config
                return np.array([1.0, 0.0], dtype=np.float32), None

            payload = extract_dataset_embeddings(
                dataset,
                embedding_extractor=fake_extractor,
                face_quality_config={
                    "enabled": True,
                    "min_face_size": 72,
                    "min_face_ratio": 0.04,
                    "min_laplacian_var": 120,
                    "max_pose_deviation": 0.45,
                    "blur_eval_size": 128,
                },
            )

            self.assertEqual(payload["embeddings"].shape, (1, 2))
            self.assertEqual(seen["config"]["min_face_size"], 72)
            self.assertEqual(seen["config"]["min_face_ratio"], 0.04)
            self.assertEqual(seen["config"]["blur_eval_size"], 128)

    def test_evaluate_face_quality_grid_and_recommendation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            (dataset / "alice").mkdir(parents=True)
            (dataset / "bob").mkdir(parents=True)
            (dataset / "alice" / "a1.jpg").write_bytes(b"")
            (dataset / "alice" / "a2.jpg").write_bytes(b"")
            (dataset / "bob" / "b1.jpg").write_bytes(b"")
            (dataset / "bob" / "b2.jpg").write_bytes(b"")

            def fake_extractor(image_path, face_quality_config):
                image_name = Path(image_path).name
                if face_quality_config["min_face_size"] >= 72 and image_name.endswith("2.jpg"):
                    return None, "filtered_min_face_size"
                if "alice" in str(image_path):
                    return np.array([1.0, 0.0], dtype=np.float32), None
                return np.array([0.0, 1.0], dtype=np.float32), None

            quality_payload = evaluate_face_quality_grid(
                dataset,
                algorithms=["dbscan"],
                metrics=["cosine"],
                eps_grid=[0.1],
                min_samples_grid=[2],
                quality_grid=[
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
                        "min_face_size": 72,
                        "min_face_ratio": 0.035,
                        "min_laplacian_var": 80.0,
                        "max_pose_deviation": 0.35,
                        "blur_eval_size": 96,
                    },
                ],
                embedding_extractor=fake_extractor,
            )

            self.assertEqual(len(quality_payload["results"]), 2)
            self.assertEqual(
                quality_payload["recommended_face_quality"]["min_face_size"],
                56,
            )
            self.assertEqual(
                recommend_face_quality(quality_payload["results"])["min_face_size"],
                56,
            )
            self.assertIn(
                "min_face_ratio",
                quality_payload["recommended_face_quality"],
            )
            self.assertEqual(
                quality_payload["recommended_face_quality"]["blur_eval_size"],
                96,
            )


if __name__ == "__main__":
    unittest.main()
