import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.benchmark import discover_identity_dataset
from core.benchmark import evaluate_face_quality_grid
from core.benchmark import export_benchmark_results
from core.benchmark import extract_dataset_embeddings
from core.benchmark import recommend_face_quality
from core.benchmark import run_benchmark_suite


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
