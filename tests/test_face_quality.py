import unittest
from types import SimpleNamespace

import numpy as np

from core.analyzer import AIProcessor


def _make_face(bbox, kps=None, embedding=None):
    if kps is None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        kps = np.array(
            [
                [x1 + 12, y1 + 18],
                [x2 - 12, y1 + 18],
                [cx, y1 + 34],
                [x1 + 16, y2 - 14],
                [x2 - 16, y2 - 14],
            ],
            dtype=np.float32,
        )
    return SimpleNamespace(
        bbox=np.asarray(bbox, dtype=np.float32),
        kps=np.asarray(kps, dtype=np.float32),
        embedding=np.asarray(embedding or [1.0, 0.0, 0.0], dtype=np.float32),
    )


class FaceQualityTests(unittest.TestCase):
    def setUp(self):
        self.processor = AIProcessor()
        self.processor.generate_description = lambda image: ""

    def test_process_image_keeps_all_valid_faces(self):
        image = np.random.default_rng(0).integers(0, 255, size=(180, 180, 3), dtype=np.uint8)
        faces = [
            _make_face([20, 20, 100, 120], embedding=[1.0, 0.0, 0.0]),
            _make_face([90, 30, 160, 120], embedding=[0.0, 1.0, 0.0]),
        ]
        self.processor._ensure_face_app = lambda: SimpleNamespace(get=lambda frame: faces)

        results = self.processor.process_image(image)

        self.assertEqual(len(results), 2)
        self.assertIn("face_metrics", results[0])
        self.assertGreater(results[0]["face_metrics"]["laplacian_var"], 0.0)

    def test_get_face_embedding_result_prefers_best_valid_face(self):
        image = np.random.default_rng(1).integers(0, 255, size=(220, 220, 3), dtype=np.uint8)
        faces = [
            _make_face([10, 20, 90, 130], embedding=[1.0, 0.0, 0.0]),
            _make_face([100, 40, 150, 105], embedding=[0.0, 1.0, 0.0]),
        ]
        self.processor._ensure_face_app = lambda: SimpleNamespace(get=lambda frame: faces)

        result = self.processor.get_face_embedding_result(image)

        self.assertIsNone(result["failure_reason"])
        self.assertEqual(result["bbox"].tolist(), [10, 20, 90, 130])

    def test_filter_face_candidates_reports_specific_reasons(self):
        image = np.random.default_rng(2).integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        image[30:120, 60:130] = 127
        faces = [
            _make_face([10, 10, 42, 42]),
            _make_face([60, 30, 130, 120]),
            _make_face(
                [120, 30, 190, 120],
                kps=np.array(
                    [
                        [126, 40],
                        [185, 100],
                        [190, 55],
                        [138, 112],
                        [172, 150],
                    ],
                    dtype=np.float32,
                ),
            ),
        ]

        candidates = self.processor.filter_face_candidates(image, faces=faces)

        self.assertEqual(candidates[0]["reason"], "filtered_min_face_size")
        self.assertEqual(candidates[1]["reason"], "filtered_blur")
        self.assertEqual(candidates[2]["reason"], "filtered_pose")

    def test_disabling_quality_filter_keeps_blurry_medium_face(self):
        image = np.full((160, 160, 3), 80, dtype=np.uint8)
        faces = [_make_face([40, 40, 90, 95])]

        candidates = self.processor.filter_face_candidates(
            image,
            faces=faces,
            face_quality_config={"enabled": False},
        )

        self.assertTrue(candidates[0]["accepted"])
        self.assertIsNone(candidates[0]["reason"])


if __name__ == "__main__":
    unittest.main()
