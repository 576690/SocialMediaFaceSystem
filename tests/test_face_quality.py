import unittest
from types import SimpleNamespace

import numpy as np

from core.analyzer import AIProcessor
from core.config import app_config


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
        self.original_face_quality = dict(app_config.data["face_quality"])
        self.processor = AIProcessor()
        self.processor.generate_description = lambda image: ""
        app_config.data["face_quality"]["enabled"] = True

    def tearDown(self):
        app_config.data["face_quality"] = self.original_face_quality

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

    def test_face_ratio_filters_same_pixel_face_in_larger_frame(self):
        small_frame = np.random.default_rng(4).integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        large_frame = np.random.default_rng(5).integers(0, 255, size=(2000, 2000, 3), dtype=np.uint8)
        face = _make_face([20, 20, 76, 76])
        quality_config = {
            "enabled": True,
            "min_face_size": 56,
            "min_face_ratio": 0.035,
            "min_laplacian_var": 0,
            "max_pose_deviation": 1.0,
            "blur_eval_size": 96,
        }

        small_candidate = self.processor.filter_face_candidates(
            small_frame,
            faces=[face],
            face_quality_config=quality_config,
        )[0]
        large_candidate = self.processor.filter_face_candidates(
            large_frame,
            faces=[face],
            face_quality_config=quality_config,
        )[0]

        self.assertTrue(small_candidate["accepted"])
        self.assertEqual(large_candidate["reason"], "filtered_min_face_ratio")
        self.assertGreaterEqual(small_candidate["metrics"]["face_ratio"], 0.035)
        self.assertLess(large_candidate["metrics"]["face_ratio"], 0.035)

    def test_pixel_floor_still_filters_small_face_with_large_ratio(self):
        image = np.random.default_rng(6).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        face = _make_face([10, 10, 50, 50])

        candidate = self.processor.filter_face_candidates(
            image,
            faces=[face],
            face_quality_config={
                "enabled": True,
                "min_face_size": 56,
                "min_face_ratio": 0.035,
                "min_laplacian_var": 0,
                "max_pose_deviation": 1.0,
                "blur_eval_size": 96,
            },
        )[0]

        self.assertEqual(candidate["reason"], "filtered_min_face_size")
        self.assertGreater(candidate["metrics"]["face_ratio"], 0.035)

    def test_laplacian_metrics_include_normalized_eval_size(self):
        image = np.random.default_rng(7).integers(0, 255, size=(180, 180, 3), dtype=np.uint8)
        face = _make_face([20, 20, 100, 120])

        candidate = self.processor.filter_face_candidates(
            image,
            faces=[face],
            face_quality_config={
                "enabled": True,
                "min_face_size": 56,
                "min_face_ratio": 0.035,
                "min_laplacian_var": 0,
                "max_pose_deviation": 1.0,
                "blur_eval_size": 64,
            },
        )[0]

        self.assertTrue(candidate["accepted"])
        self.assertEqual(candidate["metrics"]["blur_eval_size"], 64)
        self.assertIn("frame_min_side", candidate["metrics"])

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

    def test_process_image_releases_vlm_after_task(self):
        image = np.random.default_rng(3).integers(0, 255, size=(180, 180, 3), dtype=np.uint8)
        faces = [_make_face([20, 20, 100, 120], embedding=[1.0, 0.0, 0.0])]
        self.processor._ensure_face_app = lambda: SimpleNamespace(get=lambda frame: faces)

        def fake_generate_description(_image):
            self.processor.vlm_processor = object()
            self.processor.vlm_model = object()
            self.processor.vlm_model_id = "microsoft/Florence-2-large-ft"
            return "人物站在舞台中央"

        self.processor.generate_description = fake_generate_description

        results = self.processor.process_image(image)

        self.assertEqual(len(results), 1)
        self.assertIsNone(self.processor.vlm_model)
        self.assertIsNone(self.processor.vlm_processor)
        self.assertIsNone(self.processor.vlm_model_id)

    def test_transcribe_video_releases_asr_after_transcription(self):
        processor = AIProcessor()

        class FakeSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class FakeModel:
            def transcribe(self, video_path, vad_filter=True, beam_size=3, **kwargs):
                self.last_video_path = video_path
                self.last_kwargs = kwargs
                return [FakeSegment(0.0, 1.2, "测试语音")], None

        fake_model = FakeModel()
        processor.asr_model = fake_model
        processor.asr_backend = "faster_whisper"
        processor.asr_model_size = "medium"
        processor._ensure_asr_model = lambda: fake_model

        segments = processor.transcribe_video("demo.mp4")

        self.assertEqual(segments[0]["text"], "测试语音")
        self.assertIn("initial_prompt", fake_model.last_kwargs)
        self.assertIn("hotwords", fake_model.last_kwargs)
        self.assertIsNone(processor.asr_model)
        self.assertIsNone(processor.asr_backend)
        self.assertIsNone(processor.asr_model_size)


if __name__ == "__main__":
    unittest.main()
