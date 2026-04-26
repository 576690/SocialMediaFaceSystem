import unittest
from unittest.mock import patch

import torch

from core.analyzer import AIProcessor
from core.config import app_config


class SemanticTests(unittest.TestCase):
    def test_build_caption_prompt_uses_configured_style(self):
        processor = AIProcessor()
        original_style = app_config.data["vision"]["caption_style"]
        try:
            app_config.data["vision"]["caption_style"] = "retrieval_keywords"
            self.assertEqual(processor._build_caption_prompt(), "<MORE_DETAILED_CAPTION>")
            app_config.data["vision"]["caption_style"] = "detailed_caption"
            self.assertEqual(processor._build_caption_prompt(), "<DETAILED_CAPTION>")
        finally:
            app_config.data["vision"]["caption_style"] = original_style

    def test_normalize_visual_caption_removes_english_boilerplate(self):
        processor = AIProcessor()

        normalized = processor._normalize_visual_caption(
            "In this image we can see a woman wearing glasses and holding a microphone on a stage."
        )

        self.assertNotIn("In this image", normalized)
        self.assertIn("女性", normalized)
        self.assertIn("眼镜", normalized)
        self.assertIn("麦克风", normalized)
        self.assertIn("舞台", normalized)

    def test_compose_semantic_text_outputs_structured_chinese_sections(self):
        processor = AIProcessor()

        semantic_text, sources = processor.compose_semantic_text(
            visual_text="a woman wearing glasses and speaking on stage with visible text",
            subtitle_text="欢迎来到直播现场",
            post_text="演出返图",
        )

        self.assertIn("人物特征", semantic_text)
        self.assertIn("动作场景", semantic_text)
        self.assertIn("画面文字", semantic_text)
        self.assertIn("语音文本：欢迎来到直播现场", semantic_text)
        self.assertIn("正文线索：演出返图", semantic_text)
        self.assertEqual(sources, ["visual", "subtitle", "post"])

    def test_compose_semantic_text_prefers_subtitle_and_omits_duplicate_post(self):
        processor = AIProcessor()

        semantic_text, sources = processor.compose_semantic_text(
            visual_text="a man standing indoors",
            subtitle_text="同一句话",
            asr_text="另一句识别文本",
            post_text="同一句话",
        )

        self.assertIn("语音文本：同一句话", semantic_text)
        self.assertNotIn("另一句识别文本", semantic_text)
        self.assertNotIn("正文线索：同一句话", semantic_text)
        self.assertEqual(sources, ["visual", "subtitle"])

    def test_asr_model_candidates_for_medium(self):
        processor = AIProcessor()
        self.assertEqual(
            processor._build_asr_model_candidates("medium"),
            ["medium", "small", "tiny"],
        )

    def test_runtime_config_exposes_prompt_defaults(self):
        runtime_config = app_config.runtime_config()

        self.assertEqual(runtime_config["search"]["semantic_model_prompt_name"], "query")
        self.assertEqual(runtime_config["search"]["semantic_corpus_style"], "structured_zh")
        self.assertEqual(runtime_config["vision"]["caption_style"], "retrieval_keywords")
        self.assertEqual(runtime_config["vision"]["caption_language"], "zh")
        self.assertTrue(runtime_config["vision"]["caption_include_ocr_hint"])
        self.assertIn("直播", runtime_config["transcription"]["hotwords"])

    def test_rank_texts_uses_query_prompt_only_for_query_side(self):
        processor = AIProcessor()
        encode_calls = []

        class FakeEncoder:
            def encode(self, value, **kwargs):
                encode_calls.append((value, kwargs))
                if isinstance(value, list):
                    return torch.tensor([[0.4, 0.6], [0.8, 0.2]], dtype=torch.float32)
                return torch.tensor([0.3, 0.7], dtype=torch.float32)

        processor._ensure_text_encoder = lambda: FakeEncoder()

        with patch(
            "sentence_transformers.util.cos_sim",
            return_value=torch.tensor([[0.92, 0.31]], dtype=torch.float32),
        ):
            ranked = processor.rank_texts_by_similarity(
                " 测试\n查询 ",
                [
                    "[Visual] a woman on stage [Speech] 欢迎来到现场 [Post] 演出返图",
                    "人物特征：男性\n动作场景：室内采访",
                ],
            )

        self.assertAlmostEqual(ranked[0][1], 0.92, places=6)
        self.assertAlmostEqual(ranked[1][1], 0.31, places=6)
        self.assertEqual(encode_calls[0][1]["prompt_name"], "query")
        self.assertNotIn("prompt_name", encode_calls[1][1])
        self.assertEqual(encode_calls[0][0], "测试 查询")
        self.assertIn("人物特征", encode_calls[1][0][0])
        self.assertIn("语音文本", encode_calls[1][0][0])


if __name__ == "__main__":
    unittest.main()
