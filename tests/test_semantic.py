import unittest

from core.analyzer import AIProcessor


class SemanticTests(unittest.TestCase):
    def test_compose_semantic_text(self):
        processor = AIProcessor()
        semantic_text, sources = processor.compose_semantic_text(
            visual_text="a woman speaking on stage",
            subtitle_text="欢迎来到直播现场",
            post_text="演出返图",
        )
        self.assertIn("[Visual]", semantic_text)
        self.assertIn("[Speech]", semantic_text)
        self.assertIn("[Post]", semantic_text)
        self.assertEqual(sources, ["visual", "subtitle", "post"])

    def test_asr_model_candidates_for_medium(self):
        processor = AIProcessor()
        self.assertEqual(
            processor._build_asr_model_candidates("medium"),
            ["medium", "small", "tiny"],
        )


if __name__ == "__main__":
    unittest.main()
