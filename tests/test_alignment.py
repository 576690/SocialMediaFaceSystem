import tempfile
import unittest
from pathlib import Path

from core.alignment import align_text_to_timestamp, parse_srt_file, write_srt_file


class AlignmentTests(unittest.TestCase):
    def test_align_hits_direct_segment(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 2.0, "end": 3.0, "text": "world"},
        ]
        self.assertEqual(align_text_to_timestamp(segments, 2.5), "world")

    def test_align_uses_nearest_within_tolerance(self):
        segments = [{"start": 5.0, "end": 6.0, "text": "near"}]
        self.assertEqual(align_text_to_timestamp(segments, 4.0, tolerance=1.5), "near")
        self.assertEqual(align_text_to_timestamp(segments, 2.0, tolerance=1.5), "")

    def test_write_and_parse_srt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.srt"
            write_srt_file(
                [
                    {"start": 0.0, "end": 1.0, "text": "first line"},
                    {"start": 2.0, "end": 3.0, "text": "second line"},
                ],
                str(path),
            )
            parsed = parse_srt_file(str(path))
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["text"], "first line")


if __name__ == "__main__":
    unittest.main()
