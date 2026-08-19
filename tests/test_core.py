"""
Unit tests for ViralClipper AI core modules.
Run with: python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFormatTimestampAss(unittest.TestCase):

    def test_zero(self):
        from transcriber import format_timestamp_ass
        self.assertEqual(format_timestamp_ass(0), "0:00:00.00")

    def test_minutes(self):
        from transcriber import format_timestamp_ass
        self.assertEqual(format_timestamp_ass(65.5), "0:01:05.50")

    def test_hours(self):
        from transcriber import format_timestamp_ass
        self.assertEqual(format_timestamp_ass(3661.99), "1:01:01.99")

    def test_none(self):
        from transcriber import format_timestamp_ass
        self.assertEqual(format_timestamp_ass(None), "0:00:00.00")

    def test_negative_clamps_to_zero(self):
        from transcriber import format_timestamp_ass
        self.assertEqual(format_timestamp_ass(-5), "0:00:00.00")


class TestCreateWordChunks(unittest.TestCase):

    def test_empty_segments(self):
        from transcriber import create_word_chunks
        self.assertEqual(create_word_chunks([]), [])

    def test_no_words_key(self):
        from transcriber import create_word_chunks
        segments = [{"text": "hello world", "start": 0, "end": 2}]
        self.assertEqual(create_word_chunks(segments), [])

    def test_basic_chunking(self):
        from transcriber import create_word_chunks
        segments = [{"words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]}]
        chunks = create_word_chunks(segments)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "HELLO WORLD")
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 1.0)

    def test_chunk_breaks_on_max_words(self):
        from transcriber import create_word_chunks
        from config import SubtitleStyle
        style = SubtitleStyle(max_words_per_line=2, max_chars_per_line=100)
        segments = [{"words": [
            {"word": "one", "start": 0.0, "end": 0.3},
            {"word": "two", "start": 0.3, "end": 0.6},
            {"word": "three", "start": 0.6, "end": 1.0},
        ]}]
        chunks = create_word_chunks(segments, style=style)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "one two")
        self.assertEqual(chunks[1]["text"], "THREE")

    def test_skips_none_timestamps(self):
        from transcriber import create_word_chunks
        segments = [{"words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "", "start": None, "end": None},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]}]
        chunks = create_word_chunks(segments)
        self.assertEqual(len(chunks), 1)
        # Last chunk gets uppercased by create_word_chunks
        self.assertIn("HELLO", chunks[0]["text"])


class TestExtractJsonFromResponse(unittest.TestCase):

    def test_clean_json(self):
        from analyzer import _extract_json_from_response
        content = '[{"start_time": 10, "end_time": 50, "virality_score": 90, "reasoning": "test"}]'
        result = _extract_json_from_response(content)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_markdown_fenced_json(self):
        from analyzer import _extract_json_from_response
        content = '```json\n[{"start_time": 10, "end_time": 50, "virality_score": 80, "reasoning": "ok"}]\n```'
        result = _extract_json_from_response(content)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_json_with_surrounding_text(self):
        from analyzer import _extract_json_from_response
        content = 'Here are the clips:\n[{"start_time": 10, "end_time": 50, "virality_score": 70, "reasoning": "good"}]\nHope that helps!'
        result = _extract_json_from_response(content)
        self.assertIsNotNone(result)

    def test_invalid_json(self):
        from analyzer import _extract_json_from_response
        result = _extract_json_from_response("This is not JSON at all")
        self.assertIsNone(result)

    def test_empty_array(self):
        from analyzer import _extract_json_from_response
        result = _extract_json_from_response("[]")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)


class TestSplitTranscriptIntoChunks(unittest.TestCase):

    def test_short_transcript_single_chunk(self):
        from analyzer import _split_transcript_into_chunks
        segments = [{"text": "hello world", "start": 0, "end": 1}]
        chunks = _split_transcript_into_chunks(segments, max_tokens=6000)
        self.assertEqual(len(chunks), 1)

    def test_empty_segments(self):
        from analyzer import _split_transcript_into_chunks
        chunks = _split_transcript_into_chunks([], max_tokens=6000)
        self.assertEqual(len(chunks), 0)

    def test_long_transcript_multiple_chunks(self):
        from analyzer import _split_transcript_into_chunks
        segments = [
            {"text": f"word {i} " * 100, "start": i * 10, "end": (i + 1) * 10}
            for i in range(20)
        ]
        chunks = _split_transcript_into_chunks(segments, max_tokens=500)
        self.assertGreater(len(chunks), 1)


class TestSmartContextExpansion(unittest.TestCase):

    def test_expansion(self):
        from analyzer import _smart_context_expansion
        from config import AnalysisConfig
        config = AnalysisConfig()

        segments = [
            {"text": "setup", "start": 0.0, "end": 5.0},
            {"text": "hook", "start": 5.0, "end": 15.0},
            {"text": "main content", "start": 15.0, "end": 40.0},
            {"text": "punchline", "start": 40.0, "end": 45.0},
            {"text": "outro", "start": 45.0, "end": 50.0},
        ]
        clips = [{"start_time": 15.0, "end_time": 40.0, "virality_score": 90}]
        expanded = _smart_context_expansion(clips, segments, config, target_duration=90)
        self.assertEqual(len(expanded), 1)
        self.assertLessEqual(expanded[0]["start_time"], 15.0)
        self.assertGreaterEqual(expanded[0]["end_time"], 40.0)

    def test_empty_segments(self):
        from analyzer import _smart_context_expansion
        from config import AnalysisConfig
        clips = [{"start_time": 10, "end_time": 20}]
        result = _smart_context_expansion(clips, [], AnalysisConfig())
        self.assertEqual(len(result), 1)


class TestSubtitleStyleDefaults(unittest.TestCase):

    def test_default_values(self):
        from config import SubtitleStyle
        style = SubtitleStyle()
        self.assertEqual(style.font_name, "Arial Black")
        self.assertEqual(style.font_size, 85)
        self.assertEqual(style.max_words_per_line, 2)
        self.assertTrue(style.bold)


class TestClipConfigDefaults(unittest.TestCase):

    def test_default_values(self):
        from config import ClipConfig
        config = ClipConfig()
        self.assertEqual(config.min_duration, 30.0)
        self.assertEqual(config.max_duration, 179.0)
        self.assertEqual(config.aspect_ratio, "9:16")


class TestGenerateReport(unittest.TestCase):

    def test_report_generation(self):
        from report_generator import generate_report
        with tempfile.TemporaryDirectory() as tmpdir:
            clips = [{
                "start_time": 10.0, "end_time": 60.0,
                "duration": 50.0, "virality_score": 85,
                "reasoning": "Test reasoning",
            }]
            transcript = [
                {"text": "hello", "start": 0, "end": 5},
                {"text": "world", "start": 5, "end": 100},
            ]

            clips_json = os.path.join(tmpdir, "clips.json")
            transcript_json = os.path.join(tmpdir, "transcript.json")
            with open(clips_json, "w") as f:
                json.dump(clips, f)
            with open(transcript_json, "w") as f:
                json.dump(transcript, f)

            generate_report(clips_json, transcript_json, tmpdir)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "VIRALITY_REPORT.md")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "engagement_chart.png")))

    def test_empty_clips(self):
        from report_generator import generate_report
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_json = os.path.join(tmpdir, "clips.json")
            transcript_json = os.path.join(tmpdir, "transcript.json")
            with open(clips_json, "w") as f:
                json.dump([], f)
            with open(transcript_json, "w") as f:
                json.dump([{"text": "hi", "start": 0, "end": 5}], f)

            generate_report(clips_json, transcript_json, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "VIRALITY_REPORT.md")))


if __name__ == "__main__":
    unittest.main()
