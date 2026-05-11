import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yt_dlp.utils import DownloadError

from core.collector import VideoCollector
from core.config import DEFAULT_CONFIG, app_config


class VideoCollectorTests(unittest.TestCase):
    def setUp(self):
        self.original_app_config = {
            "storage_dir": app_config.storage_dir,
            "videos_dir": app_config.videos_dir,
            "faces_dir": app_config.faces_dir,
            "content_dir": app_config.content_dir,
            "asr_dir": app_config.asr_dir,
            "test_artifacts_dir": app_config.test_artifacts_dir,
            "system_config_path": app_config.system_config_path,
            "weibo_cookie_path": app_config.weibo_cookie_path,
            "bilibili_cookie_path": app_config.bilibili_cookie_path,
            "x_bearer_token_path": app_config.x_bearer_token_path,
            "data": json.loads(json.dumps(app_config.data)),
        }
        self.temp_root = Path(tempfile.mkdtemp())
        self._patch_app_config(self.temp_root)
        self.collector = VideoCollector()

    def tearDown(self):
        app_config.storage_dir = self.original_app_config["storage_dir"]
        app_config.videos_dir = self.original_app_config["videos_dir"]
        app_config.faces_dir = self.original_app_config["faces_dir"]
        app_config.content_dir = self.original_app_config["content_dir"]
        app_config.asr_dir = self.original_app_config["asr_dir"]
        app_config.test_artifacts_dir = self.original_app_config["test_artifacts_dir"]
        app_config.system_config_path = self.original_app_config["system_config_path"]
        app_config.weibo_cookie_path = self.original_app_config["weibo_cookie_path"]
        app_config.bilibili_cookie_path = self.original_app_config["bilibili_cookie_path"]
        app_config.x_bearer_token_path = self.original_app_config["x_bearer_token_path"]
        app_config.data = self.original_app_config["data"]
        app_config.ensure_dirs()
        app_config.save()
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _patch_app_config(self, temp_root):
        app_config.storage_dir = temp_root
        app_config.videos_dir = temp_root / "videos"
        app_config.faces_dir = temp_root / "faces"
        app_config.content_dir = temp_root / "content"
        app_config.asr_dir = temp_root / "artifacts" / "asr"
        app_config.test_artifacts_dir = temp_root / "test_artifacts"
        app_config.system_config_path = temp_root / "system_config.json"
        app_config.weibo_cookie_path = temp_root / "weibo_cookies.txt"
        app_config.bilibili_cookie_path = temp_root / "bilibili_cookies.txt"
        app_config.x_bearer_token_path = temp_root / "x_bearer_token.txt"
        app_config.data = json.loads(json.dumps(DEFAULT_CONFIG))
        app_config.ensure_dirs()
        app_config.save()

    def test_detect_platform_supports_x_domains(self):
        self.assertEqual(self.collector.detect_platform("https://x.com/OpenAI"), "x")
        self.assertEqual(
            self.collector.detect_platform("https://twitter.com/OpenAI"),
            "x",
        )

    def test_weibo_post_metadata_uses_weibo_adapter_without_ytdlp(self):
        calls = []

        def fake_fetch_single_post(url):
            calls.append(url)
            return {
                "platform": "weibo",
                "content_type": "post",
                "external_id": "5241373692531045",
                "title": "post",
                "post_text": "hello",
                "image_urls": ["https://wx1.sinaimg.cn/large/a.jpg"],
                "source_url": url,
            }

        self.collector.weibo_adapter.fetch_single_post = fake_fetch_single_post
        self.collector._extract_info = lambda *args, **kwargs: self.fail(
            "yt-dlp should not be used for Weibo post metadata"
        )

        result = self.collector.extract_post_metadata(
            "https://weibo.cn/comment/5241373692531045"
        )

        self.assertEqual(calls, ["https://weibo.cn/comment/5241373692531045"])
        self.assertEqual(result["platform"], "weibo")
        self.assertEqual(result["image_urls"], ["https://wx1.sinaimg.cn/large/a.jpg"])

    def test_bilibili_opts_include_impersonate_headers_and_cookiefile(self):
        app_config.bilibili_cookie_path.write_text("SESSDATA=demo", encoding="utf-8")
        self.collector._supports_impersonate_target = lambda target: (True, "supported")

        opts = self.collector._build_ydl_opts("bilibili", download=True)

        self.assertEqual(opts["impersonate"], "chrome")
        self.assertEqual(opts["http_headers"]["Referer"], "https://www.bilibili.com/")
        self.assertIn("Mozilla/5.0", opts["http_headers"]["User-Agent"])
        self.assertEqual(opts["cookiefile"], str(app_config.bilibili_cookie_path))
        self.assertEqual(opts["retry_sleep"], "extractor:1:2")

    def test_bilibili_opts_read_platform_collection_config(self):
        app_config.data["collection"]["platforms"]["bilibili"].update(
            {
                "auth_enabled": False,
                "timeout_seconds": 44,
                "retry_count": 7,
                "impersonate": "chrome110",
                "referer": "https://www.bilibili.com/platform-test",
            }
        )
        app_config.bilibili_cookie_path.write_text("SESSDATA=demo", encoding="utf-8")
        self.collector._supports_impersonate_target = lambda target: (True, "supported")

        opts = self.collector._build_ydl_opts("bilibili", download=True)

        self.assertEqual(opts["http_headers"]["Referer"], "https://www.bilibili.com/platform-test")
        self.assertEqual(opts["impersonate"], "chrome110")
        self.assertEqual(opts["socket_timeout"], 44)
        self.assertEqual(opts["extractor_retries"], 7)
        self.assertNotIn("cookiefile", opts)

    def test_bilibili_opts_drop_impersonate_when_runtime_does_not_support_it(self):
        app_config.bilibili_cookie_path.write_text("SESSDATA=demo", encoding="utf-8")
        self.collector._supports_impersonate_target = (
            lambda target: (False, 'Impersonate target "chrome" requested, but curl_cffi is not installed.')
        )

        opts = self.collector._build_ydl_opts("bilibili", download=True)

        self.assertNotIn("impersonate", opts)
        self.assertEqual(opts["cookiefile"], str(app_config.bilibili_cookie_path))
        self.assertFalse(self.collector.last_bilibili_impersonate_status["impersonate_enabled"])
        self.assertIn("curl_cffi", self.collector.last_bilibili_impersonate_status["impersonate_reason"])

    def test_bilibili_412_without_cookie_returns_clear_message(self):
        class FakeYDL:
            def __init__(self, _opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                raise DownloadError(
                    "ERROR: [BiliBili] 1X7411s7jk: Unable to download JSON metadata: HTTP Error 412: Precondition Failed"
                )

        with patch("core.collector.yt_dlp.YoutubeDL", FakeYDL):
            with self.assertRaises(RuntimeError) as ctx:
                self.collector._extract_info(
                    "https://www.bilibili.com/video/BV1X7411s7jk",
                    "bilibili",
                    download=False,
                    extra_opts=None,
                )

        self.assertIn("storage/bilibili_cookies.txt", str(ctx.exception))

    def test_bilibili_412_with_cookie_returns_refresh_hint(self):
        app_config.bilibili_cookie_path.write_text("SESSDATA=demo", encoding="utf-8")
        self.collector.last_bilibili_impersonate_status = {
            "impersonate_requested": "chrome",
            "impersonate_enabled": False,
            "impersonate_reason": 'Impersonate target "chrome" requested, but curl_cffi is not installed.',
        }

        message = self.collector._format_bilibili_download_error(
            DownloadError("ERROR: [BiliBili] xxx: Unable to download JSON metadata: HTTP Error 412: Precondition Failed")
        )
        self.assertIn("未启用浏览器模拟的降级模式", message)

    def test_impersonate_not_available_error_is_rewritten(self):
        self.collector.last_bilibili_impersonate_status = {
            "impersonate_requested": "chrome",
            "impersonate_enabled": False,
            "impersonate_reason": 'Impersonate target "chrome" requested, but curl_cffi is not installed.',
        }

        message = self.collector._format_bilibili_download_error(
            DownloadError(
                'Impersonate target "chrome" is not available. Use --list-impersonate-targets to see available targets.'
            )
        )

        self.assertIn("当前环境不支持", message)
        self.assertIn("curl_cffi", message)

    def test_download_fetch_and_extract_metadata_route_through_extract_info(self):
        calls = []

        def fake_extract_info(url, platform, download=False, extra_opts=None):
            calls.append((url, platform, download, extra_opts))
            if "space.bilibili.com" in url:
                return (
                    {
                        "id": "channel",
                        "title": "space",
                        "entries": [{"id": "BV1AAA", "title": "video1"}],
                    },
                    "",
                )
            if download:
                return (
                    {
                        "id": "BV1AAA",
                        "title": "video1",
                        "ext": "mp4",
                        "webpage_url": url,
                    },
                    str(app_config.videos_dir / "BV1AAA.mp4"),
                )
            return (
                {
                    "id": "BV1AAA",
                    "title": "video1",
                    "description": "desc",
                    "webpage_url": url,
                    "thumbnail": "https://i0.hdslb.com/demo.jpg",
                },
                "",
            )

        self.collector._extract_info = fake_extract_info
        (app_config.videos_dir / "BV1AAA.mp4").write_bytes(b"demo")

        self.collector.download("https://www.bilibili.com/video/BV1AAA")
        self.collector.fetch_source_entries("https://space.bilibili.com/123", platform="bilibili")
        self.collector.extract_post_metadata("https://www.bilibili.com/video/BV1AAA")

        self.assertEqual(calls[0][1], "bilibili")
        self.assertTrue(calls[0][2])
        self.assertEqual(calls[1][1], "bilibili")
        self.assertFalse(calls[1][2])
        self.assertTrue(calls[1][3]["extract_flat"])
        self.assertEqual(calls[2][1], "bilibili")
        self.assertFalse(calls[2][2])


if __name__ == "__main__":
    unittest.main()
