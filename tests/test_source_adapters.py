import json
import shutil
import tempfile
import unittest
from pathlib import Path

from core.config import DEFAULT_CONFIG, app_config
from core.source_adapters import (
    SourceAdapterError,
    SourceAdapterRegistry,
    parse_adapter_config,
)


class FakeCollector:
    def detect_platform(self, url):
        if "bilibili.com" in url:
            return "bilibili"
        if "x.com" in url:
            return "x"
        return "generic"

    def normalize_source_url(self, source_url, platform=None, source_type=None):
        if platform == "weibo":
            return "https://weibo.cn/123456"
        return source_url

    def normalize_keywords(self, keywords):
        return [str(item).strip() for item in (keywords or []) if str(item).strip()]

    def fetch_source_entries(
        self,
        source_url,
        limit=None,
        platform=None,
        source_type=None,
        metadata=None,
    ):
        if platform == "weibo":
            return {
                "platform": "weibo",
                "title": "weibo user",
                "user_id": "123456",
                "source_url": "https://weibo.cn/123456",
                "entries": [
                    {
                        "platform": "weibo",
                        "external_id": "post-1",
                        "title": "post",
                        "url": "https://weibo.cn/comment/post-1",
                        "post_text": "hello",
                        "image_urls": ["https://example.com/a.jpg"],
                    }
                ],
                "stats": {"fetched_count": 1, "matched_count": 1, "filtered_count": 0},
                "cursor": {"last_seen_post_id": "post-1"},
            }
        return {
            "platform": platform or "generic",
            "title": "channel",
            "entries": [
                {
                    "platform": platform or "generic",
                    "external_id": "video-1",
                    "title": "video",
                    "url": "https://example.com/video-1",
                }
            ],
        }

    def get_platform_request_headers(self, platform, referer=None):
        return {"Referer": referer}


class SourceAdapterRegistryTests(unittest.TestCase):
    def setUp(self):
        self.original_config = {
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
        app_config.storage_dir = self.temp_root
        app_config.videos_dir = self.temp_root / "videos"
        app_config.faces_dir = self.temp_root / "faces"
        app_config.content_dir = self.temp_root / "content"
        app_config.asr_dir = self.temp_root / "artifacts" / "asr"
        app_config.test_artifacts_dir = self.temp_root / "test_artifacts"
        app_config.system_config_path = self.temp_root / "system_config.json"
        app_config.weibo_cookie_path = self.temp_root / "weibo_cookies.txt"
        app_config.bilibili_cookie_path = self.temp_root / "bilibili_cookies.txt"
        app_config.x_bearer_token_path = self.temp_root / "x_bearer_token.txt"
        app_config.data = json.loads(json.dumps(DEFAULT_CONFIG))
        app_config.ensure_dirs()
        app_config.save()

    def tearDown(self):
        app_config.storage_dir = self.original_config["storage_dir"]
        app_config.videos_dir = self.original_config["videos_dir"]
        app_config.faces_dir = self.original_config["faces_dir"]
        app_config.content_dir = self.original_config["content_dir"]
        app_config.asr_dir = self.original_config["asr_dir"]
        app_config.test_artifacts_dir = self.original_config["test_artifacts_dir"]
        app_config.system_config_path = self.original_config["system_config_path"]
        app_config.weibo_cookie_path = self.original_config["weibo_cookie_path"]
        app_config.bilibili_cookie_path = self.original_config["bilibili_cookie_path"]
        app_config.x_bearer_token_path = self.original_config["x_bearer_token_path"]
        app_config.data = self.original_config["data"]
        app_config.ensure_dirs()
        app_config.save()
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_builtin_adapters_normalize_entries(self):
        registry = SourceAdapterRegistry(FakeCollector())

        weibo_adapter = registry.select_adapter(
            "weibo.com/u/123456",
            platform="weibo",
            source_type="weibo_user",
        )
        weibo_result = weibo_adapter.fetch_entries(
            {
                "source_url": "weibo.com/u/123456",
                "platform": "weibo",
                "source_type": "weibo_user",
                "metadata": {"keywords": ["hello"]},
            },
            limit=1,
        )
        yt_adapter = registry.select_adapter(
            "https://space.bilibili.com/1",
            platform="bilibili",
            source_type="channel",
        )
        yt_result = yt_adapter.fetch_entries(
            {
                "source_url": "https://space.bilibili.com/1",
                "platform": "bilibili",
                "source_type": "channel",
                "metadata": {},
            },
            limit=1,
        )

        self.assertEqual(weibo_adapter.adapter_id, "weibo_user")
        self.assertEqual(weibo_result["entries"][0]["content_type"], "post")
        self.assertEqual(weibo_adapter.get_request_headers(weibo_result["entries"][0])["Referer"], "https://weibo.cn/comment/post-1")
        self.assertIn("x_user", [item["adapter_id"] for item in registry.list_adapters()])
        self.assertEqual(
            registry.select_adapter(
                "https://x.com/OpenAI",
                platform="x",
                source_type="x_user",
            ).adapter_id,
            "x_user",
        )
        self.assertEqual(yt_adapter.adapter_id, "yt_dlp")
        self.assertEqual(yt_result["entries"][0]["content_type"], "video")

    def test_config_adapter_save_reload_and_validation(self):
        registry = SourceAdapterRegistry(FakeCollector())
        config = parse_adapter_config(
            json.dumps(
                {
                    "adapter_id": "custom_site",
                    "display_name": "Custom Site",
                    "platform": "custom_site",
                    "enabled": True,
                    "source_types": ["user", "channel"],
                    "url_patterns": ["https://example.com/*"],
                    "default_limit": 7,
                    "settings": {},
                }
            ),
            "custom_site.json",
        )

        adapter = registry.save_config(config)
        reloaded = SourceAdapterRegistry(FakeCollector())
        selected = reloaded.select_adapter(
            "https://example.com/u/1",
            platform="custom_site",
            source_type="user",
        )

        self.assertEqual(adapter["adapter_id"], "custom_site")
        self.assertFalse(selected.builtin)
        self.assertEqual(selected.default_limit, 7)
        with self.assertRaises(SourceAdapterError):
            registry.save_config({"adapter_id": "yt_dlp", "platform": "x", "source_types": ["channel"]})

    def test_invalid_config_rejected(self):
        with self.assertRaises(SourceAdapterError):
            parse_adapter_config(
                json.dumps({"adapter_id": "bad id", "platform": "x", "source_types": ["channel"]}),
                "bad.json",
            )
        with self.assertRaises(SourceAdapterError):
            parse_adapter_config(
                json.dumps({"adapter_id": "custom", "platform": "x"}),
                "bad.json",
            )


if __name__ == "__main__":
    unittest.main()
