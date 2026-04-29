import json
import shutil
import tempfile
import unittest
from pathlib import Path

from core.config import DEFAULT_CONFIG, app_config
from core.x_adapter import XUserCollector


class FakeObject:
    def __init__(self, **data):
        self.data = data
        for key, value in data.items():
            setattr(self, key, value)


class FakeResponse:
    def __init__(self, data=None, includes=None, meta=None):
        self.data = data
        self.includes = includes or {}
        self.meta = meta or {}


class FakeTweepyClient:
    def __init__(self, bearer_token=None, wait_on_rate_limit=False):
        self.bearer_token = bearer_token
        self.wait_on_rate_limit = wait_on_rate_limit
        self.timeline_calls = []

    def get_user(self, username=None, user_fields=None):
        return FakeResponse(
            data=FakeObject(
                id="42",
                name="Demo User",
                username=username,
                profile_image_url="https://example.com/profile.jpg",
            )
        )

    def get_users_tweets(self, id=None, **kwargs):
        self.timeline_calls.append((id, kwargs))
        return FakeResponse(
            data=[
                FakeObject(
                    id="100",
                    text="hello face photo",
                    created_at="2026-04-01T00:00:00Z",
                    attachments={"media_keys": ["3_100", "3_101"]},
                    public_metrics={
                        "retweet_count": 2,
                        "reply_count": 1,
                        "like_count": 5,
                        "quote_count": 0,
                    },
                ),
                FakeObject(
                    id="101",
                    text="hello video only",
                    created_at="2026-04-01T00:01:00Z",
                    attachments={"media_keys": ["7_101"]},
                    public_metrics={},
                ),
                FakeObject(
                    id="102",
                    text="other photo",
                    created_at="2026-04-01T00:02:00Z",
                    attachments={"media_keys": ["3_102"]},
                    public_metrics={},
                ),
            ],
            includes={
                "media": [
                    FakeObject(
                        media_key="3_100",
                        type="photo",
                        url="https://pbs.twimg.com/media/photo.jpg",
                    ),
                    FakeObject(
                        media_key="3_101",
                        type="photo",
                        url="https://pbs.twimg.com/media/photo.jpg",
                    ),
                    FakeObject(
                        media_key="7_101",
                        type="video",
                        preview_image_url="https://pbs.twimg.com/media/video.jpg",
                    ),
                    FakeObject(
                        media_key="3_102",
                        type="photo",
                        url="https://pbs.twimg.com/media/other.jpg",
                    ),
                ]
            },
            meta={},
        )


class XUserCollectorTests(unittest.TestCase):
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

    def test_normalize_user_source_accepts_username_and_urls(self):
        collector = XUserCollector()

        self.assertEqual(
            collector.normalize_user_source("@OpenAI")["canonical_url"],
            "https://x.com/OpenAI",
        )
        self.assertEqual(
            collector.normalize_user_source("https://twitter.com/OpenAI/status/1")[
                "canonical_url"
            ],
            "https://x.com/OpenAI",
        )
        self.assertEqual(
            collector.normalize_user_source("x.com/OpenAI")["username"],
            "OpenAI",
        )

    def test_missing_bearer_token_returns_clear_error(self):
        collector = XUserCollector(client_cls=FakeTweepyClient)

        with self.assertRaisesRegex(RuntimeError, "X bearer token file is missing"):
            collector.fetch_user_posts("@OpenAI", limit=1)

    def test_fetch_user_posts_imports_only_keyword_matching_photo_tweets(self):
        app_config.x_bearer_token_path.write_text("Bearer demo-token", encoding="utf-8")
        collector = XUserCollector(client_cls=FakeTweepyClient)

        result = collector.fetch_user_posts("@OpenAI", limit=3, keywords=["hello"])

        self.assertEqual(result["platform"], "x")
        self.assertEqual(result["title"], "Demo User")
        self.assertEqual(result["user_id"], "42")
        self.assertEqual(result["source_url"], "https://x.com/OpenAI")
        self.assertEqual(result["stats"]["fetched_count"], 3)
        self.assertEqual(result["stats"]["matched_count"], 1)
        self.assertEqual(result["stats"]["filtered_count"], 2)
        self.assertEqual(result["cursor"]["last_seen_post_id"], "100")
        self.assertEqual(len(result["entries"]), 1)
        entry = result["entries"][0]
        self.assertEqual(entry["platform"], "x")
        self.assertEqual(entry["content_type"], "post")
        self.assertEqual(entry["external_id"], "100")
        self.assertEqual(
            entry["image_urls"],
            ["https://pbs.twimg.com/media/photo.jpg"],
        )
        self.assertEqual(entry["metadata"]["username"], "OpenAI")
        self.assertEqual(entry["metadata"]["like_count"], 5)

    def test_fetch_user_posts_uses_platform_default_limit(self):
        app_config.x_bearer_token_path.write_text("Bearer demo-token", encoding="utf-8")
        app_config.data["collection"]["platforms"]["x"]["sync_limit"] = 1
        collector = XUserCollector(client_cls=FakeTweepyClient)

        result = collector.fetch_user_posts("@OpenAI")

        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(result["entries"][0]["external_id"], "100")


if __name__ == "__main__":
    unittest.main()
