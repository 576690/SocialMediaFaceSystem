import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
import core.database as database_module
from core.config import DEFAULT_CONFIG, app_config


class AppApiTests(unittest.TestCase):
    def setUp(self):
        self.original_db_path = database_module.DB_PATH
        self.original_index_path = database_module.INDEX_PATH
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
            "data": json.loads(json.dumps(app_config.data)),
        }
        self.temp_root = Path(tempfile.mkdtemp())
        self._patch_app_config(self.temp_root)
        database_module.DB_PATH = str(self.temp_root / "metadata.db")
        database_module.INDEX_PATH = str(self.temp_root / "face_index.faiss")
        app_module._clear_all_admin_sessions()
        app_module._rebuild_runtime_state()
        self.client = TestClient(app_module.app)

    def tearDown(self):
        database_module.DB_PATH = self.original_db_path
        database_module.INDEX_PATH = self.original_index_path

        app_module._clear_all_admin_sessions()
        app_config.storage_dir = self.original_app_config["storage_dir"]
        app_config.videos_dir = self.original_app_config["videos_dir"]
        app_config.faces_dir = self.original_app_config["faces_dir"]
        app_config.content_dir = self.original_app_config["content_dir"]
        app_config.asr_dir = self.original_app_config["asr_dir"]
        app_config.test_artifacts_dir = self.original_app_config["test_artifacts_dir"]
        app_config.system_config_path = self.original_app_config["system_config_path"]
        app_config.weibo_cookie_path = self.original_app_config["weibo_cookie_path"]
        app_config.bilibili_cookie_path = self.original_app_config["bilibili_cookie_path"]
        app_config.data = self.original_app_config["data"]
        app_config.ensure_dirs()
        app_config.save()
        app_module._rebuild_runtime_state()
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
        app_config.data = json.loads(json.dumps(DEFAULT_CONFIG))
        app_config.ensure_dirs()
        app_config.save()

    def _setup_admin(self, password="secret123"):
        response = self.client.post("/api/admin/setup", json={"password": password})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        return password

    def test_admin_status_reports_uninitialized_state(self):
        response = self.client.get("/api/admin/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertFalse(payload["admin_initialized"])
        self.assertFalse(payload["admin_authenticated"])

    def test_admin_setup_persists_hash_and_disallows_repeat(self):
        self._setup_admin()

        status_payload = self.client.get("/api/admin/status").json()
        config_payload = json.loads(app_config.system_config_path.read_text(encoding="utf-8"))
        repeat_payload = self.client.post("/api/admin/setup", json={"password": "another123"}).json()

        self.assertTrue(status_payload["admin_initialized"])
        self.assertTrue(status_payload["admin_authenticated"])
        self.assertTrue(config_payload["admin"]["password_hash"])
        self.assertTrue(config_payload["admin"]["salt"])
        self.assertNotEqual(config_payload["admin"]["password_hash"], "secret123")
        self.assertEqual(repeat_payload["status"], "error")

    def test_protected_endpoints_require_login_after_setup(self):
        password = self._setup_admin()
        self.client.post("/api/admin/logout")

        protected_payload = self.client.post(
            "/api/system/face-quality",
            json={
                "enabled": True,
                "min_face_size": 72,
                "min_laplacian_var": 120,
                "max_pose_deviation": 0.45,
            },
        ).json()
        bad_login_payload = self.client.post("/api/admin/login", json={"password": "wrong-password"}).json()
        good_login_payload = self.client.post("/api/admin/login", json={"password": password}).json()
        update_payload = self.client.post(
            "/api/system/face-quality",
            json={
                "enabled": True,
                "min_face_size": 72,
                "min_laplacian_var": 120,
                "max_pose_deviation": 0.45,
            },
        ).json()

        self.assertEqual(protected_payload["status"], "error")
        self.assertTrue(protected_payload["admin_required"])
        self.assertEqual(bad_login_payload["status"], "error")
        self.assertEqual(good_login_payload["status"], "success")
        self.assertEqual(update_payload["status"], "success")
        self.assertEqual(update_payload["face_quality_config"]["min_face_size"], 72)

    def test_public_endpoints_remain_available_without_admin_login(self):
        self._setup_admin()
        self.client.post("/api/admin/logout")

        status_payload = self.client.get("/api/system/status").json()
        people_payload = self.client.get("/api/people").json()

        self.assertEqual(status_payload["status"], "success")
        self.assertIn("runtime_config", status_payload)
        self.assertEqual(people_payload["status"], "success")
        self.assertIn("people", people_payload)

    def test_system_config_update_persists_and_rejects_invalid_payload(self):
        self._setup_admin()

        success_payload = self.client.post(
            "/api/system/config",
            json={
                "processing": {"frame_sample_seconds": 1.5},
                "search": {
                    "text_threshold": 0.25,
                    "image_cosine_threshold": 0.55,
                    "text_top_k": 12,
                    "image_top_k": 8,
                    "semantic_model_id": "Qwen/Qwen3-Embedding-0.6B",
                    "semantic_model_prompt_name": "query",
                    "semantic_model_mode": "standard",
                    "semantic_corpus_style": "structured_zh",
                },
                "collection": {
                    "source_sync_limit": 15,
                    "weibo_cookie_enabled": False,
                    "weibo_source_sync_limit": 9,
                    "weibo_timeout_seconds": 30,
                    "weibo_retry_count": 4,
                },
                "transcription": {
                    "enabled": True,
                    "preferred_backend": "faster_whisper",
                    "model_size": "medium",
                    "initial_prompt": "以下内容来自活动现场，请优先识别人名和品牌名。",
                    "hotwords": ["发布会", "路演", "采访"],
                },
                "vision": {
                    "vlm_model_id": "microsoft/Florence-2-large-ft",
                    "release_vlm_after_task": True,
                    "release_text_encoder_before_vlm": True,
                    "caption_style": "retrieval_keywords",
                    "caption_language": "zh",
                    "caption_include_ocr_hint": True,
                },
                "clustering": {
                    "algorithm": "optics",
                    "metric": "euclidean",
                    "eps": 0.7,
                    "min_samples": 3,
                },
                "face_quality": {
                    "enabled": False,
                    "min_face_size": 64,
                    "min_laplacian_var": 100,
                    "max_pose_deviation": 0.4,
                },
            },
        ).json()

        invalid_payload = self.client.post(
            "/api/system/config",
            json={"search": {"text_threshold": 1.5, "image_cosine_threshold": 0.4, "text_top_k": 10, "image_top_k": 10}},
        ).json()
        invalid_hotwords_payload = self.client.post(
            "/api/system/config",
            json={"transcription": {"hotwords": {"bad": "shape"}}},
        ).json()
        status_payload = self.client.get("/api/system/status").json()
        saved_config = json.loads(app_config.system_config_path.read_text(encoding="utf-8"))

        self.assertEqual(success_payload["status"], "success")
        self.assertEqual(success_payload["runtime_config"]["processing"]["frame_sample_seconds"], 1.5)
        self.assertEqual(success_payload["runtime_config"]["collection"]["source_sync_limit"], 15)
        self.assertEqual(success_payload["runtime_config"]["clustering"]["algorithm"], "optics")
        self.assertEqual(
            success_payload["runtime_config"]["search"]["semantic_model_id"],
            "Qwen/Qwen3-Embedding-0.6B",
        )
        self.assertEqual(
            success_payload["runtime_config"]["vision"]["vlm_model_id"],
            "microsoft/Florence-2-large-ft",
        )
        self.assertEqual(
            success_payload["runtime_config"]["search"]["semantic_corpus_style"],
            "structured_zh",
        )
        self.assertEqual(
            success_payload["runtime_config"]["vision"]["caption_style"],
            "retrieval_keywords",
        )
        self.assertEqual(
            success_payload["runtime_config"]["vision"]["caption_language"],
            "zh",
        )
        self.assertTrue(success_payload["runtime_config"]["vision"]["caption_include_ocr_hint"])
        self.assertEqual(
            success_payload["runtime_config"]["transcription"]["initial_prompt"],
            "以下内容来自活动现场，请优先识别人名和品牌名。",
        )
        self.assertEqual(
            success_payload["runtime_config"]["transcription"]["hotwords"],
            ["发布会", "路演", "采访"],
        )
        self.assertFalse(success_payload["runtime_config"]["face_quality"]["enabled"])

        self.assertEqual(invalid_payload["status"], "error")
        self.assertEqual(invalid_hotwords_payload["status"], "error")
        self.assertEqual(status_payload["runtime_config"]["search"]["text_threshold"], 0.25)
        self.assertEqual(saved_config["search"]["text_threshold"], 0.25)
        self.assertEqual(saved_config["search"]["semantic_model_prompt_name"], "query")
        self.assertEqual(saved_config["search"]["semantic_corpus_style"], "structured_zh")
        self.assertEqual(saved_config["vision"]["vlm_model_id"], "microsoft/Florence-2-large-ft")
        self.assertEqual(saved_config["vision"]["caption_style"], "retrieval_keywords")
        self.assertEqual(saved_config["transcription"]["hotwords"], ["发布会", "路演", "采访"])
        self.assertEqual(saved_config["collection"]["weibo_cookie_enabled"], False)

    def test_system_reset_preserves_videos_directory_contents(self):
        self._setup_admin()

        video_file = app_config.videos_dir / "keep.mp4"
        face_file = app_config.faces_dir / "face.jpg"
        content_file = app_config.content_dir / "post.jpg"
        asr_file = app_config.asr_dir / "clip.srt"
        artifact_file = app_config.test_artifacts_dir / "report.txt"

        video_file.write_text("video", encoding="utf-8")
        face_file.write_text("face", encoding="utf-8")
        content_file.write_text("content", encoding="utf-8")
        asr_file.write_text("asr", encoding="utf-8")
        artifact_file.write_text("artifact", encoding="utf-8")

        payload = self.client.post(
            "/api/system/reset",
            json={"preserve_cookies": True},
        ).json()

        self.assertEqual(payload["status"], "success")
        self.assertTrue(video_file.exists())
        self.assertFalse(face_file.exists())
        self.assertFalse(content_file.exists())
        self.assertFalse(asr_file.exists())
        self.assertFalse(artifact_file.exists())
        self.assertGreaterEqual(payload["removed_files_count"], 5)


if __name__ == "__main__":
    unittest.main()
