import json
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
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
        response = self.client.post(
            "/api/auth/setup",
            json={"username": "admin", "password": password},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        return password

    def _login_admin(self, password="secret123"):
        response = self.client.post(
            "/api/auth/login",
            json={"username": "admin", "password": password},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    def _create_user(self, username="operator", password="user1234", role="user"):
        response = self.client.post(
            "/api/users",
            json={"username": username, "password": password, "role": role},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        return response.json()["user"]

    def _login_user(self, username="operator", password="user1234"):
        response = self.client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    def test_auth_status_reports_uninitialized_state(self):
        response = self.client.get("/api/auth/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertFalse(payload["initialized"])
        self.assertFalse(payload["authenticated"])
        self.assertIsNone(payload["user"])

    def test_initial_admin_setup_creates_user_and_disallows_repeat(self):
        self._setup_admin()

        status_payload = self.client.get("/api/auth/status").json()
        repeat_payload = self.client.post(
            "/api/auth/setup",
            json={"username": "another", "password": "another123"},
        ).json()
        users_payload = self.client.get("/api/users").json()

        self.assertTrue(status_payload["initialized"])
        self.assertTrue(status_payload["authenticated"])
        self.assertEqual(status_payload["user"]["username"], "admin")
        self.assertEqual(status_payload["user"]["role"], "admin")
        self.assertEqual(len(users_payload["users"]), 1)
        self.assertNotIn("password_hash", users_payload["users"][0])
        self.assertEqual(repeat_payload["status"], "error")

    def test_protected_endpoints_require_login_after_setup(self):
        password = self._setup_admin()
        self.client.post("/api/auth/logout")

        protected_payload = self.client.post(
            "/api/system/face-quality",
            json={
                "enabled": True,
                "min_face_size": 72,
                "min_face_ratio": 0.04,
                "min_laplacian_var": 120,
                "max_pose_deviation": 0.45,
                "blur_eval_size": 128,
            },
        ).json()
        bad_login_payload = self.client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        ).json()
        good_login_payload = self.client.post(
            "/api/auth/login",
            json={"username": "admin", "password": password},
        ).json()
        update_payload = self.client.post(
            "/api/system/face-quality",
            json={
                "enabled": True,
                "min_face_size": 72,
                "min_face_ratio": 0.04,
                "min_laplacian_var": 120,
                "max_pose_deviation": 0.45,
                "blur_eval_size": 128,
            },
        ).json()

        self.assertEqual(protected_payload["status"], "error")
        self.assertTrue(protected_payload["admin_required"])
        self.assertEqual(bad_login_payload["status"], "error")
        self.assertEqual(good_login_payload["status"], "success")
        self.assertEqual(update_payload["status"], "success")
        self.assertEqual(update_payload["face_quality_config"]["min_face_size"], 72)
        self.assertEqual(update_payload["face_quality_config"]["min_face_ratio"], 0.04)
        self.assertEqual(update_payload["face_quality_config"]["blur_eval_size"], 128)

    def test_non_auth_endpoints_require_login(self):
        self._setup_admin()
        self.client.post("/api/auth/logout")

        status_payload = self.client.get("/api/system/status").json()
        people_payload = self.client.get("/api/people").json()

        self.assertEqual(status_payload["status"], "error")
        self.assertTrue(status_payload["auth_required"])
        self.assertEqual(people_payload["status"], "error")
        self.assertTrue(people_payload["auth_required"])

    def test_text_search_prioritizes_named_person_and_supports_person_semantic_query(self):
        self._setup_admin()
        vector = np.ones(512, dtype=np.float32)
        alice_stage = app_module.db.add_face(
            video_id="v1",
            timestamp=0.0,
            image_path="/faces/alice_stage.jpg",
            full_image_path="/faces/alice_stage_full.jpg",
            source_url="https://example.com/alice-stage",
            embedding=vector,
            description="Alice on stage",
            semantic_text="conference stage",
        )
        alice_gala = app_module.db.add_face(
            video_id="v2",
            timestamp=1.0,
            image_path="/faces/alice_gala.jpg",
            full_image_path="/faces/alice_gala_full.jpg",
            source_url="https://example.com/alice-gala",
            embedding=vector,
            description="Alice at gala",
            semantic_text="red carpet gala",
        )
        bob_face = app_module.db.add_face(
            video_id="v3",
            timestamp=2.0,
            image_path="/faces/bob.jpg",
            full_image_path="/faces/bob_full.jpg",
            source_url="https://example.com/bob",
            embedding=vector,
            description="Bob launch",
            semantic_text="product launch",
        )
        app_module.db.update_person_ids(
            [(1, alice_stage), (1, alice_gala), (2, bob_face)]
        )
        app_module.db.rename_person(1, "Alice")
        app_module.db.rename_person(2, "Bob")

        name_payload = self.client.get("/api/search/text?q=Alice").json()
        with mock.patch.object(
            app_module.ai_engine,
            "rank_texts_by_similarity",
            return_value=[(0, 0.91), (1, 0.2)],
        ) as rank_mock:
            semantic_payload = self.client.get("/api/search/text?q=Alice%20gala").json()

        self.assertEqual(name_payload["status"], "success")
        self.assertEqual([item["person_id"] for item in name_payload["results"]], [1, 1])
        self.assertEqual(name_payload["results"][0]["metric"], "person_name")
        self.assertEqual(name_payload["results"][0]["person_name"], "Alice")
        self.assertEqual(semantic_payload["status"], "success")
        self.assertEqual([item["id"] for item in semantic_payload["results"]], [alice_gala, alice_stage])
        self.assertTrue(all(item["person_id"] == 1 for item in semantic_payload["results"]))
        self.assertEqual(semantic_payload["results"][0]["metric"], "person_semantic")
        rank_mock.assert_called_once()

    def test_text_search_without_person_name_uses_existing_semantic_flow(self):
        self._setup_admin()
        vector = np.ones(512, dtype=np.float32)
        first = app_module.db.add_face(
            video_id="v1",
            timestamp=0.0,
            image_path="/faces/first.jpg",
            full_image_path="/faces/first_full.jpg",
            source_url="https://example.com/first",
            embedding=vector,
            description="quiet scene",
            semantic_text="quiet scene",
        )
        second = app_module.db.add_face(
            video_id="v2",
            timestamp=1.0,
            image_path="/faces/second.jpg",
            full_image_path="/faces/second_full.jpg",
            source_url="https://example.com/second",
            embedding=vector,
            description="launch event",
            semantic_text="launch event",
        )
        app_module.db.update_person_ids([(1, first), (2, second)])
        app_module.db.rename_person(1, "Alice")

        with mock.patch.object(
            app_module.ai_engine,
            "rank_texts_by_similarity",
            return_value=[(1, 0.9), (0, 0.1)],
        ) as rank_mock:
            payload = self.client.get("/api/search/text?q=launch").json()

        self.assertEqual(payload["status"], "success")
        self.assertEqual([item["id"] for item in payload["results"]], [second])
        self.assertEqual(payload["results"][0]["metric"], "semantic")
        rank_mock.assert_called_once()

    def test_admin_can_manage_users_and_disabled_user_cannot_login(self):
        self._setup_admin()
        user = self._create_user()

        password_payload = self.client.post(
            f"/api/users/{user['id']}/password",
            json={"password": "changed123"},
        ).json()
        role_payload = self.client.patch(
            f"/api/users/{user['id']}",
            json={"role": "admin"},
        ).json()
        disabled_payload = self.client.patch(
            f"/api/users/{user['id']}",
            json={"enabled": False},
        ).json()
        login_payload = self.client.post(
            "/api/auth/login",
            json={"username": "operator", "password": "changed123"},
        ).json()

        self.assertEqual(password_payload["status"], "success")
        self.assertEqual(role_payload["status"], "success")
        self.assertEqual(role_payload["user"]["role"], "admin")
        self.assertEqual(disabled_payload["status"], "success")
        self.assertFalse(disabled_payload["user"]["enabled"])
        self.assertEqual(login_payload["status"], "error")

    def test_cannot_disable_or_demote_last_enabled_admin(self):
        self._setup_admin()
        admin = self.client.get("/api/auth/status").json()["user"]

        disable_payload = self.client.patch(
            f"/api/users/{admin['id']}",
            json={"enabled": False},
        ).json()
        demote_payload = self.client.patch(
            f"/api/users/{admin['id']}",
            json={"role": "user"},
        ).json()

        self.assertEqual(disable_payload["status"], "error")
        self.assertEqual(demote_payload["status"], "error")

    def test_regular_user_can_collect_import_and_add_sources_but_not_admin_actions(self):
        self._setup_admin()
        self._create_user()
        self.client.post("/api/auth/logout")
        self._login_user()

        with mock.patch.object(
            app_module.collector,
            "download",
            return_value={
                "platform": "generic",
                "external_id": "video-1",
                "title": "video",
                "url": "https://example.com/video",
                "path": "",
            },
        ), mock.patch.object(
            app_module.db,
            "upsert_content",
            return_value={"id": 1},
        ), mock.patch.object(
            app_module.db,
            "content_has_faces",
            return_value=True,
        ):
            collect_payload = self.client.post(
                "/api/collect?url=https%3A%2F%2Fexample.com%2Fvideo"
            ).json()

            import_payload = self.client.post(
                "/api/import/post",
                json={
                    "external_id": "post-1",
                    "post_text": "hello",
                    "image_urls": ["https://example.com/a.jpg"],
                },
            ).json()

        fake_adapter = mock.Mock()
        fake_adapter.adapter_id = "weibo_user"
        fake_adapter.default_limit = 5
        fake_adapter.normalize_source.return_value = "https://weibo.cn/123456"
        source_record = {"id": 1, "platform": "weibo", "source_url": "https://weibo.cn/123456"}
        with mock.patch.object(
            app_module.source_adapter_registry,
            "select_adapter",
            return_value=fake_adapter,
        ), mock.patch.object(
            app_module.db,
            "register_collection_source",
            return_value=source_record,
        ), mock.patch.object(
            app_module.db,
            "get_collection_source",
            return_value=source_record,
        ), mock.patch.object(
            app_module,
            "sync_source_task",
            return_value={"status": "success", "stats": {}},
        ):
            source_payload = self.client.post(
                "/api/collect/source",
                json={
                    "source_url": "https://weibo.cn/123456",
                    "platform": "weibo",
                    "source_type": "weibo_user",
                },
            ).json()

        delete_payload = self.client.post(
            "/api/source/delete",
            json={"source_id": 1},
        ).json()
        config_payload = self.client.post(
            "/api/system/config",
            json={"search": {"text_threshold": 0.2}},
        ).json()
        users_payload = self.client.get("/api/users").json()

        self.assertEqual(collect_payload["status"], "success")
        self.assertTrue(collect_payload["duplicate"])
        self.assertEqual(import_payload["status"], "success")
        self.assertTrue(import_payload["duplicate"])
        self.assertEqual(source_payload["status"], "success")
        self.assertEqual(delete_payload["status"], "error")
        self.assertTrue(delete_payload["admin_required"])
        self.assertEqual(config_payload["status"], "error")
        self.assertTrue(config_payload["admin_required"])
        self.assertEqual(users_payload["status"], "error")
        self.assertTrue(users_payload["admin_required"])

    def test_import_post_accepts_weibo_url_without_manual_image_urls(self):
        self._setup_admin()
        extracted = {
            "platform": "weibo",
            "external_id": "5241373692531045",
            "title": "weibo post",
            "post_text": "hello from weibo",
            "image_urls": ["https://wx1.sinaimg.cn/large/a.jpg"],
            "source_url": "https://weibo.cn/comment/5241373692531045",
        }

        with mock.patch.object(
            app_module.collector,
            "extract_post_metadata",
            return_value=extracted,
        ) as extract_mock, mock.patch.object(
            app_module.collector,
            "download_post_images",
            return_value=[
                {
                    "local_path": str(app_config.content_dir / "a.jpg"),
                    "web_path": "/content/a.jpg",
                    "source_url": extracted["image_urls"][0],
                }
            ],
        ), mock.patch.object(
            app_module.db,
            "upsert_content",
            return_value={"id": 1},
        ), mock.patch.object(
            app_module.db,
            "content_has_faces",
            return_value=False,
        ), mock.patch.object(app_module, "process_post_task") as process_mock:
            payload = self.client.post(
                "/api/import/post",
                json={"url": "https://weibo.cn/comment/5241373692531045"},
            ).json()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["images"], 1)
        extract_mock.assert_called_once_with("https://weibo.cn/comment/5241373692531045")
        process_mock.assert_called_once()

    def test_collect_weibo_post_routes_to_post_import_without_video_download(self):
        self._setup_admin()
        extracted = {
            "platform": "weibo",
            "external_id": "5241373692531045",
            "title": "weibo post",
            "post_text": "hello from weibo",
            "image_urls": ["https://wx1.sinaimg.cn/large/a.jpg"],
            "source_url": "https://weibo.cn/comment/5241373692531045",
        }

        with mock.patch.object(
            app_module.collector,
            "extract_post_metadata",
            return_value=extracted,
        ), mock.patch.object(
            app_module.collector,
            "download",
            side_effect=AssertionError("video download should not run"),
        ) as download_mock, mock.patch.object(
            app_module.collector,
            "download_post_images",
            return_value=[
                {
                    "local_path": str(app_config.content_dir / "a.jpg"),
                    "web_path": "/content/a.jpg",
                    "source_url": extracted["image_urls"][0],
                }
            ],
        ), mock.patch.object(
            app_module.db,
            "upsert_content",
            return_value={"id": 1},
        ), mock.patch.object(
            app_module.db,
            "content_has_faces",
            return_value=False,
        ), mock.patch.object(app_module, "process_post_task") as process_mock:
            payload = self.client.post(
                "/api/collect?url=https%3A%2F%2Fweibo.cn%2Fcomment%2F5241373692531045"
            ).json()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["images"], 1)
        download_mock.assert_not_called()
        process_mock.assert_called_once()

    def test_import_post_reports_weibo_cookie_auth_error(self):
        self._setup_admin()
        error = app_module.WeiboAuthenticationError(
            "微博 Cookie 无效或已过期，请刷新 storage/weibo_cookies.txt 后重试。"
        )

        with mock.patch.object(
            app_module.collector,
            "extract_post_metadata",
            side_effect=error,
        ):
            payload = self.client.post(
                "/api/import/post",
                json={"url": "https://weibo.com/1793285524/5241373692531045"},
            ).json()

        self.assertEqual(payload["status"], "error")
        self.assertIn("微博 Cookie 无效或已过期", payload["msg"])
        self.assertNotIn("yt-dlp", payload["msg"])

    def test_collect_weibo_cookie_auth_error_does_not_fall_back_to_video_download(self):
        self._setup_admin()
        error = app_module.WeiboAuthenticationError(
            "微博 Cookie 无效或已过期，请刷新 storage/weibo_cookies.txt 后重试。"
        )

        with mock.patch.object(
            app_module.collector,
            "extract_post_metadata",
            side_effect=error,
        ), mock.patch.object(
            app_module.collector,
            "download",
            side_effect=AssertionError("video download should not run"),
        ) as download_mock:
            payload = self.client.post(
                "/api/collect?url=https%3A%2F%2Fweibo.com%2F1793285524%2F5241373692531045"
            ).json()

        self.assertEqual(payload["status"], "error")
        self.assertIn("微博 Cookie 无效或已过期", payload["msg"])
        download_mock.assert_not_called()

    def test_source_adapter_config_api_requires_admin_and_persists_config(self):
        config_payload = {
            "adapter_id": "custom_site",
            "display_name": "Custom Site",
            "platform": "custom_site",
            "enabled": True,
            "source_types": ["user", "channel"],
            "url_patterns": ["https://example.com/*"],
            "default_limit": 6,
            "settings": {},
        }
        unauthenticated_payload = self.client.post(
            "/api/source-adapters/config",
            files={
                "file": (
                    "custom_site.json",
                    json.dumps(config_payload),
                    "application/json",
                )
            },
        ).json()

        self._setup_admin()
        upload_payload = self.client.post(
            "/api/source-adapters/config",
            files={
                "file": (
                    "custom_site.json",
                    json.dumps(config_payload),
                    "application/json",
                )
            },
        ).json()
        config_file_exists_after_upload = (
            app_config.source_adapter_config_dir / "custom_site.json"
        ).exists()
        listed_payload = self.client.get("/api/source-adapters").json()
        disabled_payload = self.client.post(
            "/api/source-adapters/enable",
            json={"adapter_id": "custom_site", "enabled": False},
        ).json()
        deleted_payload = self.client.delete("/api/source-adapters/custom_site").json()

        self.assertEqual(unauthenticated_payload["status"], "error")
        self.assertTrue(unauthenticated_payload["admin_required"])
        self.assertEqual(upload_payload["status"], "success")
        self.assertEqual(upload_payload["adapter"]["adapter_id"], "custom_site")
        self.assertTrue(config_file_exists_after_upload)
        self.assertIn("custom_site", [item["adapter_id"] for item in listed_payload["adapters"]])
        self.assertEqual(disabled_payload["status"], "success")
        self.assertFalse(disabled_payload["adapter"]["enabled"])
        self.assertEqual(deleted_payload["status"], "success")
        self.assertFalse((app_config.source_adapter_config_dir / "custom_site.json").exists())

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
                    "min_face_ratio": 0.025,
                    "min_laplacian_var": 100,
                    "max_pose_deviation": 0.4,
                    "blur_eval_size": 128,
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
        invalid_face_ratio_payload = self.client.post(
            "/api/system/config",
            json={"face_quality": {"min_face_size": 64, "min_face_ratio": 0.8}},
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
        self.assertEqual(
            success_payload["runtime_config"]["face_quality"]["min_face_ratio"],
            0.025,
        )
        self.assertEqual(
            success_payload["runtime_config"]["face_quality"]["blur_eval_size"],
            128,
        )

        self.assertEqual(invalid_payload["status"], "error")
        self.assertEqual(invalid_hotwords_payload["status"], "error")
        self.assertEqual(invalid_face_ratio_payload["status"], "error")
        self.assertEqual(status_payload["runtime_config"]["search"]["text_threshold"], 0.25)
        self.assertEqual(saved_config["search"]["text_threshold"], 0.25)
        self.assertEqual(saved_config["search"]["semantic_model_prompt_name"], "query")
        self.assertEqual(saved_config["search"]["semantic_corpus_style"], "structured_zh")
        self.assertEqual(saved_config["vision"]["vlm_model_id"], "microsoft/Florence-2-large-ft")
        self.assertEqual(saved_config["vision"]["caption_style"], "retrieval_keywords")
        self.assertEqual(saved_config["transcription"]["hotwords"], ["发布会", "路演", "采访"])
        self.assertEqual(saved_config["collection"]["weibo_cookie_enabled"], False)
        self.assertIn("platforms", saved_config["collection"])
        self.assertFalse(saved_config["collection"]["platforms"]["weibo"]["auth_enabled"])
        self.assertEqual(saved_config["collection"]["platforms"]["weibo"]["sync_limit"], 9)
        self.assertEqual(saved_config["collection"]["platforms"]["generic"]["sync_limit"], 15)

    def test_platform_collection_config_update_persists_and_validates(self):
        self._setup_admin()

        payload = self.client.post(
            "/api/system/config",
            json={
                "collection": {
                    "platforms": {
                        "weibo": {
                            "enabled": True,
                            "auth_enabled": False,
                            "sync_limit": 7,
                            "sync_interval_minutes": 30,
                            "timeout_seconds": 25,
                            "retry_count": 2,
                        },
                        "bilibili": {
                            "enabled": True,
                            "auth_enabled": False,
                            "sync_limit": 11,
                            "sync_interval_minutes": 60,
                            "timeout_seconds": 35,
                            "retry_count": 6,
                            "impersonate": "chrome",
                            "referer": "https://www.bilibili.com/",
                        },
                        "youtube": {
                            "enabled": False,
                            "auth_enabled": True,
                            "sync_limit": 5,
                            "sync_interval_minutes": 120,
                            "timeout_seconds": 40,
                            "retry_count": 3,
                        },
                        "x": {
                            "enabled": True,
                            "auth_enabled": True,
                            "sync_limit": 4,
                            "sync_interval_minutes": 15,
                            "timeout_seconds": 45,
                            "retry_count": 4,
                        },
                        "generic": {
                            "enabled": True,
                            "sync_limit": 13,
                            "sync_interval_minutes": 0,
                            "timeout_seconds": 20,
                            "retry_count": 3,
                        },
                    }
                }
            },
        ).json()
        invalid_limit = self.client.post(
            "/api/system/config",
            json={"collection": {"platforms": {"weibo": {"sync_limit": 0}}}},
        ).json()
        invalid_interval = self.client.post(
            "/api/system/config",
            json={
                "collection": {
                    "platforms": {"weibo": {"sync_interval_minutes": -1}}
                }
            },
        ).json()
        invalid_timeout = self.client.post(
            "/api/system/config",
            json={"collection": {"platforms": {"x": {"timeout_seconds": 1}}}},
        ).json()
        invalid_retry = self.client.post(
            "/api/system/config",
            json={"collection": {"platforms": {"youtube": {"retry_count": 99}}}},
        ).json()
        status_payload = self.client.get("/api/system/status").json()
        saved_config = json.loads(app_config.system_config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "success")
        self.assertEqual(
            payload["runtime_config"]["collection"]["platforms"]["weibo"]["sync_limit"],
            7,
        )
        self.assertEqual(
            payload["runtime_config"]["collection"]["platforms"]["bilibili"]["retry_count"],
            6,
        )
        self.assertFalse(
            status_payload["runtime_config"]["collection"]["platforms"]["youtube"]["enabled"]
        )
        self.assertEqual(saved_config["collection"]["platforms"]["x"]["sync_limit"], 4)
        self.assertEqual(saved_config["collection"]["source_sync_limit"], 13)
        self.assertEqual(invalid_limit["status"], "error")
        self.assertEqual(invalid_interval["status"], "error")
        self.assertEqual(invalid_timeout["status"], "error")
        self.assertEqual(invalid_retry["status"], "error")

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
