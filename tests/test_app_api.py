import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
import core.database as database_module
from core.config import DEFAULT_CONFIG, app_config


class FaceQualityApiTests(unittest.TestCase):
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
            "data": json.loads(json.dumps(app_config.data)),
        }
        self.temp_root = Path(tempfile.mkdtemp())
        self._patch_app_config(self.temp_root)
        database_module.DB_PATH = str(self.temp_root / "metadata.db")
        database_module.INDEX_PATH = str(self.temp_root / "face_index.faiss")
        app_module._rebuild_runtime_state()
        self.client = TestClient(app_module.app)

    def tearDown(self):
        database_module.DB_PATH = self.original_db_path
        database_module.INDEX_PATH = self.original_index_path

        app_config.storage_dir = self.original_app_config["storage_dir"]
        app_config.videos_dir = self.original_app_config["videos_dir"]
        app_config.faces_dir = self.original_app_config["faces_dir"]
        app_config.content_dir = self.original_app_config["content_dir"]
        app_config.asr_dir = self.original_app_config["asr_dir"]
        app_config.test_artifacts_dir = self.original_app_config["test_artifacts_dir"]
        app_config.system_config_path = self.original_app_config["system_config_path"]
        app_config.weibo_cookie_path = self.original_app_config["weibo_cookie_path"]
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
        app_config.data = json.loads(json.dumps(DEFAULT_CONFIG))
        app_config.ensure_dirs()
        app_config.save()

    def test_system_status_exposes_default_face_quality(self):
        response = self.client.get("/api/system/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["face_quality_config"], DEFAULT_CONFIG["face_quality"])

    def test_update_face_quality_persists_config(self):
        response = self.client.post(
            "/api/system/face-quality",
            json={
                "enabled": True,
                "min_face_size": 72,
                "min_laplacian_var": 120,
                "max_pose_deviation": 0.45,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["face_quality_config"]["min_face_size"], 72)

        status_payload = self.client.get("/api/system/status").json()
        self.assertEqual(status_payload["face_quality_config"]["min_laplacian_var"], 120.0)
        self.assertEqual(
            json.loads(app_config.system_config_path.read_text(encoding="utf-8"))["face_quality"]["max_pose_deviation"],
            0.45,
        )

    def test_update_face_quality_rejects_invalid_thresholds(self):
        response = self.client.post(
            "/api/system/face-quality",
            json={
                "enabled": True,
                "min_face_size": 10,
                "min_laplacian_var": 80,
                "max_pose_deviation": 0.35,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(app_config.face_quality_config(), DEFAULT_CONFIG["face_quality"])


if __name__ == "__main__":
    unittest.main()
