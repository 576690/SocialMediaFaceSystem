import gc
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import core.database as database_module
from core.config import DEFAULT_CONFIG, app_config
from core.database import DatabaseManager


class DatabaseSnapshotTests(unittest.TestCase):
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

    def _make_file(self, path, content=b"x"):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def test_cluster_snapshot_save_and_restore(self):
        db = DatabaseManager()
        try:
            vector = np.zeros(512, dtype=np.float32)
            face_1 = db.add_face(
                video_id="v1",
                timestamp=0.0,
                image_path="/faces/1.jpg",
                full_image_path="/faces/1_full.jpg",
                source_url="https://example.com/1",
                embedding=vector,
                description="face 1",
            )
            face_2 = db.add_face(
                video_id="v2",
                timestamp=1.0,
                image_path="/faces/2.jpg",
                full_image_path="/faces/2_full.jpg",
                source_url="https://example.com/2",
                embedding=vector,
                description="face 2",
            )
            db.update_person_ids([(1, face_1), (2, face_2)])
            db.rename_person(1, "Alice")
            db.rename_person(2, "Bob")

            self.assertFalse(db.has_cluster_snapshot())
            db.save_cluster_snapshot(
                {
                    "algorithm": "dbscan",
                    "metric": "cosine",
                    "eps": 0.4,
                    "min_samples": 2,
                }
            )
            self.assertTrue(db.has_cluster_snapshot())

            db.replace_all_person_ids([(7, face_1), (-1, face_2)], {7: "Carol"})
            restored = db.restore_cluster_snapshot()

            self.assertEqual(restored["restored_faces"], 2)
            self.assertEqual(restored["restored_people"], 2)
            self.assertEqual(restored["cluster_config"]["algorithm"], "dbscan")

            faces = db.get_all_faces()
            self.assertEqual([face["person_id"] for face in faces], [1, 2])
            self.assertEqual(db.get_person_name_map(), {1: "Alice", 2: "Bob"})
        finally:
            del db
            gc.collect()

    def test_clustered_people_cover_uses_highest_quality_face(self):
        db = DatabaseManager()
        try:
            vector = np.ones(512, dtype=np.float32)
            low_face = db.add_face(
                video_id="v1",
                timestamp=0.0,
                image_path="/faces/a_low.jpg",
                full_image_path="/faces/a_low_full.jpg",
                source_url="https://example.com/low",
                embedding=vector,
                description="low quality",
                face_metrics={
                    "min_face_size": 40,
                    "area": 1600,
                    "face_ratio": 0.03,
                    "laplacian_var": 20,
                    "pose_deviation": 0.7,
                },
            )
            high_face = db.add_face(
                video_id="v2",
                timestamp=1.0,
                image_path="/faces/z_high.jpg",
                full_image_path="/faces/z_high_full.jpg",
                source_url="https://example.com/high",
                embedding=vector,
                description="high quality",
                face_metrics={
                    "min_face_size": 120,
                    "area": 14400,
                    "face_ratio": 0.08,
                    "laplacian_var": 240,
                    "pose_deviation": 0.1,
                },
            )
            db.update_person_ids([(1, low_face), (1, high_face)])

            people = db.get_clustered_people()

            self.assertEqual(people[0]["cover_image"], "/faces/z_high.jpg")
        finally:
            del db
            gc.collect()

    def test_backfill_missing_face_quality_uses_stored_crop_and_handles_missing_file(self):
        db = DatabaseManager()
        try:
            vector = np.ones(512, dtype=np.float32)
            image_path = app_config.faces_dir / "sharp_face.jpg"
            sharp = np.zeros((80, 80, 3), dtype=np.uint8)
            sharp[::2, :] = 255
            cv2.imwrite(str(image_path), sharp)

            good_face = db.add_face(
                video_id="v1",
                timestamp=0.0,
                image_path="/faces/sharp_face.jpg",
                full_image_path="/faces/sharp_full.jpg",
                source_url="https://example.com/good",
                embedding=vector,
                description="sharp",
            )
            missing_face = db.add_face(
                video_id="v2",
                timestamp=1.0,
                image_path="/faces/missing_face.jpg",
                full_image_path="/faces/missing_full.jpg",
                source_url="https://example.com/missing",
                embedding=vector,
                description="missing",
            )
            db.update_person_ids([(1, good_face), (1, missing_face)])
            with db._connect() as conn:
                conn.execute(
                    """
                    UPDATE faces
                    SET face_min_side = NULL,
                        face_area = NULL,
                        face_ratio = NULL,
                        laplacian_var = NULL,
                        pose_deviation = NULL,
                        quality_score = NULL
                    """
                )

            people = db.get_clustered_people()
            faces = {item["id"]: item for item in db.get_all_faces()}

            self.assertEqual(people[0]["cover_image"], "/faces/sharp_face.jpg")
            self.assertGreater(faces[good_face]["quality_score"], faces[missing_face]["quality_score"])
            self.assertIsNotNone(faces[missing_face]["quality_score"])
        finally:
            del db
            gc.collect()

    def test_collection_source_metadata_roundtrip(self):
        db = DatabaseManager()
        try:
            source = db.register_collection_source(
                platform="weibo",
                source_type="weibo_user",
                source_url="https://weibo.cn/u/123456",
                title="测试用户",
                metadata={
                    "keywords": ["发布会", "上海"],
                    "limit": 20,
                },
            )
            self.assertEqual(source["keywords"], ["发布会", "上海"])

            db.mark_source_synced(
                source["id"],
                metadata={
                    "last_sync_stats": {
                        "fetched_count": 5,
                        "matched_count": 2,
                    },
                    "user_id": "123456",
                },
            )

            sources = db.list_collection_sources()
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["keywords"], ["发布会", "上海"])
            self.assertEqual(sources[0]["last_sync_stats"]["matched_count"], 2)
            self.assertEqual(sources[0]["user_id"], "123456")
        finally:
            del db
            gc.collect()

    def test_delete_source_without_data_keeps_contents_and_faces(self):
        db = DatabaseManager()
        try:
            source = db.register_collection_source(
                platform="bilibili",
                source_type="channel",
                source_url="https://space.bilibili.com/1",
                title="demo",
            )
            vector = np.zeros(512, dtype=np.float32)
            content = db.upsert_content(
                platform="bilibili",
                external_id="BV1",
                content_type="video",
                source_url="https://www.bilibili.com/video/BV1",
                metadata={"source_sync_url": source["source_url"]},
                collection_source_id=source["id"],
            )
            db.add_face(
                video_id="BV1",
                timestamp=0.0,
                image_path="/faces/keep.jpg",
                full_image_path="/faces/keep_full.jpg",
                source_url="https://www.bilibili.com/video/BV1",
                embedding=vector,
                description="keep",
                content_id=content["id"],
            )

            deleted_source = db.delete_collection_source(source["id"])
            self.assertEqual(deleted_source, 1)
            self.assertIsNotNone(db.get_content_by_id(content["id"]))
            self.assertEqual(len(db.get_all_faces()), 1)
            self.assertEqual(db.index.ntotal, 1)
            self.assertEqual(db.list_collection_sources(), [])
        finally:
            del db
            gc.collect()

    def test_delete_source_with_data_removes_records_files_and_index(self):
        db = DatabaseManager()
        try:
            source = db.register_collection_source(
                platform="bilibili",
                source_type="channel",
                source_url="https://space.bilibili.com/5500601",
                title="channel",
            )
            vector = np.zeros(512, dtype=np.float32)

            explicit_video = self._make_file(app_config.videos_dir / "explicit.mp4")
            explicit_subtitle = self._make_file(app_config.videos_dir / "explicit.srt")
            explicit_asr = self._make_file(app_config.asr_dir / "explicit.srt")
            legacy_video = self._make_file(app_config.videos_dir / "legacy.mp4")
            unrelated_video = self._make_file(app_config.videos_dir / "manual.mp4")
            explicit_face = self._make_file(app_config.faces_dir / "explicit_face.jpg")
            explicit_full = self._make_file(app_config.faces_dir / "explicit_full.jpg")
            legacy_face = self._make_file(app_config.faces_dir / "legacy_face.jpg")
            legacy_full = self._make_file(app_config.faces_dir / "legacy_full.jpg")
            unrelated_face = self._make_file(app_config.faces_dir / "manual_face.jpg")
            unrelated_full = self._make_file(app_config.faces_dir / "manual_full.jpg")

            explicit = db.upsert_content(
                platform="bilibili",
                external_id="BV-explicit",
                content_type="video",
                source_url="https://www.bilibili.com/video/BV-explicit",
                local_path=str(explicit_video),
                subtitle_path=str(explicit_subtitle),
                asr_path=str(explicit_asr),
                metadata={"source_sync_url": source["source_url"]},
                collection_source_id=source["id"],
            )
            legacy = db.upsert_content(
                platform="bilibili",
                external_id="BV-legacy",
                content_type="video",
                source_url="https://www.bilibili.com/video/BV-legacy",
                local_path=str(legacy_video),
                metadata={"source_sync_url": source["source_url"]},
            )
            unrelated = db.upsert_content(
                platform="bilibili",
                external_id="BV-manual",
                content_type="video",
                source_url="https://www.bilibili.com/video/BV-manual",
                local_path=str(unrelated_video),
                metadata={"download_url": "manual"},
            )

            face_explicit = db.add_face(
                video_id="BV-explicit",
                timestamp=0.0,
                image_path="/faces/explicit_face.jpg",
                full_image_path="/faces/explicit_full.jpg",
                source_url="https://www.bilibili.com/video/BV-explicit",
                embedding=vector,
                description="explicit",
                content_id=explicit["id"],
            )
            face_legacy = db.add_face(
                video_id="BV-legacy",
                timestamp=1.0,
                image_path="/faces/legacy_face.jpg",
                full_image_path="/faces/legacy_full.jpg",
                source_url="https://www.bilibili.com/video/BV-legacy",
                embedding=vector,
                description="legacy",
                content_id=legacy["id"],
            )
            face_unrelated = db.add_face(
                video_id="BV-manual",
                timestamp=2.0,
                image_path="/faces/manual_face.jpg",
                full_image_path="/faces/manual_full.jpg",
                source_url="https://www.bilibili.com/video/BV-manual",
                embedding=vector,
                description="manual",
                content_id=unrelated["id"],
            )
            db.update_person_ids([(1, face_explicit), (1, face_legacy), (2, face_unrelated)])
            db.rename_person(1, "Source Person")
            db.rename_person(2, "Manual Person")
            db.save_cluster_snapshot({"algorithm": "dbscan"})

            result = db.delete_source_with_data(source["id"])

            self.assertEqual(result["deleted_source"], 1)
            self.assertEqual(result["deleted_contents"], 2)
            self.assertEqual(result["deleted_faces"], 2)
            self.assertEqual(result["deleted_files"], 8)
            self.assertTrue(result["cleared_cluster_snapshot"])
            self.assertEqual(result["unresolved_legacy_items"], 0)
            self.assertIsNone(db.get_content_by_id(explicit["id"]))
            self.assertIsNone(db.get_content_by_id(legacy["id"]))
            self.assertIsNotNone(db.get_content_by_id(unrelated["id"]))
            self.assertEqual([face["id"] for face in db.get_all_faces()], [face_unrelated])
            self.assertEqual(db.get_person_name_map(), {2: "Manual Person"})
            self.assertFalse(db.has_cluster_snapshot())
            self.assertEqual(db.index.ntotal, 1)
            self.assertFalse(explicit_video.exists())
            self.assertFalse(explicit_subtitle.exists())
            self.assertFalse(explicit_asr.exists())
            self.assertFalse(legacy_video.exists())
            self.assertFalse(explicit_face.exists())
            self.assertFalse(explicit_full.exists())
            self.assertFalse(legacy_face.exists())
            self.assertFalse(legacy_full.exists())
            self.assertTrue(unrelated_video.exists())
            self.assertTrue(unrelated_face.exists())
            self.assertTrue(unrelated_full.exists())
        finally:
            del db
            gc.collect()

    def test_delete_source_with_data_keeps_unresolved_legacy_weibo_records(self):
        db = DatabaseManager()
        try:
            source = db.register_collection_source(
                platform="weibo",
                source_type="weibo_user",
                source_url="https://weibo.cn/123456",
                title="weibo-user",
                metadata={"user_id": "123456"},
            )
            db.upsert_content(
                platform="weibo",
                external_id="old-post",
                content_type="post",
                source_url="https://weibo.cn/comment/old-post",
                metadata={"user_id": "123456"},
            )

            result = db.delete_source_with_data(source["id"])

            self.assertEqual(result["deleted_source"], 1)
            self.assertEqual(result["deleted_contents"], 0)
            self.assertEqual(result["deleted_faces"], 0)
            self.assertEqual(result["unresolved_legacy_items"], 1)
            self.assertIsNotNone(db.get_content_by_identity("weibo", "old-post"))
        finally:
            del db
            gc.collect()

    def test_upsert_content_with_status_reports_existing_identity(self):
        db = DatabaseManager()
        try:
            first, first_created = db.upsert_content_with_status(
                platform="weibo",
                external_id="5241373692531045",
                content_type="post",
                title="first",
                source_url="https://weibo.cn/comment/5241373692531045",
            )
            second, second_created = db.upsert_content_with_status(
                platform="weibo",
                external_id="5241373692531045",
                content_type="post",
                title="second",
                source_url="https://weibo.cn/comment/5241373692531045",
            )

            self.assertTrue(first_created)
            self.assertFalse(second_created)
            self.assertEqual(second["id"], first["id"])
            self.assertEqual(second["title"], "second")
        finally:
            del db
            gc.collect()

    def test_reset_database_state_and_config_defaults(self):
        db = DatabaseManager()
        try:
            vector = np.zeros(512, dtype=np.float32)
            cookie_path = app_config.weibo_cookie_path
            youtube_cookie_path = app_config.storage_dir / "www.youtube.com_cookies.txt"
            cookie_path.write_text("cookie", encoding="utf-8")
            youtube_cookie_path.write_text("cookie", encoding="utf-8")

            db.upsert_content(
                platform="bilibili",
                external_id="BV-reset",
                content_type="video",
                source_url="https://www.bilibili.com/video/BV-reset",
                metadata={"download_url": "demo"},
            )
            db.add_face(
                video_id="BV-reset",
                timestamp=0.0,
                image_path="/faces/reset.jpg",
                full_image_path="/faces/reset_full.jpg",
                source_url="https://www.bilibili.com/video/BV-reset",
                embedding=vector,
                description="reset",
            )
            db.save_cluster_snapshot({"algorithm": "optics"})
            app_config.update_cluster_defaults(
                algorithm="optics",
                metric="euclidean",
                eps=0.7,
                min_samples=3,
            )

            removed = db.reset_database_state()
            app_config.reset_to_defaults()

            self.assertGreaterEqual(removed, 1)
            self.assertEqual(db.get_all_faces(), [])
            self.assertEqual(db.list_collection_sources(), [])
            self.assertFalse(db.has_cluster_snapshot())
            self.assertEqual(db.index.ntotal, 0)
            self.assertEqual(app_config.cluster_defaults(), DEFAULT_CONFIG["clustering"])
            self.assertEqual(
                json.loads(app_config.system_config_path.read_text(encoding="utf-8")),
                DEFAULT_CONFIG,
            )
            self.assertTrue(cookie_path.exists())
            self.assertTrue(youtube_cookie_path.exists())
        finally:
            del db
            gc.collect()


if __name__ == "__main__":
    unittest.main()
