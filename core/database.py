import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from core.config import app_config

DB_PATH = str(app_config.storage_dir / "metadata.db")
INDEX_PATH = str(app_config.storage_dir / "face_index.faiss")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
PASSWORD_ITERATIONS = 200_000
USER_ROLES = {"user", "admin"}


class DatabaseManager:
    def __init__(self):
        app_config.ensure_dirs()
        self.dimension = 512
        self.init_db()
        self.init_index()

    def _connect(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _normalize_embedding(self, embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    def _new_index(self):
        return faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))

    def _get_columns(self, table_name):
        with self._connect() as conn:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _ensure_column(self, table_name, column_name, definition):
        columns = self._get_columns(table_name)
        if column_name in columns:
            return
        with self._connect() as conn:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )

    def _ensure_index(self, index_name, statement):
        with self._connect() as conn:
            conn.execute(statement)

    def _parse_json(self, value, default):
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    timestamp REAL,
                    image_path TEXT,
                    full_image_path TEXT,
                    source_url TEXT,
                    description TEXT,
                    embedding BLOB,
                    person_id INTEGER DEFAULT -1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS face_fts USING fts5(
                    id, description
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id INTEGER PRIMARY KEY,
                    name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    title TEXT,
                    source_url TEXT,
                    local_path TEXT,
                    subtitle_path TEXT,
                    asr_path TEXT,
                    post_text TEXT,
                    metadata_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_processed_at DATETIME,
                    UNIQUE(platform, external_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    enabled INTEGER DEFAULT 1,
                    last_synced_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_snapshots (
                    slot INTEGER PRIMARY KEY,
                    face_assignments_json TEXT NOT NULL,
                    people_json TEXT NOT NULL,
                    cluster_config_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    iterations INTEGER NOT NULL DEFAULT 200000,
                    role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_role_enabled
                ON users(role, enabled)
                """
            )

        for column_name, definition in (
            ("content_id", "INTEGER"),
            ("content_type", "TEXT DEFAULT 'video'"),
            ("platform", "TEXT"),
            ("external_id", "TEXT"),
            ("visual_text", "TEXT"),
            ("subtitle_text", "TEXT"),
            ("asr_text", "TEXT"),
            ("post_text", "TEXT"),
            ("semantic_text", "TEXT"),
            ("semantic_source", "TEXT"),
        ):
            self._ensure_column("faces", column_name, definition)

        self._ensure_column("collection_sources", "metadata_json", "TEXT")
        self._ensure_column("contents", "collection_source_id", "INTEGER")
        self._ensure_index(
            "idx_contents_collection_source_id",
            """
            CREATE INDEX IF NOT EXISTS idx_contents_collection_source_id
            ON contents(collection_source_id)
            """,
        )

    def _validate_username(self, username):
        cleaned = str(username or "").strip()
        if not USERNAME_RE.match(cleaned):
            raise ValueError(
                "Username must be 3-32 ASCII letters, numbers, underscores, dots, or hyphens."
            )
        return cleaned

    def _validate_role(self, role):
        cleaned = str(role or "").strip().lower()
        if cleaned not in USER_ROLES:
            raise ValueError("Role must be user or admin.")
        return cleaned

    def _hash_password(self, password):
        cleaned = str(password or "")
        if len(cleaned) < 6:
            raise ValueError("Password must be at least 6 characters.")
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            cleaned.encode("utf-8"),
            bytes.fromhex(salt),
            PASSWORD_ITERATIONS,
        ).hex()
        return password_hash, salt, PASSWORD_ITERATIONS

    def _serialize_user_row(self, row):
        if row is None:
            return None
        item = dict(row)
        item["enabled"] = bool(item.get("enabled"))
        item.pop("password_hash", None)
        item.pop("salt", None)
        item.pop("iterations", None)
        return item

    def users_initialized(self):
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return count > 0

    def setup_initial_admin(self, username, password):
        username = self._validate_username(username)
        password_hash, salt, iterations = self._hash_password(password)
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count:
                raise ValueError("Initial administrator is already set.")
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, iterations, role, enabled)
                VALUES (?, ?, ?, ?, 'admin', 1)
                """,
                (username, password_hash, salt, iterations),
            )
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return self._serialize_user_row(row)

    def verify_user_password(self, username, password):
        username = str(username or "").strip()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row is None or not int(row["enabled"] or 0):
            return None
        try:
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                str(password or "").encode("utf-8"),
                bytes.fromhex(row["salt"]),
                int(row["iterations"] or PASSWORD_ITERATIONS),
            ).hex()
        except (TypeError, ValueError):
            return None
        if not hmac.compare_digest(candidate, row["password_hash"]):
            return None
        return self._serialize_user_row(row)

    def get_user(self, user_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return self._serialize_user_row(row)

    def list_users(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM users
                ORDER BY role = 'admin' DESC, username COLLATE NOCASE ASC
                """
            ).fetchall()
        return [self._serialize_user_row(row) for row in rows]

    def create_user(self, username, password, role="user", enabled=True):
        username = self._validate_username(username)
        role = self._validate_role(role)
        password_hash, salt, iterations = self._hash_password(password)
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (username, password_hash, salt, iterations, role, enabled)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username, password_hash, salt, iterations, role, 1 if enabled else 0),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Username already exists.") from exc
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return self._serialize_user_row(row)

    def _enabled_admin_count(self, conn):
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND enabled = 1"
            ).fetchone()[0]
            or 0
        )

    def update_user(self, user_id, role=None, enabled=None):
        user_id = int(user_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise ValueError("User not found.")

            next_role = row["role"] if role is None else self._validate_role(role)
            next_enabled = int(row["enabled"] if enabled is None else bool(enabled))
            if row["role"] == "admin" and int(row["enabled"] or 0):
                would_remove_admin = next_role != "admin" or not next_enabled
                if would_remove_admin and self._enabled_admin_count(conn) <= 1:
                    raise ValueError("Cannot disable or demote the last enabled administrator.")

            conn.execute(
                """
                UPDATE users
                SET role = ?, enabled = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (next_role, next_enabled, user_id),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._serialize_user_row(row)

    def set_user_password(self, user_id, password):
        password_hash, salt, iterations = self._hash_password(password)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if row is None:
                raise ValueError("User not found.")
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, salt = ?, iterations = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (password_hash, salt, iterations, int(user_id)),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return self._serialize_user_row(row)

    def init_index(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = self._new_index()

        if not isinstance(self.index, faiss.IndexIDMap):
            self.rebuild_index()
            return

        self._ensure_index_synced()

    def _ensure_index_synced(self):
        with self._connect() as conn:
            face_count = conn.execute(
                "SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL"
            ).fetchone()[0]

        if self.index.ntotal != face_count:
            self.rebuild_index()

    def rebuild_index(self):
        index = self._new_index()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM faces WHERE embedding IS NOT NULL ORDER BY id ASC"
            ).fetchall()

        embeddings = []
        ids = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != self.dimension:
                continue
            embeddings.append(self._normalize_embedding(vector))
            ids.append(row["id"])

        if embeddings:
            index.add_with_ids(
                np.vstack(embeddings).astype(np.float32),
                np.asarray(ids, dtype=np.int64),
            )

        self.index = index
        faiss.write_index(self.index, INDEX_PATH)

    def upsert_content(
        self,
        platform,
        external_id,
        content_type,
        title="",
        source_url="",
        local_path="",
        subtitle_path="",
        asr_path="",
        post_text="",
        metadata=None,
        collection_source_id=None,
    ):
        metadata_json = (
            json.dumps(metadata or {}, ensure_ascii=False)
            if metadata is not None
            else None
        )
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM contents WHERE platform = ? AND external_id = ?
                """,
                (platform, external_id),
            ).fetchone()

            if existing:
                merged = dict(existing)
                merged["content_type"] = content_type or merged["content_type"]
                merged["title"] = title or merged["title"]
                merged["source_url"] = source_url or merged["source_url"]
                merged["local_path"] = local_path or merged["local_path"]
                merged["subtitle_path"] = subtitle_path or merged["subtitle_path"]
                merged["asr_path"] = asr_path or merged["asr_path"]
                merged["post_text"] = post_text or merged["post_text"]
                merged["metadata_json"] = (
                    metadata_json
                    if metadata_json is not None
                    else merged["metadata_json"]
                )
                merged["collection_source_id"] = (
                    int(collection_source_id)
                    if collection_source_id is not None
                    else merged.get("collection_source_id")
                )
                conn.execute(
                    """
                    UPDATE contents
                    SET content_type = ?, title = ?, source_url = ?, local_path = ?,
                        subtitle_path = ?, asr_path = ?, post_text = ?, metadata_json = ?,
                        collection_source_id = ?
                    WHERE id = ?
                    """,
                    (
                        merged["content_type"],
                        merged["title"],
                        merged["source_url"],
                        merged["local_path"],
                        merged["subtitle_path"],
                        merged["asr_path"],
                        merged["post_text"],
                        merged["metadata_json"],
                        merged["collection_source_id"],
                        existing["id"],
                    ),
                )
                content_id = existing["id"]
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO contents (
                        platform, external_id, content_type, title, source_url,
                        local_path, subtitle_path, asr_path, post_text, metadata_json,
                        collection_source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        platform,
                        external_id,
                        content_type,
                        title,
                        source_url,
                        local_path,
                        subtitle_path,
                        asr_path,
                        post_text,
                        metadata_json or json.dumps({}, ensure_ascii=False),
                        int(collection_source_id)
                        if collection_source_id is not None
                        else None,
                    ),
                )
                content_id = cursor.lastrowid

        return self.get_content_by_id(content_id)

    def update_content_paths(
        self,
        content_id,
        local_path=None,
        subtitle_path=None,
        asr_path=None,
        post_text=None,
    ):
        updates = {}
        if local_path:
            updates["local_path"] = local_path
        if subtitle_path:
            updates["subtitle_path"] = subtitle_path
        if asr_path:
            updates["asr_path"] = asr_path
        if post_text is not None:
            updates["post_text"] = post_text

        if not updates:
            return

        fields = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [content_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE contents SET {fields}, last_processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                params,
            )

    def get_content_by_id(self, content_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contents WHERE id = ?", (content_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_content_by_identity(self, platform, external_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM contents WHERE platform = ? AND external_id = ?
                """,
                (platform, external_id),
            ).fetchone()
        return dict(row) if row else None

    def content_has_faces(self, content_id):
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM faces WHERE content_id = ?",
                (content_id,),
            ).fetchone()[0]
        return count > 0

    def _serialize_source_row(self, row):
        item = dict(row)
        metadata = self._parse_json(item.get("metadata_json"), {})
        item["metadata"] = metadata
        item["keywords"] = metadata.get("keywords", [])
        item["last_sync_stats"] = metadata.get("last_sync_stats", {})
        item["user_id"] = metadata.get("user_id", "")
        item["sync_limit"] = metadata.get("limit")
        item["last_seen_post_id"] = metadata.get("last_seen_post_id", "")
        return item

    def register_collection_source(
        self,
        platform,
        source_type,
        source_url,
        title="",
        metadata=None,
    ):
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM collection_sources WHERE source_url = ?",
                (source_url,),
            ).fetchone()
            merged_metadata = self._parse_json(
                existing["metadata_json"] if existing else None,
                {},
            )
            merged_metadata.update(metadata or {})
            metadata_json = json.dumps(merged_metadata, ensure_ascii=False)

            conn.execute(
                """
                INSERT INTO collection_sources (platform, source_type, source_url, title, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_url) DO UPDATE SET
                    platform = excluded.platform,
                    source_type = excluded.source_type,
                    title = CASE WHEN excluded.title != '' THEN excluded.title ELSE collection_sources.title END,
                    metadata_json = excluded.metadata_json,
                    enabled = 1
                """,
                (platform, source_type, source_url, title or "", metadata_json),
            )
            row = conn.execute(
                "SELECT * FROM collection_sources WHERE source_url = ?",
                (source_url,),
            ).fetchone()
        return self._serialize_source_row(row)

    def list_collection_sources(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM collection_sources
                WHERE enabled = 1
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._serialize_source_row(row) for row in rows]

    def get_collection_source(self, source_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM collection_sources WHERE id = ?",
                (source_id,),
            ).fetchone()
        return self._serialize_source_row(row) if row else None

    def delete_collection_source(self, source_id):
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM collection_sources WHERE id = ?",
                (source_id,),
            )
        return int(cursor.rowcount or 0)

    def _storage_path_candidates(self, item):
        return (
            item.get("local_path"),
            item.get("subtitle_path"),
            item.get("asr_path"),
            item.get("image_path"),
            item.get("full_image_path"),
        )

    def _delete_managed_files(self, raw_paths):
        deleted_files = 0
        for raw_path in raw_paths:
            candidate = app_config.resolve_managed_path(raw_path)
            if candidate is None:
                continue
            path = Path(candidate)
            if not path.exists() or not path.is_file():
                continue
            try:
                path.unlink()
                deleted_files += 1
            except OSError:
                continue
        return deleted_files

    def _cleanup_orphan_people(self, conn):
        cursor = conn.execute(
            """
            DELETE FROM people
            WHERE person_id NOT IN (
                SELECT DISTINCT person_id
                FROM faces
                WHERE person_id != -1
            )
            """
        )
        return int(cursor.rowcount or 0)

    def clear_cluster_snapshots(self):
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM cluster_snapshots")
        return int(cursor.rowcount or 0)

    def get_source_deletion_preview(self, source_id):
        source = self.get_collection_source(source_id)
        if source is None:
            return None

        with self._connect() as conn:
            content_rows = conn.execute(
                """
                SELECT *
                FROM contents
                WHERE collection_source_id = ?
                UNION
                SELECT *
                FROM contents
                WHERE collection_source_id IS NULL
                  AND content_type = 'video'
                  AND json_extract(COALESCE(metadata_json, '{}'), '$.source_sync_url') = ?
                ORDER BY id ASC
                """,
                (source_id, source["source_url"]),
            ).fetchall()

            contents = [dict(row) for row in content_rows]
            content_ids = [int(item["id"]) for item in contents]
            faces = []
            if content_ids:
                placeholders = ",".join("?" for _ in content_ids)
                faces = [
                    dict(row)
                    for row in conn.execute(
                        f"""
                        SELECT id, content_id, image_path, full_image_path, person_id
                        FROM faces
                        WHERE content_id IN ({placeholders})
                        ORDER BY id ASC
                        """,
                        content_ids,
                    ).fetchall()
                ]

            unresolved_legacy_items = 0
            if source.get("platform") == "weibo":
                user_id = str(source.get("user_id") or "").strip()
                if user_id:
                    unresolved_legacy_items = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM contents
                        WHERE platform = 'weibo'
                          AND collection_source_id IS NULL
                          AND json_extract(COALESCE(metadata_json, '{}'), '$.user_id') = ?
                        """,
                        (user_id,),
                    ).fetchone()[0]

        file_paths = []
        for item in contents:
            file_paths.extend(self._storage_path_candidates(item))
        for face in faces:
            file_paths.extend(self._storage_path_candidates(face))

        return {
            "source": source,
            "contents": contents,
            "faces": faces,
            "file_paths": file_paths,
            "unresolved_legacy_items": int(unresolved_legacy_items or 0),
        }

    def delete_source_with_data(self, source_id):
        preview = self.get_source_deletion_preview(source_id)
        if preview is None:
            return None

        source = preview["source"]
        content_ids = [int(item["id"]) for item in preview["contents"]]
        face_ids = [int(item["id"]) for item in preview["faces"]]

        with self._connect() as conn:
            if face_ids:
                placeholders = ",".join("?" for _ in face_ids)
                conn.execute(
                    f"DELETE FROM face_fts WHERE id IN ({placeholders})",
                    face_ids,
                )

            deleted_faces = 0
            deleted_contents = 0
            if content_ids:
                placeholders = ",".join("?" for _ in content_ids)
                deleted_faces = int(
                    (
                        conn.execute(
                            f"DELETE FROM faces WHERE content_id IN ({placeholders})",
                            content_ids,
                        ).rowcount
                    )
                    or 0
                )
                deleted_contents = int(
                    (
                        conn.execute(
                            f"DELETE FROM contents WHERE id IN ({placeholders})",
                            content_ids,
                        ).rowcount
                    )
                    or 0
                )

            deleted_people = self._cleanup_orphan_people(conn)
            cleared_cluster_snapshot = int(
                (conn.execute("DELETE FROM cluster_snapshots").rowcount) or 0
            )
            deleted_source = int(
                (
                    conn.execute(
                        "DELETE FROM collection_sources WHERE id = ?",
                        (source_id,),
                    ).rowcount
                )
                or 0
            )

        self.rebuild_index()
        deleted_files = self._delete_managed_files(preview["file_paths"])
        return {
            "source": source,
            "deleted_source": deleted_source,
            "deleted_contents": deleted_contents,
            "deleted_faces": deleted_faces,
            "deleted_people": deleted_people,
            "deleted_files": deleted_files,
            "cleared_cluster_snapshot": cleared_cluster_snapshot > 0,
            "unresolved_legacy_items": preview["unresolved_legacy_items"],
        }

    def mark_source_synced(self, source_id, metadata=None, title=None):
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM collection_sources WHERE id = ?",
                (source_id,),
            ).fetchone()
            if existing is None:
                return
            merged_metadata = self._parse_json(existing["metadata_json"], {})
            merged_metadata.update(metadata or {})
            conn.execute(
                """
                UPDATE collection_sources
                SET last_synced_at = CURRENT_TIMESTAMP,
                    title = COALESCE(?, title),
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    title if title else None,
                    json.dumps(merged_metadata, ensure_ascii=False),
                    source_id,
                ),
            )

    def _fetch_faces_by_ids(self, face_id_list):
        if not face_id_list:
            return {}

        placeholders = ",".join("?" for _ in face_id_list)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, content_id, content_type, platform, external_id, video_id,
                       timestamp, image_path, full_image_path, source_url, description,
                       visual_text, subtitle_text, asr_text, post_text,
                       semantic_text, semantic_source, person_id
                FROM faces
                WHERE id IN ({placeholders})
                """,
                face_id_list,
            ).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def add_face(
        self,
        video_id,
        timestamp,
        image_path,
        full_image_path,
        source_url,
        embedding,
        description,
        content_id=None,
        content_type="video",
        platform="",
        external_id="",
        visual_text="",
        subtitle_text="",
        asr_text="",
        post_text="",
        semantic_text="",
        semantic_source=None,
    ):
        embedding = self._normalize_embedding(embedding)
        semantic_text = semantic_text or description or ""
        semantic_source_json = json.dumps(semantic_source or [], ensure_ascii=False)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO faces (
                    content_id, content_type, platform, external_id,
                    video_id, timestamp, image_path, full_image_path,
                    source_url, description, visual_text, subtitle_text,
                    asr_text, post_text, semantic_text, semantic_source, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_id,
                    content_type,
                    platform,
                    external_id,
                    video_id,
                    timestamp,
                    image_path,
                    full_image_path,
                    source_url,
                    semantic_text,
                    visual_text,
                    subtitle_text,
                    asr_text,
                    post_text,
                    semantic_text,
                    semantic_source_json,
                    embedding.tobytes(),
                ),
            )
            face_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO face_fts (id, description) VALUES (?, ?)",
                (face_id, semantic_text),
            )

        self.index.add_with_ids(
            embedding.reshape(1, -1).astype(np.float32),
            np.asarray([face_id], dtype=np.int64),
        )
        faiss.write_index(self.index, INDEX_PATH)
        return face_id

    def get_all_faces(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, content_id, content_type, platform, external_id, video_id,
                       timestamp, image_path, full_image_path, source_url, description,
                       visual_text, subtitle_text, asr_text, post_text,
                       semantic_text, semantic_source, person_id
                FROM faces
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_faces_with_embeddings(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, content_id, content_type, platform, external_id, video_id,
                       timestamp, image_path, full_image_path, source_url, description,
                       visual_text, subtitle_text, asr_text, post_text,
                       semantic_text, semantic_source, person_id, embedding
                FROM faces
                WHERE embedding IS NOT NULL
                ORDER BY id ASC
                """
            ).fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            if embedding.size != self.dimension:
                continue
            item = dict(row)
            item["embedding"] = embedding
            results.append(item)
        return results

    def get_labeled_faces_with_embeddings(self):
        return [
            record
            for record in self.get_all_faces_with_embeddings()
            if int(record.get("person_id", -1)) != -1
        ]

    def search_faces_by_embedding(self, embedding, top_k=10, min_score=0.4):
        if self.index.ntotal == 0:
            return []

        query = self._normalize_embedding(embedding).reshape(1, -1).astype(np.float32)
        limit = max(top_k * 5, top_k)
        scores, face_ids = self.index.search(query, limit)

        valid_matches = [
            (int(face_id), float(score))
            for score, face_id in zip(scores[0], face_ids[0])
            if face_id != -1 and score >= min_score
        ]
        if not valid_matches:
            return []

        ordered_ids = [face_id for face_id, _ in valid_matches]
        row_map = self._fetch_faces_by_ids(ordered_ids)
        results = []
        for face_id, score in valid_matches:
            row = row_map.get(face_id)
            if not row:
                continue
            row["score"] = score
            row["metric"] = "cosine"
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def search_faces_by_metric(self, embedding, metric="cosine", top_k=10):
        metric = (metric or "cosine").lower()
        records = self.get_all_faces_with_embeddings()
        if not records:
            return []

        query = self._normalize_embedding(embedding)
        scored = []
        for record in records:
            target = self._normalize_embedding(record["embedding"])
            if metric == "euclidean":
                distance = float(np.linalg.norm(query - target))
                score = 1.0 / (1.0 + distance)
                scored.append((record["id"], score, distance))
            else:
                score = float(np.dot(query, target))
                scored.append((record["id"], score, None))

        if metric == "euclidean":
            scored.sort(key=lambda item: item[2])
        else:
            scored.sort(key=lambda item: item[1], reverse=True)

        selected = scored[:top_k]
        row_map = self._fetch_faces_by_ids([item[0] for item in selected])
        results = []
        for face_id, score, distance in selected:
            row = row_map.get(face_id)
            if not row:
                continue
            row["score"] = score
            row["metric"] = metric
            if distance is not None:
                row["distance"] = distance
            results.append(row)
        return results

    def update_person_ids(self, id_label_map):
        with self._connect() as conn:
            conn.executemany("UPDATE faces SET person_id = ? WHERE id = ?", id_label_map)

    def replace_all_person_ids(self, id_label_map, people_name_map=None):
        people_name_map = people_name_map or {}
        with self._connect() as conn:
            conn.execute("UPDATE faces SET person_id = -1")
            if id_label_map:
                conn.executemany(
                    "UPDATE faces SET person_id = ? WHERE id = ?",
                    id_label_map,
                )
            conn.execute("DELETE FROM people")
            named_people = [
                (int(person_id), str(name).strip())
                for person_id, name in people_name_map.items()
                if str(name or "").strip()
            ]
            if named_people:
                conn.executemany(
                    "INSERT INTO people (person_id, name) VALUES (?, ?)",
                    named_people,
                )

    def get_clustered_people(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.person_id, COUNT(f.id) AS count, MIN(f.image_path) AS cover_image, p.name
                FROM faces f
                LEFT JOIN people p ON f.person_id = p.person_id
                WHERE f.person_id != -1
                GROUP BY f.person_id
                ORDER BY count DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_person_timeline(self, person_id):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, content_id, content_type, platform, external_id, video_id,
                       timestamp, image_path, full_image_path, source_url, description,
                       visual_text, subtitle_text, asr_text, post_text,
                       semantic_text, semantic_source, person_id
                FROM faces
                WHERE person_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (person_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_unassigned_faces(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, content_id, content_type, platform, external_id, video_id,
                       timestamp, image_path, full_image_path, source_url, description,
                       visual_text, subtitle_text, asr_text, post_text,
                       semantic_text, semantic_source, person_id
                FROM faces
                WHERE person_id = -1
                ORDER BY created_at DESC
                LIMIT 50
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_person(self, person_id, name):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO people (person_id, name) VALUES (?, ?)",
                (person_id, name),
            )

    def get_person_name_map(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT person_id, name
                FROM people
                WHERE name IS NOT NULL AND TRIM(name) != ''
                ORDER BY person_id ASC
                """
            ).fetchall()
        return {int(row["person_id"]): row["name"] for row in rows}

    def reassign_face(self, face_id, target_person_id):
        with self._connect() as conn:
            conn.execute(
                "UPDATE faces SET person_id = ? WHERE id = ?",
                (target_person_id, face_id),
            )

    def merge_persons(self, source_person_id, target_person_id):
        with self._connect() as conn:
            conn.execute(
                "UPDATE faces SET person_id = ? WHERE person_id = ?",
                (target_person_id, source_person_id),
            )
            conn.execute("DELETE FROM people WHERE person_id = ?", (source_person_id,))

    def get_max_person_id(self):
        with self._connect() as conn:
            result = conn.execute(
                "SELECT MAX(person_id) FROM faces WHERE person_id != -1"
            ).fetchone()[0]
        return result if result is not None else -1

    def get_unassigned_faces_with_embeddings(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, person_id, embedding
                FROM faces
                WHERE person_id = -1 AND embedding IS NOT NULL
                """
            ).fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            if embedding.size != self.dimension:
                continue
            results.append({"id": row["id"], "person_id": row["person_id"], "embedding": embedding})
        return results

    def has_cluster_snapshot(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM cluster_snapshots WHERE slot = 1"
            ).fetchone()
        return row is not None

    def save_cluster_snapshot(self, cluster_config=None):
        with self._connect() as conn:
            face_rows = conn.execute(
                "SELECT id, person_id FROM faces ORDER BY id ASC"
            ).fetchall()
            people_rows = conn.execute(
                "SELECT person_id, name FROM people ORDER BY person_id ASC"
            ).fetchall()

            face_assignments_json = json.dumps(
                [
                    {"id": int(row["id"]), "person_id": int(row["person_id"])}
                    for row in face_rows
                ],
                ensure_ascii=False,
            )
            people_json = json.dumps(
                [
                    {"person_id": int(row["person_id"]), "name": row["name"] or ""}
                    for row in people_rows
                ],
                ensure_ascii=False,
            )
            cluster_config_json = json.dumps(cluster_config or {}, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO cluster_snapshots (
                    slot, face_assignments_json, people_json, cluster_config_json, created_at
                ) VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(slot) DO UPDATE SET
                    face_assignments_json = excluded.face_assignments_json,
                    people_json = excluded.people_json,
                    cluster_config_json = excluded.cluster_config_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    face_assignments_json,
                    people_json,
                    cluster_config_json,
                ),
            )
        return len(face_rows)

    def restore_cluster_snapshot(self):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT face_assignments_json, people_json, cluster_config_json
                FROM cluster_snapshots
                WHERE slot = 1
                """
            ).fetchone()
            if row is None:
                return None

            face_assignments = json.loads(row["face_assignments_json"] or "[]")
            people_rows = json.loads(row["people_json"] or "[]")
            cluster_config = json.loads(row["cluster_config_json"] or "{}")

            conn.execute("UPDATE faces SET person_id = -1")
            if face_assignments:
                conn.executemany(
                    "UPDATE faces SET person_id = ? WHERE id = ?",
                    [
                        (int(item["person_id"]), int(item["id"]))
                        for item in face_assignments
                    ],
                )

            conn.execute("DELETE FROM people")
            named_people = [
                (int(item["person_id"]), str(item.get("name") or "").strip())
                for item in people_rows
                if str(item.get("name") or "").strip()
            ]
            if named_people:
                conn.executemany(
                    "INSERT INTO people (person_id, name) VALUES (?, ?)",
                    named_people,
                )

        return {
            "restored_faces": len(face_assignments),
            "restored_people": len(named_people),
            "cluster_config": cluster_config,
        }

    def reset_database_state(self):
        removed_files = 0
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
            except PermissionError:
                with self._connect() as conn:
                    conn.execute("DROP TABLE IF EXISTS faces")
                    conn.execute("DROP TABLE IF EXISTS face_fts")
                    conn.execute("DROP TABLE IF EXISTS people")
                    conn.execute("DROP TABLE IF EXISTS contents")
                    conn.execute("DROP TABLE IF EXISTS collection_sources")
                    conn.execute("DROP TABLE IF EXISTS cluster_snapshots")
                    conn.execute("DROP TABLE IF EXISTS users")
            removed_files += 1
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
            removed_files += 1
        self.init_db()
        self.init_index()
        return removed_files
