import os
import sqlite3

import faiss
import numpy as np

DB_PATH = "storage/metadata.db"
INDEX_PATH = "storage/face_index.faiss"


class DatabaseManager:
    def __init__(self):
        os.makedirs("storage", exist_ok=True)
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
            face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

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

    def add_face(
        self,
        video_id,
        timestamp,
        image_path,
        full_image_path,
        source_url,
        embedding,
        description,
    ):
        embedding = self._normalize_embedding(embedding)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO faces (
                    video_id, timestamp, image_path, full_image_path, source_url, description, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    timestamp,
                    image_path,
                    full_image_path,
                    source_url,
                    description,
                    embedding.tobytes(),
                ),
            )
            face_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO face_fts (id, description) VALUES (?, ?)",
                (face_id, description),
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
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description
                FROM faces
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_faces_with_embeddings(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description, embedding
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
            results.append(
                {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "timestamp": row["timestamp"],
                    "image_path": row["image_path"],
                    "full_image_path": row["full_image_path"],
                    "source_url": row["source_url"],
                    "description": row["description"],
                    "embedding": embedding,
                }
            )
        return results

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

        face_id_list = [face_id for face_id, _ in valid_matches]
        score_map = {face_id: score for face_id, score in valid_matches}
        placeholders = ",".join("?" for _ in face_id_list)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description
                FROM faces
                WHERE id IN ({placeholders})
                """,
                face_id_list,
            ).fetchall()

        row_map = {row["id"]: dict(row) for row in rows}
        ordered_results = []
        for face_id, _ in valid_matches:
            row = row_map.get(face_id)
            if not row:
                continue
            row["score"] = score_map[face_id]
            ordered_results.append(row)
            if len(ordered_results) >= top_k:
                break
        return ordered_results

    def update_person_ids(self, id_label_map):
        with self._connect() as conn:
            conn.executemany("UPDATE faces SET person_id = ? WHERE id = ?", id_label_map)

    def get_clustered_people(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.person_id, COUNT(f.id) as count, MIN(f.image_path) as cover_image, p.name
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
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description
                FROM faces
                WHERE person_id = ?
                ORDER BY timestamp ASC
                """,
                (person_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_unassigned_faces(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description
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
                "SELECT id, embedding FROM faces WHERE person_id = -1 AND embedding IS NOT NULL"
            ).fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            if embedding.size != self.dimension:
                continue
            results.append({"id": row["id"], "embedding": embedding})
        return results
