import sqlite3
import faiss
import numpy as np
import os

DB_PATH = "storage/metadata.db"
INDEX_PATH = "storage/face_index.faiss"


class DatabaseManager:
    def __init__(self):
        os.makedirs("storage", exist_ok=True)
        self.init_db()
        self.init_index()

    def init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            # 【修改点】增加 full_image_path 和 source_url
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
            # 管理姓名
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """
            )

    def init_index(self):
        self.dimension = 512
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

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
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (video_id, timestamp, image_path, full_image_path, source_url, description, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
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

        vec = np.array([embedding], dtype="float32")
        self.index.add(vec)
        faiss.write_index(self.index, INDEX_PATH)
        return face_id

    def get_all_faces(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # 【修改点】查出新字段
            cursor.execute(
                "SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description FROM faces"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_faces_with_embeddings(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # 【修改点】查出新字段
            cursor.execute(
                "SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description, embedding FROM faces"
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                emb_bytes = row["embedding"]
                if emb_bytes:
                    embedding = np.frombuffer(emb_bytes, dtype=np.float32)
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

    def update_person_ids(self, id_label_map):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE faces SET person_id = ? WHERE id = ?", id_label_map
            )
            conn.commit()

    def get_clustered_people(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT person_id, COUNT(*) as count, MIN(image_path) as cover_image
                FROM faces WHERE person_id != -1 GROUP BY person_id ORDER BY count DESC
            """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_person_timeline(self, person_id):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description 
                FROM faces WHERE person_id = ? ORDER BY timestamp ASC
            """,
                (person_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # 1. 获取所有未归类的人脸 (person_id = -1)
    def get_unassigned_faces(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, video_id, timestamp, image_path, full_image_path, source_url, description 
                FROM faces WHERE person_id = -1 ORDER BY created_at DESC LIMIT 50
            """
            )
            return [dict(row) for row in cursor.fetchall()]

    # 2. 更新人物名称
    def rename_person(self, person_id, name):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO people (person_id, name) VALUES (?, ?)",
                (person_id, name),
            )

    # 3. 重新分配人脸到指定人物
    def reassign_face(self, face_id, target_person_id):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE faces SET person_id = ? WHERE id = ?",
                (target_person_id, face_id),
            )

    # 4. (修改现有的) 获取聚类人物列表，并连表查询出姓名
    def get_clustered_people(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # 联表查询查出姓名，如果没有姓名则为 NULL
            cursor.execute(
                """
                SELECT f.person_id, COUNT(f.id) as count, MIN(f.image_path) as cover_image, p.name
                FROM faces f
                LEFT JOIN people p ON f.person_id = p.person_id
                WHERE f.person_id != -1 
                GROUP BY f.person_id 
                ORDER BY count DESC
            """
            )
            return [dict(row) for row in cursor.fetchall()]

    def merge_persons(self, source_person_id, target_person_id):
        with sqlite3.connect(DB_PATH) as conn:
            # 1. 把原人物的所有脸转移给目标人物
            conn.execute(
                "UPDATE faces SET person_id = ? WHERE person_id = ?",
                (target_person_id, source_person_id),
            )
            # 2. 删除原人物的命名记录，保持数据库整洁
            conn.execute("DELETE FROM people WHERE person_id = ?", (source_person_id,))

    # 在 core/database.py 中增加：
    def get_max_person_id(self):
        """获取当前最大的 person_id，以便增量分配"""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(person_id) FROM faces WHERE person_id != -1")
            result = cursor.fetchone()[0]
            return result if result is not None else -1

    def get_unassigned_faces_with_embeddings(self):
        """只获取还没有归类的脸进行特征聚类"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM faces WHERE person_id = -1")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                emb_bytes = row["embedding"]
                if emb_bytes:
                    embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                    results.append({"id": row["id"], "embedding": embedding})
            return results
