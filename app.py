import os

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.analyzer import AIProcessor
from core.clustering import perform_clustering
from core.collector import VideoCollector
from core.database import DatabaseManager

app = FastAPI(title="FaceRetriever 2026")

TEXT_SEARCH_THRESHOLD = 0.15
IMAGE_SEARCH_THRESHOLD = 0.4
MAX_TEXT_RESULTS = 20
MAX_IMAGE_RESULTS = 10


db = DatabaseManager()
ai_engine = AIProcessor()
collector = VideoCollector()

os.makedirs("storage/faces", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/faces", StaticFiles(directory="storage/faces"), name="faces")


@app.get("/")
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def process_video_task(video_info):
    cap = cv2.VideoCapture(video_info["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 25

    frame_interval = max(int(fps * 2), 1)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            results = ai_engine.process_frame(frame)
            if results:
                full_filename = f"{video_info['id']}_{frame_id}_full.jpg"
                full_frame = results[0]["full_frame"]

                if full_frame.shape[1] > 1280:
                    full_frame = cv2.resize(
                        full_frame,
                        (1280, int(1280 * full_frame.shape[0] / full_frame.shape[1])),
                    )
                cv2.imwrite(os.path.join("storage/faces", full_filename), full_frame)

                for i, face_data in enumerate(results):
                    face_filename = f"{video_info['id']}_{frame_id}_face_{i}.jpg"
                    cv2.imwrite(
                        os.path.join("storage/faces", face_filename),
                        face_data["face_img"],
                    )
                    db.add_face(
                        video_id=video_info["id"],
                        timestamp=count / fps,
                        image_path=f"/faces/{face_filename}",
                        full_image_path=f"/faces/{full_filename}",
                        source_url=video_info.get("url", ""),
                        embedding=face_data["embedding"],
                        description=face_data["description"],
                    )

        count += 1
        frame_id += 1

    cap.release()


@app.post("/api/collect")
async def collect_video(url: str, background_tasks: BackgroundTasks):
    try:
        info = collector.download(url)
        background_tasks.add_task(process_video_task, info)
        return {
            "status": "success",
            "msg": "Video downloaded, processing in background",
            "video": info["title"],
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


@app.get("/api/search/text")
async def search_by_text(q: str):
    query = q.strip()
    if not query:
        return {"status": "success", "results": []}

    all_records = db.get_all_faces()
    if not all_records:
        return {"status": "success", "results": []}

    descriptions = [record.get("description", "") for record in all_records]
    ranked = ai_engine.rank_texts_by_similarity(query, descriptions)

    search_results = []
    for index, similarity in ranked:
        if similarity < TEXT_SEARCH_THRESHOLD:
            continue

        record = all_records[index]
        search_results.append(
            {
                "id": record.get("id"),
                "image": record.get("image_path"),
                "full_image": record.get("full_image_path"),
                "source_url": record.get("source_url"),
                "video_id": record.get("video_id"),
                "timestamp": record.get("timestamp"),
                "desc": record.get("description", ""),
                "score": round(float(similarity), 4),
            }
        )

    search_results.sort(key=lambda item: item["score"], reverse=True)
    return {"status": "success", "results": search_results[:MAX_TEXT_RESULTS]}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Invalid image file"}

        target_embedding, _ = ai_engine.get_face_embedding(img)
        if target_embedding is None:
            return {"status": "error", "message": "No face detected"}

        matches = db.search_faces_by_embedding(
            target_embedding,
            top_k=MAX_IMAGE_RESULTS,
            min_score=IMAGE_SEARCH_THRESHOLD,
        )
        results = [
            {
                "id": face["id"],
                "video_id": face["video_id"],
                "timestamp": face["timestamp"],
                "image": face["image_path"],
                "full_image": face["full_image_path"],
                "source_url": face["source_url"],
                "desc": face["description"],
                "score": round(float(face["score"]), 4),
            }
            for face in matches
        ]
        return {"status": "success", "results": results}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/api/cluster/run")
async def run_clustering():
    try:
        result = perform_clustering(db)
        return result
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/api/people")
async def get_people():
    people = db.get_clustered_people()
    unassigned = db.get_unassigned_faces()
    return {"status": "success", "people": people, "unassigned": unassigned}


@app.get("/api/person/{person_id}")
async def get_person_details(person_id: int):
    timeline = db.get_person_timeline(person_id)
    return {"status": "success", "timeline": timeline}


@app.post("/api/person/rename")
async def rename_person(data: dict):
    person_id = data.get("person_id")
    new_name = data.get("name")
    if person_id is None or not new_name:
        return {"status": "error", "message": "Missing required fields"}

    db.rename_person(person_id, new_name)
    return {"status": "success"}


@app.post("/api/face/reassign")
async def reassign_face(data: dict):
    face_id = data.get("face_id")
    target_person_id = data.get("target_person_id")
    if face_id is None or target_person_id is None:
        return {"status": "error", "message": "Missing required fields"}

    db.reassign_face(face_id, target_person_id)
    return {"status": "success"}


@app.post("/api/person/merge")
async def merge_person(data: dict):
    source_id = data.get("source_person_id")
    target_id = data.get("target_person_id")
    if source_id is None or target_id is None:
        return {"status": "error", "message": "Missing required fields"}

    try:
        db.merge_persons(source_id, target_id)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)