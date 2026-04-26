import json
import os
import secrets
import threading
from contextlib import contextmanager
from pathlib import Path

# Default to the domestic mirror unless the process already selected an endpoint.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
from fastapi import BackgroundTasks, Body, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.alignment import align_text_to_timestamp, parse_srt_file, write_srt_file
from core.analyzer import AIProcessor
from core.clustering import (
    compare_image_search_metrics,
    evaluate_clustering,
    perform_clustering,
)
from core.collector import VideoCollector
from core.config import app_config
from core.database import DatabaseManager
from core.source_adapters import (
    SourceAdapterError,
    SourceAdapterRegistry,
    parse_adapter_config,
)

app = FastAPI(title="FaceRetriever 2026")

db = DatabaseManager()
ai_engine = AIProcessor()
collector = VideoCollector()
source_adapter_registry = SourceAdapterRegistry(collector)

_task_state = {"active": 0, "maintenance": False}
_task_state_lock = threading.Lock()
_task_local = threading.local()
_admin_sessions = set()
_admin_sessions_lock = threading.Lock()
_admin_cookie_name = "face_admin_session"
_admin_session_seconds = 8 * 60 * 60

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/faces", StaticFiles(directory=str(app_config.faces_dir)), name="faces")
app.mount("/content", StaticFiles(directory=str(app_config.content_dir)), name="content")


@app.get("/")
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _safe_filename(name):
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def _parse_sources(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return [part.strip() for part in str(value).split(",") if part.strip()]


def _normalize_keywords(value):
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).replace("，", ",").splitlines()
    merged = []
    for item in raw:
        merged.extend(str(item).split(","))
    keywords = []
    for item in merged:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in keywords:
            keywords.append(cleaned)
    return keywords


def _system_status_payload():
    return {
        "status": "success",
        "cluster_config": app_config.cluster_defaults(),
        "face_quality_config": app_config.face_quality_config(),
        "runtime_config": app_config.runtime_config(),
        "has_cluster_snapshot": db.has_cluster_snapshot(),
        "sources": db.list_collection_sources(),
        "source_adapters": source_adapter_registry.list_adapters(),
        "source_platform_options": source_adapter_registry.list_platform_options(),
    }


def _create_admin_session(response: Response):
    token = secrets.token_urlsafe(32)
    with _admin_sessions_lock:
        _admin_sessions.add(token)
    response.set_cookie(
        _admin_cookie_name,
        token,
        max_age=_admin_session_seconds,
        httponly=True,
        samesite="lax",
    )


def _clear_admin_session(request: Request, response: Response):
    token = request.cookies.get(_admin_cookie_name)
    if token:
        with _admin_sessions_lock:
            _admin_sessions.discard(token)
    response.delete_cookie(_admin_cookie_name)


def _clear_all_admin_sessions():
    with _admin_sessions_lock:
        _admin_sessions.clear()


def _is_admin_authenticated(request: Request):
    token = request.cookies.get(_admin_cookie_name)
    if not token:
        return False
    with _admin_sessions_lock:
        return token in _admin_sessions


def _admin_status_payload(request: Request):
    return {
        "status": "success",
        "admin_initialized": app_config.has_admin_password(),
        "admin_authenticated": _is_admin_authenticated(request),
    }


def _require_admin(request: Request):
    if app_config.has_admin_password() and _is_admin_authenticated(request):
        return None
    if not app_config.has_admin_password():
        message = "Admin password is not set."
    else:
        message = "Admin authentication required."
    return {"status": "error", "message": message, "admin_required": True}


def _active_task_count():
    with _task_state_lock:
        return int(_task_state["active"])


def _maintenance_active():
    with _task_state_lock:
        return bool(_task_state["maintenance"])


def _maintenance_message():
    return "System maintenance is in progress. Try again later."


def _busy_message():
    return "Background collection or processing is still running."


def _runtime_blocked_response():
    return {"status": "error", "message": _maintenance_message()}


def _enter_task_scope():
    depth = getattr(_task_local, "depth", 0)
    if depth == 0:
        with _task_state_lock:
            _task_state["active"] += 1
    _task_local.depth = depth + 1


def _leave_task_scope():
    depth = getattr(_task_local, "depth", 0)
    if depth <= 1:
        if hasattr(_task_local, "depth"):
            delattr(_task_local, "depth")
        with _task_state_lock:
            _task_state["active"] = max(int(_task_state["active"]) - 1, 0)
        return
    _task_local.depth = depth - 1


@contextmanager
def _tracked_task():
    _enter_task_scope()
    try:
        yield
    finally:
        _leave_task_scope()


@contextmanager
def _maintenance_guard():
    with _task_state_lock:
        if _task_state["maintenance"]:
            raise RuntimeError(_maintenance_message())
        if _task_state["active"] > 0:
            raise RuntimeError(_busy_message())
        _task_state["maintenance"] = True
    try:
        yield
    finally:
        with _task_state_lock:
            _task_state["maintenance"] = False


def _remove_runtime_storage():
    removed_files_count = 0
    for directory in app_config.managed_runtime_dirs(include_videos=False):
        directory.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            directory.rglob("*"),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for path in paths:
            try:
                if path.is_file():
                    path.unlink()
                    removed_files_count += 1
                elif path.is_dir():
                    path.rmdir()
            except OSError:
                continue
        directory.mkdir(parents=True, exist_ok=True)
    return removed_files_count


def _cookie_files():
    return [
        app_config.weibo_cookie_path,
        app_config.storage_dir / "www.youtube.com_cookies.txt",
    ]


def _preserved_cookie_files(preserve_cookies):
    if not preserve_cookies:
        return []
    return [str(path) for path in _cookie_files() if path.exists()]


def _remove_cookie_files():
    removed = 0
    for path in _cookie_files():
        if not path.exists() or not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _rebuild_runtime_state():
    global db, collector, source_adapter_registry
    db = DatabaseManager()
    collector = VideoCollector()
    source_adapter_registry = SourceAdapterRegistry(collector)


def _validate_face_quality_payload(data):
    try:
        enabled = bool(data.get("enabled", True))
        min_face_size = int(data.get("min_face_size"))
        min_laplacian_var = float(data.get("min_laplacian_var"))
        max_pose_deviation = float(data.get("max_pose_deviation"))
    except (TypeError, ValueError):
        raise ValueError("Invalid face quality parameters.")

    if not 20 <= min_face_size <= 256:
        raise ValueError("min_face_size must be between 20 and 256.")
    if not 0 <= min_laplacian_var <= 1000:
        raise ValueError("min_laplacian_var must be between 0 and 1000.")
    if not 0.0 <= max_pose_deviation <= 1.0:
        raise ValueError("max_pose_deviation must be between 0.0 and 1.0.")

    return {
        "enabled": enabled,
        "min_face_size": min_face_size,
        "min_laplacian_var": min_laplacian_var,
        "max_pose_deviation": max_pose_deviation,
    }


def _validate_runtime_config_payload(data):
    payload = data or {}
    updates = {}

    processing = payload.get("processing") or {}
    if processing:
        try:
            frame_sample_seconds = float(processing.get("frame_sample_seconds"))
        except (TypeError, ValueError):
            raise ValueError("frame_sample_seconds must be a number.")
        if not 0.2 <= frame_sample_seconds <= 30:
            raise ValueError("frame_sample_seconds must be between 0.2 and 30.")
        updates["processing"] = {"frame_sample_seconds": frame_sample_seconds}

    search = payload.get("search") or {}
    if search:
        try:
            text_threshold = float(search.get("text_threshold"))
            image_cosine_threshold = float(search.get("image_cosine_threshold"))
            image_top_k = int(search.get("image_top_k"))
            text_top_k = int(search.get("text_top_k"))
            semantic_model_id = str(
                search.get("semantic_model_id") or app_config.semantic_model_id
            ).strip()
            semantic_model_prompt_name = str(
                search.get("semantic_model_prompt_name")
                or app_config.semantic_model_prompt_name
            ).strip()
            semantic_model_mode = str(
                search.get("semantic_model_mode") or app_config.semantic_model_mode
            ).strip()
            semantic_corpus_style = str(
                search.get("semantic_corpus_style") or app_config.semantic_corpus_style
            ).strip()
        except (TypeError, ValueError):
            raise ValueError("Invalid search parameters.")
        if not 0 <= text_threshold <= 1:
            raise ValueError("text_threshold must be between 0 and 1.")
        if not 0 <= image_cosine_threshold <= 1:
            raise ValueError("image_cosine_threshold must be between 0 and 1.")
        if not 1 <= image_top_k <= 100:
            raise ValueError("image_top_k must be between 1 and 100.")
        if not 1 <= text_top_k <= 100:
            raise ValueError("text_top_k must be between 1 and 100.")
        if not semantic_model_id:
            raise ValueError("semantic_model_id is required.")
        if semantic_model_mode not in {"standard", "qwen3_4b_int4_experimental"}:
            raise ValueError("Unsupported semantic_model_mode.")
        if semantic_corpus_style not in {"structured_zh"}:
            raise ValueError("Unsupported semantic_corpus_style.")
        updates["search"] = {
            "text_threshold": text_threshold,
            "image_cosine_threshold": image_cosine_threshold,
            "image_top_k": image_top_k,
            "text_top_k": text_top_k,
            "semantic_model_id": semantic_model_id,
            "semantic_model_prompt_name": semantic_model_prompt_name,
            "semantic_model_mode": semantic_model_mode,
            "semantic_corpus_style": semantic_corpus_style,
        }

    collection = payload.get("collection") or {}
    if collection:
        try:
            source_sync_limit = int(collection.get("source_sync_limit"))
            weibo_cookie_enabled = bool(collection.get("weibo_cookie_enabled"))
            weibo_source_sync_limit = int(collection.get("weibo_source_sync_limit"))
            weibo_timeout_seconds = int(collection.get("weibo_timeout_seconds"))
            weibo_retry_count = int(collection.get("weibo_retry_count"))
            bilibili_cookie_enabled = bool(
                collection.get(
                    "bilibili_cookie_enabled",
                    app_config.bilibili_cookie_enabled,
                )
            )
            bilibili_impersonate = str(
                collection.get("bilibili_impersonate", app_config.bilibili_impersonate)
                or ""
            ).strip()
            bilibili_referer = str(
                collection.get("bilibili_referer", app_config.bilibili_referer) or ""
            ).strip()
        except (TypeError, ValueError):
            raise ValueError("Invalid collection parameters.")
        if not 1 <= source_sync_limit <= 100:
            raise ValueError("source_sync_limit must be between 1 and 100.")
        if not 1 <= weibo_source_sync_limit <= 100:
            raise ValueError("weibo_source_sync_limit must be between 1 and 100.")
        if not 5 <= weibo_timeout_seconds <= 120:
            raise ValueError("weibo_timeout_seconds must be between 5 and 120.")
        if not 1 <= weibo_retry_count <= 10:
            raise ValueError("weibo_retry_count must be between 1 and 10.")
        if not bilibili_impersonate:
            raise ValueError("bilibili_impersonate is required.")
        if not bilibili_referer:
            raise ValueError("bilibili_referer is required.")
        updates["collection"] = {
            "source_sync_limit": source_sync_limit,
            "weibo_cookie_enabled": weibo_cookie_enabled,
            "weibo_source_sync_limit": weibo_source_sync_limit,
            "weibo_timeout_seconds": weibo_timeout_seconds,
            "weibo_retry_count": weibo_retry_count,
            "bilibili_cookie_enabled": bilibili_cookie_enabled,
            "bilibili_impersonate": bilibili_impersonate,
            "bilibili_referer": bilibili_referer,
        }

    transcription = payload.get("transcription") or {}
    if transcription:
        preferred_backend = str(
            transcription.get("preferred_backend") or app_config.transcription_backend
        ).strip()
        model_size = str(
            transcription.get("model_size") or app_config.transcription_model_size
        ).strip()
        initial_prompt = str(
            transcription.get("initial_prompt") or app_config.transcription_initial_prompt
        ).strip()
        hotwords = transcription.get("hotwords", app_config.transcription_hotwords)
        if isinstance(hotwords, str):
            hotword_values = [
                item.strip()
                for item in hotwords.replace("\n", ",").split(",")
            ]
        elif isinstance(hotwords, list):
            hotword_values = [str(item).strip() for item in hotwords]
        else:
            raise ValueError("hotwords must be a list or comma-separated string.")
        hotword_values = [item for item in hotword_values if item]
        if preferred_backend not in {"faster_whisper", "whisper"}:
            raise ValueError("Unsupported transcription backend.")
        if model_size not in {
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        }:
            raise ValueError("Unsupported transcription model_size.")
        updates["transcription"] = {
            "enabled": bool(
                transcription.get("enabled", app_config.transcription_enabled)
            ),
            "preferred_backend": preferred_backend,
            "model_size": model_size,
            "initial_prompt": initial_prompt,
            "hotwords": hotword_values,
        }

    vision = payload.get("vision") or {}
    if vision:
        vlm_model_id = str(vision.get("vlm_model_id") or app_config.vlm_model_id).strip()
        caption_style = str(
            vision.get("caption_style") or app_config.caption_style
        ).strip()
        caption_language = str(
            vision.get("caption_language") or app_config.caption_language
        ).strip()
        if not vlm_model_id:
            raise ValueError("vlm_model_id is required.")
        if caption_style not in {"retrieval_keywords", "detailed_caption"}:
            raise ValueError("Unsupported caption_style.")
        if caption_language not in {"zh", "en"}:
            raise ValueError("Unsupported caption_language.")
        updates["vision"] = {
            "vlm_model_id": vlm_model_id,
            "release_vlm_after_task": bool(
                vision.get("release_vlm_after_task", app_config.release_vlm_after_task)
            ),
            "release_text_encoder_before_vlm": bool(
                vision.get(
                    "release_text_encoder_before_vlm",
                    app_config.release_text_encoder_before_vlm,
                )
            ),
            "caption_style": caption_style,
            "caption_language": caption_language,
            "caption_include_ocr_hint": bool(
                vision.get(
                    "caption_include_ocr_hint",
                    app_config.caption_include_ocr_hint,
                )
            ),
        }

    clustering = payload.get("clustering") or {}
    if clustering:
        try:
            algorithm = str(clustering.get("algorithm") or "").lower()
            metric = str(clustering.get("metric") or "").lower()
            eps = float(clustering.get("eps"))
            min_samples = int(clustering.get("min_samples"))
        except (TypeError, ValueError):
            raise ValueError("Invalid clustering parameters.")
        if algorithm not in {"dbscan", "hdbscan", "optics"}:
            raise ValueError("Unsupported clustering algorithm.")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("Unsupported clustering metric.")
        if not 0.01 <= eps <= 5:
            raise ValueError("eps must be between 0.01 and 5.")
        if not 2 <= min_samples <= 50:
            raise ValueError("min_samples must be between 2 and 50.")
        updates["clustering"] = {
            "algorithm": algorithm,
            "metric": metric,
            "eps": eps,
            "min_samples": min_samples,
        }

    face_quality = payload.get("face_quality") or {}
    if face_quality:
        updates["face_quality"] = _validate_face_quality_payload(face_quality)

    if not updates:
        raise ValueError("No supported configuration fields were provided.")
    return updates


def _serialize_face(record):
    semantic_text = record.get("semantic_text") or record.get("description") or ""
    timestamp = float(record.get("timestamp") or 0.0)
    content_type = record.get("content_type") or "video"
    return {
        "id": record.get("id"),
        "content_id": record.get("content_id"),
        "content_type": content_type,
        "platform": record.get("platform") or "",
        "external_id": record.get("external_id") or record.get("video_id") or "",
        "video_id": record.get("video_id") or record.get("external_id") or "",
        "timestamp": timestamp,
        "image": record.get("image_path"),
        "image_path": record.get("image_path"),
        "full_image": record.get("full_image_path"),
        "full_image_path": record.get("full_image_path"),
        "source_url": record.get("source_url") or "",
        "desc": semantic_text,
        "description": semantic_text,
        "visual_text": record.get("visual_text") or "",
        "subtitle_text": record.get("subtitle_text") or "",
        "asr_text": record.get("asr_text") or "",
        "speech_text": record.get("subtitle_text") or record.get("asr_text") or "",
        "post_text": record.get("post_text") or "",
        "semantic_text": semantic_text,
        "semantic_source": _parse_sources(record.get("semantic_source")),
        "person_id": record.get("person_id"),
        "cluster_config": app_config.cluster_defaults(),
        "score": record.get("score"),
        "metric": record.get("metric"),
        "distance": record.get("distance"),
        "link_url": record.get("link_url") or _get_source_link(record),
    }


def _get_source_link(item):
    source_url = item.get("source_url") or ""
    if not source_url:
        return ""

    if item.get("content_type") != "video":
        return source_url

    timestamp = int(float(item.get("timestamp") or 0.0))
    if "bilibili.com" in source_url:
        connector = "&" if "?" in source_url else "?"
        return f"{source_url}{connector}t={timestamp}"
    if "youtube.com" in source_url or "youtu.be" in source_url:
        connector = "&" if "?" in source_url else "?"
        return f"{source_url}{connector}t={timestamp}s"
    return source_url


def _find_best_subtitle_path(video_info):
    subtitle_path = video_info.get("subtitle_path") or ""
    if subtitle_path and os.path.exists(subtitle_path):
        return subtitle_path

    video_path = video_info.get("path") or ""
    if not video_path:
        return ""

    base = os.path.splitext(video_path)[0]
    candidates = sorted(Path(video_path).parent.glob(f"{Path(base).name}*.srt"))
    return str(candidates[0]) if candidates else ""


def process_video_task(video_info):
    with _tracked_task():
        if _maintenance_active():
            return

        content_id = video_info["content_id"]
        subtitle_path = _find_best_subtitle_path(video_info)
        if subtitle_path:
            db.update_content_paths(content_id, subtitle_path=subtitle_path)

        subtitle_segments = parse_srt_file(subtitle_path)
        asr_segments = []
        asr_path = ""
        if not subtitle_segments:
            asr_segments = ai_engine.transcribe_video(video_info["path"])
            if asr_segments:
                asr_path = str(
                    app_config.asr_dir / f"{_safe_filename(video_info['external_id'])}.srt"
                )
                write_srt_file(asr_segments, asr_path)
                db.update_content_paths(content_id, asr_path=asr_path)

        cap = cv2.VideoCapture(video_info["path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            fps = 25

        frame_interval = max(int(fps * app_config.frame_sample_seconds), 1)
        count = 0
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                timestamp = count / fps
                subtitle_text = align_text_to_timestamp(
                    subtitle_segments,
                    timestamp,
                    tolerance=app_config.subtitle_tolerance_seconds,
                )
                asr_text = ""
                if not subtitle_text:
                    asr_text = align_text_to_timestamp(
                        asr_segments,
                        timestamp,
                        tolerance=app_config.subtitle_tolerance_seconds,
                    )

                results = ai_engine.process_frame(
                    frame,
                    subtitle_text=subtitle_text,
                    asr_text=asr_text,
                )
                if results:
                    full_filename = (
                        f"{_safe_filename(video_info['external_id'])}_{frame_id}_full.jpg"
                    )
                    full_frame = results[0]["full_frame"]

                    if full_frame.shape[1] > 1280:
                        full_frame = cv2.resize(
                            full_frame,
                            (
                                1280,
                                int(1280 * full_frame.shape[0] / full_frame.shape[1]),
                            ),
                        )

                    cv2.imwrite(str(app_config.faces_dir / full_filename), full_frame)
                    full_web_path = f"/faces/{full_filename}"

                    for index, face_data in enumerate(results):
                        face_filename = (
                            f"{_safe_filename(video_info['external_id'])}_{frame_id}_face_{index}.jpg"
                        )
                        face_path = app_config.faces_dir / face_filename
                        cv2.imwrite(str(face_path), face_data["face_img"])

                        db.add_face(
                            content_id=content_id,
                            content_type="video",
                            platform=video_info.get("platform", ""),
                            external_id=video_info.get("external_id", ""),
                            video_id=video_info.get("external_id", ""),
                            timestamp=timestamp,
                            image_path=f"/faces/{face_filename}",
                            full_image_path=full_web_path,
                            source_url=video_info.get("url", ""),
                            embedding=face_data["embedding"],
                            description=face_data["semantic_text"],
                            visual_text=face_data["visual_text"],
                            subtitle_text=face_data["subtitle_text"],
                            asr_text=face_data["asr_text"],
                            post_text="",
                            semantic_text=face_data["semantic_text"],
                            semantic_source=face_data["semantic_source"],
                        )

            count += 1
            frame_id += 1

        cap.release()
        db.update_content_paths(content_id, local_path=video_info["path"])


def process_post_task(post_info):
    with _tracked_task():
        if _maintenance_active():
            return

        content_id = post_info["content_id"]
        post_text = post_info.get("post_text", "")
        images = post_info.get("images", [])

        for index, image_item in enumerate(images):
            image = cv2.imread(image_item["local_path"])
            if image is None:
                continue

            results = ai_engine.process_image(image, post_text=post_text)
            if not results:
                continue

            for face_index, face_data in enumerate(results):
                face_filename = (
                    f"{_safe_filename(post_info['external_id'])}_{index}_face_{face_index}.jpg"
                )
                cv2.imwrite(
                    str(app_config.faces_dir / face_filename),
                    face_data["face_img"],
                )
                db.add_face(
                    content_id=content_id,
                    content_type="post",
                    platform=post_info.get("platform", "weibo"),
                    external_id=post_info.get("external_id", ""),
                    video_id=post_info.get("external_id", ""),
                    timestamp=float(index),
                    image_path=f"/faces/{face_filename}",
                    full_image_path=image_item["web_path"],
                    source_url=post_info.get("url", ""),
                    embedding=face_data["embedding"],
                    description=face_data["semantic_text"],
                    visual_text=face_data["visual_text"],
                    subtitle_text="",
                    asr_text="",
                    post_text=face_data["post_text"],
                    semantic_text=face_data["semantic_text"],
                    semantic_source=face_data["semantic_source"],
                )

        db.update_content_paths(content_id, post_text=post_text)


def _sync_post_entry(source_record, adapter, entry):
    existing = db.get_content_by_identity(entry["platform"], entry["external_id"])
    if existing:
        return "duplicate"

    content_metadata = dict(entry.get("metadata") or {})
    content_metadata["source_sync_url"] = source_record["source_url"]
    content = db.upsert_content(
        platform=entry["platform"],
        external_id=entry["external_id"],
        content_type="post",
        title=entry.get("title", ""),
        source_url=entry.get("url", ""),
        post_text=entry.get("post_text", ""),
        metadata=content_metadata,
        collection_source_id=source_record["id"],
    )
    try:
        request_headers = entry.get("request_headers")
        if request_headers is None:
            request_headers = adapter.get_request_headers(entry)
        images = collector.download_post_images(
            entry["platform"],
            entry["external_id"],
            entry.get("image_urls") or [],
            request_headers=request_headers,
        )
    except Exception:
        return "failed"

    if not images:
        return "failed"

    process_post_task(
        {
            "content_id": content["id"],
            "platform": entry["platform"],
            "external_id": entry["external_id"],
            "post_text": entry.get("post_text", ""),
            "url": entry.get("url", ""),
            "images": images,
        }
    )
    return "imported"


def _sync_video_entry(source_record, entry):
    entry_platform = entry.get("platform") or source_record.get("platform", "")
    entry_external_id = entry.get("external_id", "")
    if entry_platform and entry_external_id:
        existing = db.get_content_by_identity(entry_platform, entry_external_id)
        if existing and db.content_has_faces(existing["id"]):
            return "duplicate"

    info = collector.download(entry["url"])
    content_platform = entry_platform or info["platform"]
    content_external_id = entry_external_id or info["external_id"]
    content = db.upsert_content(
        platform=content_platform,
        external_id=content_external_id,
        content_type="video",
        title=info.get("title") or entry.get("title", ""),
        source_url=info.get("url") or entry.get("url", ""),
        local_path=info["path"],
        subtitle_path=info.get("subtitle_path", ""),
        metadata={
            **(entry.get("metadata") or {}),
            "source_sync_url": source_record["source_url"],
        },
        collection_source_id=source_record["id"],
    )
    if db.content_has_faces(content["id"]):
        return "duplicate"

    info["content_id"] = content["id"]
    info["platform"] = content_platform
    info["external_id"] = content_external_id
    process_video_task(info)
    return "imported"


def sync_source_task(source_record, limit):
    with _tracked_task():
        if _maintenance_active():
            return {
                "status": "error",
                "message": _maintenance_message(),
            }

        metadata = source_record.get("metadata") or {}
        adapter = source_adapter_registry.select_adapter(
            source_record["source_url"],
            platform=source_record.get("platform"),
            source_type=source_record.get("source_type"),
            adapter_id=metadata.get("adapter_id"),
        )
        fetched = adapter.fetch_entries(source_record, limit=limit)
        duplicate_count = 0
        imported_count = 0
        failed_count = 0
        for entry in fetched.get("entries", []):
            content_type = str(entry.get("content_type") or "video").lower()
            if content_type == "post":
                result = _sync_post_entry(source_record, adapter, entry)
            else:
                result = _sync_video_entry(source_record, entry)
            if result == "duplicate":
                duplicate_count += 1
            elif result == "failed":
                failed_count += 1
            elif result == "imported":
                imported_count += 1

        base_stats = fetched.get("stats") or {}
        sync_stats = {
            "fetched_count": int(base_stats.get("fetched_count", len(fetched.get("entries", []))) or 0),
            "matched_count": int(base_stats.get("matched_count", len(fetched.get("entries", []))) or 0),
            "filtered_count": int(base_stats.get("filtered_count", 0) or 0),
            "imported_count": imported_count,
            "duplicate_count": duplicate_count,
            "failed_count": failed_count,
        }
        cursor = fetched.get("cursor") or {}
        mark_metadata = {
            "adapter_id": adapter.adapter_id,
            "limit": int(limit),
            "last_sync_stats": sync_stats,
        }
        if metadata.get("keywords") is not None:
            mark_metadata["keywords"] = _normalize_keywords(metadata.get("keywords", []))
        if fetched.get("user_id"):
            mark_metadata["user_id"] = fetched.get("user_id")
        if cursor.get("last_seen_post_id"):
            mark_metadata["last_seen_post_id"] = cursor.get("last_seen_post_id")
        if cursor.get("last_seen_publish_time"):
            mark_metadata["last_seen_publish_time"] = cursor.get("last_seen_publish_time")

        db.mark_source_synced(
            source_record["id"],
            title=fetched.get("title") or source_record.get("title", ""),
            metadata=mark_metadata,
        )
        return {
            "platform": fetched.get("platform", source_record.get("platform", "")),
            "title": fetched.get("title") or source_record.get("title", ""),
            "stats": sync_stats,
        }


@app.get("/api/system/status")
async def system_status():
    return _system_status_payload()


@app.get("/api/admin/status")
async def admin_status(request: Request):
    return _admin_status_payload(request)


@app.post("/api/admin/setup")
async def admin_setup(response: Response, data: dict = Body(default={})):
    if app_config.has_admin_password():
        return {"status": "error", "message": "Admin password is already set."}
    try:
        password = data.get("password", "")
        app_config.set_admin_password(password)
        _create_admin_session(response)
        return {
            "status": "success",
            "admin_initialized": True,
            "admin_authenticated": True,
        }
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/admin/login")
async def admin_login(response: Response, data: dict = Body(default={})):
    if not app_config.has_admin_password():
        return {"status": "error", "message": "Admin password is not set."}
    if not app_config.verify_admin_password(data.get("password", "")):
        return {"status": "error", "message": "Invalid admin password."}
    _create_admin_session(response)
    return {
        "status": "success",
        "admin_initialized": True,
        "admin_authenticated": True,
    }


@app.post("/api/admin/logout")
async def admin_logout(request: Request, response: Response):
    _clear_admin_session(request, response)
    return {
        "status": "success",
        "admin_initialized": app_config.has_admin_password(),
        "admin_authenticated": False,
    }


@app.post("/api/system/face-quality")
async def update_face_quality(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        payload = _validate_face_quality_payload(data)
        app_config.update_face_quality(**payload)
        return {
            "status": "success",
            "face_quality_config": app_config.face_quality_config(),
        }
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/system/config")
async def update_system_config(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        payload = _validate_runtime_config_payload(data)
        app_config.update_runtime_config(payload)
        return {
            "status": "success",
            "runtime_config": app_config.runtime_config(),
            "cluster_config": app_config.cluster_defaults(),
            "face_quality_config": app_config.face_quality_config(),
        }
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/api/source-adapters")
async def list_source_adapters():
    return {
        "status": "success",
        "adapters": source_adapter_registry.list_adapters(),
        "platform_options": source_adapter_registry.list_platform_options(),
    }


@app.post("/api/source-adapters/config")
async def upload_source_adapter_config(
    request: Request,
    file: UploadFile = File(...),
):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        config = parse_adapter_config(await file.read(), file.filename)
        adapter = source_adapter_registry.save_config(config)
        return {
            "status": "success",
            "adapter": adapter,
            "adapters": source_adapter_registry.list_adapters(),
            "platform_options": source_adapter_registry.list_platform_options(),
        }
    except (json.JSONDecodeError, SourceAdapterError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/source-adapters/enable")
async def enable_source_adapter(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        adapter_id = str(data.get("adapter_id") or "").strip()
        adapter = source_adapter_registry.set_enabled(adapter_id, bool(data.get("enabled")))
        return {
            "status": "success",
            "adapter": adapter,
            "adapters": source_adapter_registry.list_adapters(),
            "platform_options": source_adapter_registry.list_platform_options(),
        }
    except SourceAdapterError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.delete("/api/source-adapters/{adapter_id}")
async def delete_source_adapter(request: Request, adapter_id: str):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        adapter = source_adapter_registry.delete_config(adapter_id)
        return {
            "status": "success",
            "adapter": adapter,
            "adapters": source_adapter_registry.list_adapters(),
            "platform_options": source_adapter_registry.list_platform_options(),
        }
    except SourceAdapterError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/collect")
async def collect_video(request: Request, url: str, background_tasks: BackgroundTasks):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    if _maintenance_active():
        return _runtime_blocked_response()

    try:
        info = collector.download(url)
        content = db.upsert_content(
            platform=info["platform"],
            external_id=info["external_id"],
            content_type="video",
            title=info["title"],
            source_url=info["url"],
            local_path=info["path"],
            subtitle_path=info.get("subtitle_path", ""),
            metadata={"download_url": url},
        )
        info["content_id"] = content["id"]

        if db.content_has_faces(content["id"]):
            return {
                "status": "success",
                "msg": "Content already exists, skipping duplicate processing.",
                "video": info["title"],
                "duplicate": True,
            }

        background_tasks.add_task(process_video_task, info)
        return {
            "status": "success",
            "msg": "Video downloaded, semantic processing started in background.",
            "video": info["title"],
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


@app.post("/api/collect/source")
async def collect_source(
    request: Request,
    background_tasks: BackgroundTasks,
    data: dict = Body(default={}),
):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    if _maintenance_active():
        return _runtime_blocked_response()

    try:
        source_url = data.get("source_url", "").strip()
        if not source_url:
            return {"status": "error", "msg": "Missing source_url"}
        platform = str(data.get("platform") or collector.detect_platform(source_url)).lower()
        source_type = str(data.get("source_type") or (
            "weibo_user" if platform == "weibo" else "channel"
        )).lower()
        if platform == "x" and source_type == "channel":
            source_type = "x_user"
        adapter_id = data.get("adapter_id") or data.get("source_adapter_id")
        adapter = source_adapter_registry.select_adapter(
            source_url,
            platform=platform,
            source_type=source_type,
            adapter_id=adapter_id,
        )
        limit_default = adapter.default_limit or app_config.source_sync_limit
        limit = int(data.get("limit") or limit_default)
        keywords = _normalize_keywords(data.get("keywords", []))
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        metadata = {
            **metadata,
            "adapter_id": adapter.adapter_id,
            "keywords": keywords,
            "limit": limit,
        }

        source_url = adapter.normalize_source(
            source_url,
            platform=platform,
            source_type=source_type,
            metadata=metadata,
        )

        source_record = db.register_collection_source(
            platform=platform,
            source_type=source_type,
            source_url=source_url,
            title=data.get("title", ""),
            metadata=metadata,
        )
        if adapter.adapter_id == "weibo_user":
            sync_result = sync_source_task(source_record, limit)
            if sync_result.get("status") == "error":
                return sync_result
            return {
                "status": "success",
                "msg": "Source sync completed.",
                "source": db.get_collection_source(source_record["id"]),
                "sync_stats": sync_result.get("stats", {}),
            }

        background_tasks.add_task(sync_source_task, source_record, limit)
        return {
            "status": "success",
            "msg": "Source sync scheduled in background.",
            "source": source_record,
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


@app.get("/api/sources")
async def get_sources():
    return {"status": "success", "sources": db.list_collection_sources()}


@app.post("/api/source/delete")
async def delete_source(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    source_id = data.get("source_id")
    delete_data = bool(data.get("delete_data"))
    if source_id is None:
        return {"status": "error", "message": "Missing source_id"}

    try:
        source_id = int(source_id)
        with _maintenance_guard():
            source = db.get_collection_source(source_id)
            if source is None:
                return {"status": "error", "message": "Collection source not found."}

            if delete_data:
                result = db.delete_source_with_data(source_id)
            else:
                deleted_source = db.delete_collection_source(source_id)
                result = {
                    "source": source,
                    "deleted_source": deleted_source,
                    "deleted_contents": 0,
                    "deleted_faces": 0,
                    "deleted_people": 0,
                    "deleted_files": 0,
                    "cleared_cluster_snapshot": False,
                    "unresolved_legacy_items": 0,
                }

        return {
            "status": "success",
            **result,
            "cluster_config": app_config.cluster_defaults(),
            "face_quality_config": app_config.face_quality_config(),
            "has_cluster_snapshot": db.has_cluster_snapshot(),
            "sources": db.list_collection_sources(),
        }
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/system/reset")
async def reset_system(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    preserve_cookies = bool(data.get("preserve_cookies", True))
    try:
        with _maintenance_guard():
            preserved_files = _preserved_cookie_files(preserve_cookies)
            removed_files_count = _remove_runtime_storage()
            if not preserve_cookies:
                removed_files_count += _remove_cookie_files()
            removed_files_count += db.reset_database_state()
            app_config.reset_to_defaults()
            _clear_all_admin_sessions()
            _rebuild_runtime_state()

        return {
            "status": "success",
            "reset": True,
            "preserved_files": preserved_files,
            "removed_files_count": removed_files_count,
            "cluster_config": app_config.cluster_defaults(),
            "face_quality_config": app_config.face_quality_config(),
            "has_cluster_snapshot": db.has_cluster_snapshot(),
            "sources": db.list_collection_sources(),
        }
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/api/import/post")
async def import_post(
    request: Request,
    background_tasks: BackgroundTasks,
    data: dict = Body(default={}),
):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    if _maintenance_active():
        return _runtime_blocked_response()

    try:
        url = data.get("url", "").strip()
        provided_text = data.get("post_text", "").strip()
        image_urls = [
            item.strip() for item in data.get("image_urls", []) if item.strip()
        ]
        platform = data.get("platform") or collector.detect_platform(
            url or "https://weibo.com"
        )

        extracted = collector.extract_post_metadata(url) if url else {}
        external_id = (
            data.get("external_id")
            or extracted.get("external_id")
            or collector.derive_external_id(url, provided_text or json.dumps(image_urls))
        )
        post_text = provided_text or extracted.get("post_text", "")
        merged_image_urls = image_urls or extracted.get("image_urls", [])

        if not merged_image_urls:
            return {
                "status": "error",
                "msg": "No image URLs were provided or extracted.",
            }

        content = db.upsert_content(
            platform=platform,
            external_id=external_id,
            content_type="post",
            title=data.get("title") or extracted.get("title", ""),
            source_url=url or extracted.get("source_url", ""),
            post_text=post_text,
            metadata={"import_mode": "post"},
        )

        if db.content_has_faces(content["id"]):
            return {
                "status": "success",
                "msg": "Post already exists, skipping duplicate processing.",
                "duplicate": True,
            }

        images = collector.download_post_images(platform, external_id, merged_image_urls)
        background_tasks.add_task(
            process_post_task,
            {
                "content_id": content["id"],
                "platform": platform,
                "external_id": external_id,
                "post_text": post_text,
                "url": url or extracted.get("source_url", ""),
                "images": images,
            },
        )
        return {
            "status": "success",
            "msg": "Post import scheduled in background.",
            "images": len(images),
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

    descriptions = [
        record.get("semantic_text") or record.get("description", "")
        for record in all_records
    ]
    ranked = ai_engine.rank_texts_by_similarity(query, descriptions)

    search_results = []
    for index, similarity in ranked:
        if similarity < app_config.text_threshold:
            continue

        record = dict(all_records[index])
        record["score"] = round(float(similarity), 4)
        record["metric"] = "semantic"
        record["link_url"] = _get_source_link(record)
        search_results.append(_serialize_face(record))

    search_results.sort(key=lambda item: item["score"], reverse=True)
    return {"status": "success", "results": search_results[: app_config.text_top_k]}


@app.post("/api/search/image")
async def search_image(metric: str = "cosine", file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Invalid image file"}

        target_embedding, _ = ai_engine.get_face_embedding(
            img,
            face_quality_config={"enabled": False},
        )
        if target_embedding is None:
            return {"status": "error", "message": "No face detected"}

        if metric.lower() == "euclidean":
            matches = db.search_faces_by_metric(
                target_embedding,
                metric="euclidean",
                top_k=app_config.image_top_k,
            )
        else:
            matches = db.search_faces_by_embedding(
                target_embedding,
                top_k=app_config.image_top_k,
                min_score=app_config.image_cosine_threshold,
            )

        results = []
        for face in matches:
            face["link_url"] = _get_source_link(face)
            results.append(_serialize_face(face))

        return {"status": "success", "metric": metric.lower(), "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/cluster/run")
async def run_clustering(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        previous_config = app_config.cluster_defaults()
        algorithm = data.get("algorithm", previous_config["algorithm"])
        metric = data.get("metric", previous_config["metric"])
        eps = float(data.get("eps", previous_config["eps"]))
        min_samples = int(data.get("min_samples", previous_config["min_samples"]))
        mode = data.get("mode", "incremental")
        app_config.update_cluster_defaults(
            algorithm=algorithm,
            metric=metric,
            eps=eps,
            min_samples=min_samples,
        )
        result = perform_clustering(
            db,
            algorithm=algorithm,
            metric=metric,
            eps=eps,
            min_samples=min_samples,
            mode=mode,
            snapshot_config=previous_config,
        )
        if result.get("status") == "success":
            result["has_cluster_snapshot"] = db.has_cluster_snapshot()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/cluster/rollback")
async def rollback_clustering(request: Request):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    try:
        restored = db.restore_cluster_snapshot()
        if restored is None:
            return {"status": "error", "message": "No cluster snapshot available."}

        snapshot_config = restored.get("cluster_config") or {}
        if snapshot_config:
            app_config.update_cluster_defaults(
                algorithm=snapshot_config.get("algorithm"),
                metric=snapshot_config.get("metric"),
                eps=snapshot_config.get("eps"),
                min_samples=snapshot_config.get("min_samples"),
            )

        return {
            "status": "success",
            "restored_faces": restored.get("restored_faces", 0),
            "restored_people": restored.get("restored_people", 0),
            "cluster_config": app_config.cluster_defaults(),
            "has_cluster_snapshot": db.has_cluster_snapshot(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/cluster/evaluate")
async def cluster_evaluation(data: dict = Body(default={})):
    try:
        defaults = app_config.cluster_defaults()
        eps = float(data.get("eps", defaults["eps"]))
        min_samples = int(data.get("min_samples", defaults["min_samples"]))
        algorithms = data.get("algorithms") or ["dbscan", "hdbscan", "optics"]
        metrics = data.get("metrics") or ["cosine", "euclidean"]
        cluster_results = evaluate_clustering(
            db,
            algorithms=algorithms,
            metrics=metrics,
            eps=eps,
            min_samples=min_samples,
        )
        image_results = compare_image_search_metrics(db, top_k=5)
        return {
            "status": "success",
            "cluster_results": cluster_results.get("results", []),
            "image_results": image_results.get("results", []),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/people")
async def get_people():
    people = db.get_clustered_people()
    unassigned = [_serialize_face(face) for face in db.get_unassigned_faces()]
    return {
        "status": "success",
        "people": people,
        "unassigned": unassigned,
        "cluster_config": app_config.cluster_defaults(),
        "has_cluster_snapshot": db.has_cluster_snapshot(),
    }


@app.get("/api/person/{person_id}")
async def get_person_details(person_id: int):
    timeline = [_serialize_face(item) for item in db.get_person_timeline(person_id)]
    for item in timeline:
        item["link_url"] = _get_source_link(item)
    return {
        "status": "success",
        "timeline": timeline,
        "cluster_config": app_config.cluster_defaults(),
    }


@app.post("/api/person/rename")
async def rename_person(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    person_id = data.get("person_id")
    new_name = data.get("name")
    if person_id is None or not new_name:
        return {"status": "error", "message": "Missing required fields"}

    db.rename_person(person_id, new_name)
    return {"status": "success"}


@app.post("/api/face/reassign")
async def reassign_face(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
    face_id = data.get("face_id")
    target_person_id = data.get("target_person_id")
    if face_id is None or target_person_id is None:
        return {"status": "error", "message": "Missing required fields"}

    db.reassign_face(face_id, target_person_id)
    return {"status": "success"}


@app.post("/api/person/merge")
async def merge_person(request: Request, data: dict = Body(default={})):
    auth_error = _require_admin(request)
    if auth_error:
        return auth_error
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
