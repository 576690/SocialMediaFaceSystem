import hashlib
import hmac
import json
import secrets
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path


PLATFORM_CONFIG_DEFAULTS = {
    "weibo": {
        "enabled": True,
        "auth_enabled": True,
        "sync_limit": 10,
        "sync_interval_minutes": 0,
        "timeout_seconds": 15,
        "retry_count": 3,
    },
    "bilibili": {
        "enabled": True,
        "auth_enabled": True,
        "sync_limit": 10,
        "sync_interval_minutes": 0,
        "timeout_seconds": 20,
        "retry_count": 5,
        "impersonate": "chrome",
        "referer": "https://www.bilibili.com/",
    },
    "youtube": {
        "enabled": True,
        "auth_enabled": True,
        "sync_limit": 10,
        "sync_interval_minutes": 0,
        "timeout_seconds": 20,
        "retry_count": 3,
    },
    "x": {
        "enabled": True,
        "auth_enabled": True,
        "sync_limit": 10,
        "sync_interval_minutes": 0,
        "timeout_seconds": 20,
        "retry_count": 3,
    },
    "generic": {
        "enabled": True,
        "sync_limit": 10,
        "sync_interval_minutes": 0,
        "timeout_seconds": 20,
        "retry_count": 3,
    },
}


def _default_platforms():
    return deepcopy(PLATFORM_CONFIG_DEFAULTS)


DEFAULT_CONFIG = {
    "processing": {
        "frame_sample_seconds": 2.0,
        "subtitle_tolerance_seconds": 1.5,
    },
    "search": {
        "text_threshold": 0.15,
        "image_cosine_threshold": 0.4,
        "image_top_k": 10,
        "text_top_k": 20,
        "semantic_model_id": "Qwen/Qwen3-Embedding-0.6B",
        "semantic_model_prompt_name": "query",
        "semantic_model_mode": "standard",
        "semantic_corpus_style": "structured_zh",
    },
    "collection": {
        "source_sync_limit": 10,
        "subtitle_languages": ["zh-Hans", "zh-CN", "zh", "en"],
        "subtitle_formats": "srt/best",
        "weibo_cookie_enabled": True,
        "weibo_source_sync_limit": 10,
        "weibo_timeout_seconds": 15,
        "weibo_retry_count": 3,
        "bilibili_cookie_enabled": True,
        "bilibili_impersonate": "chrome",
        "bilibili_referer": "https://www.bilibili.com/",
        "platforms": _default_platforms(),
    },
    "transcription": {
        "enabled": True,
        "preferred_backend": "faster_whisper",
        "model_size": "medium",
        "initial_prompt": "以下内容来自中文社交媒体视频，请优先准确识别人名、品牌名、活动名和口语表达。",
        "hotwords": ["直播", "采访", "活动", "品牌", "发布会", "路演"],
    },
    "vision": {
        "vlm_model_id": "microsoft/Florence-2-large-ft",
        "release_vlm_after_task": True,
        "release_text_encoder_before_vlm": True,
        "caption_style": "retrieval_keywords",
        "caption_language": "zh",
        "caption_include_ocr_hint": True,
    },
    "face_quality": {
        "enabled": True,
        "min_face_size": 56,
        "min_face_ratio": 0.035,
        "min_laplacian_var": 80.0,
        "max_pose_deviation": 0.35,
        "blur_eval_size": 96,
    },
    "clustering": {
        "algorithm": "dbscan",
        "metric": "cosine",
        "eps": 0.4,
        "min_samples": 2,
    },
    "admin": {
        "password_hash": "",
        "salt": "",
        "iterations": 200_000,
    },
}


def _merge_dicts(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class AppConfig:
    root_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data: dict = field(default_factory=lambda: deepcopy(DEFAULT_CONFIG))

    def __post_init__(self):
        self.storage_dir = self.root_dir / "storage"
        self.videos_dir = self.storage_dir / "videos"
        self.faces_dir = self.storage_dir / "faces"
        self.content_dir = self.storage_dir / "content"
        self.asr_dir = self.storage_dir / "artifacts" / "asr"
        self.test_artifacts_dir = self.storage_dir / "test_artifacts"
        self.source_adapter_config_dir = self.storage_dir / "source_adapters"
        self.source_adapter_code_dir = self.storage_dir / "source_adapter_modules"
        self.system_config_path = self.storage_dir / "system_config.json"
        self.weibo_cookie_path = self.storage_dir / "weibo_cookies.txt"
        self.bilibili_cookie_path = self.storage_dir / "bilibili_cookies.txt"
        self.x_bearer_token_path = self.storage_dir / "x_bearer_token.txt"
        self.load()

    def ensure_dirs(self):
        expected_adapter_config_dir = self.storage_dir / "source_adapters"
        expected_adapter_code_dir = self.storage_dir / "source_adapter_modules"
        if (
            not hasattr(self, "source_adapter_config_dir")
            or Path(self.source_adapter_config_dir) != expected_adapter_config_dir
        ):
            self.source_adapter_config_dir = self.storage_dir / "source_adapters"
        if (
            not hasattr(self, "source_adapter_code_dir")
            or Path(self.source_adapter_code_dir) != expected_adapter_code_dir
        ):
            self.source_adapter_code_dir = self.storage_dir / "source_adapter_modules"
        expected_x_bearer_token_path = self.storage_dir / "x_bearer_token.txt"
        if (
            not hasattr(self, "x_bearer_token_path")
            or Path(self.x_bearer_token_path) != expected_x_bearer_token_path
        ):
            self.x_bearer_token_path = self.storage_dir / "x_bearer_token.txt"
        for directory in (
            self.storage_dir,
            self.videos_dir,
            self.faces_dir,
            self.content_dir,
            self.asr_dir,
            self.test_artifacts_dir,
            self.source_adapter_config_dir,
            self.source_adapter_code_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def load(self):
        self.ensure_dirs()
        loaded = {}
        if self.system_config_path.exists():
            with open(self.system_config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.data = _merge_dicts(DEFAULT_CONFIG, loaded)
        else:
            self.data = deepcopy(DEFAULT_CONFIG)
            self.save()
        self._normalize_collection_config(loaded.get("collection") if isinstance(loaded, dict) else None)

    def save(self):
        self.ensure_dirs()
        with open(self.system_config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def reset_to_defaults(self):
        self.data = deepcopy(DEFAULT_CONFIG)
        self._normalize_collection_config({"platforms": self.data["collection"]["platforms"]})
        self.save()

    def has_admin_password(self):
        admin = self.data.get("admin", {})
        return bool(admin.get("password_hash") and admin.get("salt"))

    def set_admin_password(self, password):
        cleaned = str(password or "")
        if len(cleaned) < 6:
            raise ValueError("Admin password must be at least 6 characters.")

        iterations = int(self.data.get("admin", {}).get("iterations") or 200_000)
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            cleaned.encode("utf-8"),
            bytes.fromhex(salt),
            iterations,
        ).hex()
        self.data["admin"] = {
            "password_hash": password_hash,
            "salt": salt,
            "iterations": iterations,
        }
        self.save()

    def verify_admin_password(self, password):
        admin = self.data.get("admin", {})
        password_hash = admin.get("password_hash") or ""
        salt = admin.get("salt") or ""
        if not password_hash or not salt:
            return False

        try:
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                str(password or "").encode("utf-8"),
                bytes.fromhex(salt),
                int(admin.get("iterations") or 200_000),
            ).hex()
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(candidate, password_hash)

    def managed_runtime_dirs(self, include_videos=True):
        directories = [
            self.faces_dir,
            self.content_dir,
            self.asr_dir,
            self.test_artifacts_dir,
        ]
        if include_videos:
            directories.insert(0, self.videos_dir)
        return tuple(directories)

    def resolve_managed_path(self, raw_path):
        raw = str(raw_path or "").strip()
        if not raw:
            return None

        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(self.storage_dir.resolve())
            except ValueError:
                return None
            return candidate

        normalized = raw.lstrip("/")
        if normalized.startswith("faces/"):
            return self.storage_dir / normalized
        if normalized.startswith("content/"):
            return self.storage_dir / normalized

        fallback = self.storage_dir / normalized
        try:
            fallback.resolve().relative_to(self.storage_dir.resolve())
        except ValueError:
            return None
        return fallback

    @property
    def frame_sample_seconds(self):
        return float(self.data["processing"]["frame_sample_seconds"])

    @property
    def subtitle_tolerance_seconds(self):
        return float(self.data["processing"]["subtitle_tolerance_seconds"])

    @property
    def text_threshold(self):
        return float(self.data["search"]["text_threshold"])

    @property
    def image_cosine_threshold(self):
        return float(self.data["search"]["image_cosine_threshold"])

    @property
    def image_top_k(self):
        return int(self.data["search"]["image_top_k"])

    @property
    def text_top_k(self):
        return int(self.data["search"]["text_top_k"])

    @property
    def semantic_model_id(self):
        return self.data["search"]["semantic_model_id"]

    @property
    def semantic_model_prompt_name(self):
        return self.data["search"]["semantic_model_prompt_name"]

    @property
    def semantic_model_mode(self):
        return self.data["search"]["semantic_model_mode"]

    @property
    def semantic_corpus_style(self):
        return self.data["search"]["semantic_corpus_style"]

    @property
    def source_sync_limit(self):
        return self.platform_sync_limit("generic")

    @property
    def subtitle_languages(self):
        return list(self.data["collection"]["subtitle_languages"])

    @property
    def subtitle_formats(self):
        return self.data["collection"]["subtitle_formats"]

    @property
    def weibo_cookie_enabled(self):
        return self.platform_auth_enabled("weibo")

    @property
    def weibo_source_sync_limit(self):
        return self.platform_sync_limit("weibo")

    @property
    def weibo_timeout_seconds(self):
        return self.platform_timeout_seconds("weibo")

    @property
    def weibo_retry_count(self):
        return self.platform_retry_count("weibo")

    @property
    def bilibili_cookie_enabled(self):
        return self.platform_auth_enabled("bilibili")

    @property
    def bilibili_impersonate(self):
        return self.platform_config("bilibili").get("impersonate", "chrome")

    @property
    def bilibili_referer(self):
        return self.platform_config("bilibili").get(
            "referer",
            "https://www.bilibili.com/",
        )

    def _normalize_collection_config(self, loaded_collection=None):
        collection = self.data.setdefault("collection", {})
        if not isinstance(collection, dict):
            self.data["collection"] = deepcopy(DEFAULT_CONFIG["collection"])
            collection = self.data["collection"]

        loaded_collection = loaded_collection if isinstance(loaded_collection, dict) else {}
        loaded_platforms = loaded_collection.get("platforms")
        has_loaded_platforms = isinstance(loaded_platforms, dict)

        platforms = _merge_dicts(
            PLATFORM_CONFIG_DEFAULTS,
            collection.get("platforms") if isinstance(collection.get("platforms"), dict) else {},
        )

        if not has_loaded_platforms:
            default_collection = DEFAULT_CONFIG["collection"]
            source_sync_limit = collection.get(
                "source_sync_limit",
                default_collection["source_sync_limit"],
            )
            for platform in ("bilibili", "youtube", "x", "generic"):
                platforms[platform]["sync_limit"] = source_sync_limit

            platforms["weibo"]["auth_enabled"] = collection.get(
                "weibo_cookie_enabled",
                default_collection["weibo_cookie_enabled"],
            )
            platforms["weibo"]["sync_limit"] = collection.get(
                "weibo_source_sync_limit",
                default_collection["weibo_source_sync_limit"],
            )
            platforms["weibo"]["timeout_seconds"] = collection.get(
                "weibo_timeout_seconds",
                default_collection["weibo_timeout_seconds"],
            )
            platforms["weibo"]["retry_count"] = collection.get(
                "weibo_retry_count",
                default_collection["weibo_retry_count"],
            )
            platforms["bilibili"]["auth_enabled"] = collection.get(
                "bilibili_cookie_enabled",
                default_collection["bilibili_cookie_enabled"],
            )
            platforms["bilibili"]["impersonate"] = collection.get(
                "bilibili_impersonate",
                default_collection["bilibili_impersonate"],
            )
            platforms["bilibili"]["referer"] = collection.get(
                "bilibili_referer",
                default_collection["bilibili_referer"],
            )

        collection["platforms"] = platforms
        self._sync_legacy_collection_fields()

    def _sync_legacy_collection_fields(self):
        collection = self.data.setdefault("collection", {})
        platforms = collection.get("platforms") or {}
        generic = platforms.get("generic", PLATFORM_CONFIG_DEFAULTS["generic"])
        weibo = platforms.get("weibo", PLATFORM_CONFIG_DEFAULTS["weibo"])
        bilibili = platforms.get("bilibili", PLATFORM_CONFIG_DEFAULTS["bilibili"])
        collection["source_sync_limit"] = int(generic.get("sync_limit", 10))
        collection["weibo_cookie_enabled"] = bool(weibo.get("auth_enabled", True))
        collection["weibo_source_sync_limit"] = int(weibo.get("sync_limit", 10))
        collection["weibo_timeout_seconds"] = int(weibo.get("timeout_seconds", 15))
        collection["weibo_retry_count"] = int(weibo.get("retry_count", 3))
        collection["bilibili_cookie_enabled"] = bool(
            bilibili.get("auth_enabled", True)
        )
        collection["bilibili_impersonate"] = str(
            bilibili.get("impersonate", "chrome") or "chrome"
        )
        collection["bilibili_referer"] = str(
            bilibili.get("referer", "https://www.bilibili.com/")
            or "https://www.bilibili.com/"
        )

    def collection_platforms_config(self):
        self._normalize_collection_config(
            {"platforms": self.data.get("collection", {}).get("platforms", {})}
        )
        return deepcopy(self.data["collection"]["platforms"])

    def platform_config(self, platform):
        platform = str(platform or "generic").lower()
        platforms = self.collection_platforms_config()
        return platforms.get(platform) or platforms["generic"]

    def platform_enabled(self, platform):
        return bool(self.platform_config(platform).get("enabled", True))

    def platform_auth_enabled(self, platform):
        return bool(self.platform_config(platform).get("auth_enabled", False))

    def platform_sync_limit(self, platform):
        return int(self.platform_config(platform).get("sync_limit", 10))

    def platform_sync_interval_minutes(self, platform):
        return int(self.platform_config(platform).get("sync_interval_minutes", 0))

    def platform_timeout_seconds(self, platform):
        return int(self.platform_config(platform).get("timeout_seconds", 20))

    def platform_retry_count(self, platform):
        return int(self.platform_config(platform).get("retry_count", 3))

    @property
    def transcription_enabled(self):
        return bool(self.data["transcription"]["enabled"])

    @property
    def transcription_backend(self):
        return self.data["transcription"]["preferred_backend"]

    @property
    def transcription_model_size(self):
        return self.data["transcription"]["model_size"]

    @property
    def transcription_initial_prompt(self):
        return self.data["transcription"]["initial_prompt"]

    @property
    def transcription_hotwords(self):
        return list(self.data["transcription"]["hotwords"])

    @property
    def vlm_model_id(self):
        return self.data["vision"]["vlm_model_id"]

    @property
    def release_vlm_after_task(self):
        return bool(self.data["vision"]["release_vlm_after_task"])

    @property
    def release_text_encoder_before_vlm(self):
        return bool(self.data["vision"]["release_text_encoder_before_vlm"])

    @property
    def caption_style(self):
        return self.data["vision"]["caption_style"]

    @property
    def caption_language(self):
        return self.data["vision"]["caption_language"]

    @property
    def caption_include_ocr_hint(self):
        return bool(self.data["vision"]["caption_include_ocr_hint"])

    def face_quality_config(self):
        face_quality = self.data["face_quality"]
        return {
            "enabled": bool(face_quality["enabled"]),
            "min_face_size": int(face_quality["min_face_size"]),
            "min_face_ratio": float(face_quality.get("min_face_ratio", 0.035)),
            "min_laplacian_var": float(face_quality["min_laplacian_var"]),
            "max_pose_deviation": float(face_quality["max_pose_deviation"]),
            "blur_eval_size": int(face_quality.get("blur_eval_size", 96)),
        }

    def runtime_config(self):
        return {
            "processing": {
                "frame_sample_seconds": self.frame_sample_seconds,
                "subtitle_tolerance_seconds": self.subtitle_tolerance_seconds,
            },
            "search": {
                "text_threshold": self.text_threshold,
                "image_cosine_threshold": self.image_cosine_threshold,
                "image_top_k": self.image_top_k,
                "text_top_k": self.text_top_k,
                "semantic_model_id": self.semantic_model_id,
                "semantic_model_prompt_name": self.semantic_model_prompt_name,
                "semantic_model_mode": self.semantic_model_mode,
                "semantic_corpus_style": self.semantic_corpus_style,
            },
            "collection": {
                "source_sync_limit": self.source_sync_limit,
                "weibo_cookie_enabled": self.weibo_cookie_enabled,
                "weibo_source_sync_limit": self.weibo_source_sync_limit,
                "weibo_timeout_seconds": self.weibo_timeout_seconds,
                "weibo_retry_count": self.weibo_retry_count,
                "bilibili_cookie_enabled": self.bilibili_cookie_enabled,
                "bilibili_impersonate": self.bilibili_impersonate,
                "bilibili_referer": self.bilibili_referer,
                "platforms": self.collection_platforms_config(),
            },
            "transcription": {
                "enabled": self.transcription_enabled,
                "preferred_backend": self.transcription_backend,
                "model_size": self.transcription_model_size,
                "initial_prompt": self.transcription_initial_prompt,
                "hotwords": self.transcription_hotwords,
            },
            "vision": {
                "vlm_model_id": self.vlm_model_id,
                "release_vlm_after_task": self.release_vlm_after_task,
                "release_text_encoder_before_vlm": self.release_text_encoder_before_vlm,
                "caption_style": self.caption_style,
                "caption_language": self.caption_language,
                "caption_include_ocr_hint": self.caption_include_ocr_hint,
            },
            "clustering": self.cluster_defaults(),
            "face_quality": self.face_quality_config(),
        }

    def update_runtime_config(self, payload):
        for section, values in (payload or {}).items():
            if section not in self.data or not isinstance(values, dict):
                continue
            if section == "admin":
                continue
            target = self.data[section]
            if not isinstance(target, dict):
                continue
            for key, value in values.items():
                if key in target:
                    target[key] = value
        self._normalize_collection_config(
            {"platforms": self.data.get("collection", {}).get("platforms", {})}
        )
        self.save()

    def update_face_quality(
        self,
        enabled=None,
        min_face_size=None,
        min_face_ratio=None,
        min_laplacian_var=None,
        max_pose_deviation=None,
        blur_eval_size=None,
    ):
        face_quality = self.data["face_quality"]
        if enabled is not None:
            face_quality["enabled"] = bool(enabled)
        if min_face_size is not None:
            face_quality["min_face_size"] = int(min_face_size)
        if min_face_ratio is not None:
            face_quality["min_face_ratio"] = float(min_face_ratio)
        if min_laplacian_var is not None:
            face_quality["min_laplacian_var"] = float(min_laplacian_var)
        if max_pose_deviation is not None:
            face_quality["max_pose_deviation"] = float(max_pose_deviation)
        if blur_eval_size is not None:
            face_quality["blur_eval_size"] = int(blur_eval_size)
        self.save()

    def cluster_defaults(self):
        return {
            "algorithm": self.data["clustering"]["algorithm"],
            "metric": self.data["clustering"]["metric"],
            "eps": float(self.data["clustering"]["eps"]),
            "min_samples": int(self.data["clustering"]["min_samples"]),
        }

    def update_cluster_defaults(
        self,
        algorithm=None,
        metric=None,
        eps=None,
        min_samples=None,
    ):
        clustering = self.data["clustering"]
        if algorithm:
            clustering["algorithm"] = algorithm
        if metric:
            clustering["metric"] = metric
        if eps is not None:
            clustering["eps"] = float(eps)
        if min_samples is not None:
            clustering["min_samples"] = int(min_samples)
        self.save()


app_config = AppConfig()
