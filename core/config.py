import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path


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
    },
    "collection": {
        "source_sync_limit": 10,
        "subtitle_languages": ["zh-Hans", "zh-CN", "zh", "en"],
        "subtitle_formats": "srt/best",
        "weibo_cookie_enabled": True,
        "weibo_source_sync_limit": 10,
        "weibo_timeout_seconds": 15,
        "weibo_retry_count": 3,
    },
    "transcription": {
        "enabled": True,
        "preferred_backend": "faster_whisper",
        "model_size": "medium",
    },
    "face_quality": {
        "enabled": True,
        "min_face_size": 56,
        "min_laplacian_var": 80.0,
        "max_pose_deviation": 0.35,
    },
    "clustering": {
        "algorithm": "dbscan",
        "metric": "cosine",
        "eps": 0.4,
        "min_samples": 2,
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
        self.system_config_path = self.storage_dir / "system_config.json"
        self.weibo_cookie_path = self.storage_dir / "weibo_cookies.txt"
        self.load()

    def ensure_dirs(self):
        for directory in (
            self.storage_dir,
            self.videos_dir,
            self.faces_dir,
            self.content_dir,
            self.asr_dir,
            self.test_artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def load(self):
        self.ensure_dirs()
        if self.system_config_path.exists():
            with open(self.system_config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.data = _merge_dicts(DEFAULT_CONFIG, loaded)
        else:
            self.data = deepcopy(DEFAULT_CONFIG)
            self.save()

    def save(self):
        self.ensure_dirs()
        with open(self.system_config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def reset_to_defaults(self):
        self.data = deepcopy(DEFAULT_CONFIG)
        self.save()

    def managed_runtime_dirs(self):
        return (
            self.videos_dir,
            self.faces_dir,
            self.content_dir,
            self.asr_dir,
            self.test_artifacts_dir,
        )

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
    def source_sync_limit(self):
        return int(self.data["collection"]["source_sync_limit"])

    @property
    def subtitle_languages(self):
        return list(self.data["collection"]["subtitle_languages"])

    @property
    def subtitle_formats(self):
        return self.data["collection"]["subtitle_formats"]

    @property
    def weibo_cookie_enabled(self):
        return bool(self.data["collection"]["weibo_cookie_enabled"])

    @property
    def weibo_source_sync_limit(self):
        return int(self.data["collection"]["weibo_source_sync_limit"])

    @property
    def weibo_timeout_seconds(self):
        return int(self.data["collection"]["weibo_timeout_seconds"])

    @property
    def weibo_retry_count(self):
        return int(self.data["collection"]["weibo_retry_count"])

    @property
    def transcription_enabled(self):
        return bool(self.data["transcription"]["enabled"])

    @property
    def transcription_backend(self):
        return self.data["transcription"]["preferred_backend"]

    @property
    def transcription_model_size(self):
        return self.data["transcription"]["model_size"]

    def face_quality_config(self):
        return {
            "enabled": bool(self.data["face_quality"]["enabled"]),
            "min_face_size": int(self.data["face_quality"]["min_face_size"]),
            "min_laplacian_var": float(self.data["face_quality"]["min_laplacian_var"]),
            "max_pose_deviation": float(self.data["face_quality"]["max_pose_deviation"]),
        }

    def update_face_quality(
        self,
        enabled=None,
        min_face_size=None,
        min_laplacian_var=None,
        max_pose_deviation=None,
    ):
        face_quality = self.data["face_quality"]
        if enabled is not None:
            face_quality["enabled"] = bool(enabled)
        if min_face_size is not None:
            face_quality["min_face_size"] = int(min_face_size)
        if min_laplacian_var is not None:
            face_quality["min_laplacian_var"] = float(min_laplacian_var)
        if max_pose_deviation is not None:
            face_quality["max_pose_deviation"] = float(max_pose_deviation)
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
