import fnmatch
import importlib.util
import json
import re
from pathlib import Path

from core.config import app_config
from core.x_adapter import XUserCollector


ADAPTER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")


class SourceAdapterError(ValueError):
    pass


class BaseSourceAdapter:
    adapter_id = ""
    display_name = ""
    platform = ""
    source_types = ("channel",)
    builtin = True
    configurable = False
    enabled = True
    default_limit = None
    url_patterns = ()

    def __init__(self, collector=None, config=None):
        self.collector = collector
        self.config = config or {}

    def match(self, source_url, platform=None, source_type=None):
        platform = str(platform or "").lower()
        source_type = str(source_type or "").lower()
        if source_type and source_type not in self.source_types:
            return False
        if platform and platform not in {self.platform, self.adapter_id}:
            return False
        return True

    def normalize_source(self, source_url, platform=None, source_type=None, metadata=None):
        return str(source_url or "").strip()

    def fetch_entries(self, source_record, limit):
        raise NotImplementedError

    def get_request_headers(self, entry):
        return {}

    def to_dict(self):
        return {
            "adapter_id": self.adapter_id,
            "display_name": self.display_name or self.adapter_id,
            "platform": self.platform,
            "source_types": list(self.source_types),
            "builtin": bool(self.builtin),
            "configurable": bool(self.configurable),
            "enabled": bool(self.enabled),
            "default_limit": self.default_limit,
            "url_patterns": list(self.url_patterns),
        }


class YtDlpSourceAdapter(BaseSourceAdapter):
    adapter_id = "yt_dlp"
    display_name = "yt-dlp"
    platform = "yt_dlp"
    source_types = ("channel",)
    supported_platforms = ("bilibili", "youtube", "generic")

    def match(self, source_url, platform=None, source_type=None):
        source_type = str(source_type or "channel").lower()
        platform = str(platform or "").lower()
        if source_type not in self.source_types:
            return False
        if not platform:
            platform = self.collector.detect_platform(source_url)
        return platform in self.supported_platforms or platform == self.adapter_id

    def fetch_entries(self, source_record, limit):
        platform = str(source_record.get("platform") or "").lower()
        if platform == self.adapter_id or not platform:
            platform = self.collector.detect_platform(source_record["source_url"])
        fetched = self.collector.fetch_source_entries(
            source_record["source_url"],
            limit=limit,
            platform=platform,
            source_type=source_record.get("source_type") or "channel",
            metadata=source_record.get("metadata") or {},
        )
        entries = []
        for entry in fetched.get("entries", []):
            normalized = dict(entry)
            normalized.setdefault("platform", platform)
            normalized.setdefault("content_type", "video")
            entries.append(normalized)
        return {
            "platform": fetched.get("platform") or platform,
            "title": fetched.get("title") or source_record.get("title", ""),
            "source_url": fetched.get("source_url") or source_record["source_url"],
            "entries": entries,
            "stats": fetched.get(
                "stats",
                {
                    "fetched_count": len(entries),
                    "matched_count": len(entries),
                    "filtered_count": 0,
                },
            ),
            "cursor": fetched.get("cursor", {}),
        }

    def to_dict(self):
        item = super().to_dict()
        item["platforms"] = list(self.supported_platforms)
        item["platform_options"] = [
            {
                "adapter_id": self.adapter_id,
                "platform": platform,
                "source_type": "channel",
                "label": platform,
            }
            for platform in self.supported_platforms
        ]
        return item


class WeiboUserSourceAdapter(BaseSourceAdapter):
    adapter_id = "weibo_user"
    display_name = "Weibo user"
    platform = "weibo"
    source_types = ("weibo_user",)

    @property
    def default_limit(self):
        return app_config.platform_sync_limit("weibo")

    def normalize_source(self, source_url, platform=None, source_type=None, metadata=None):
        return self.collector.normalize_source_url(
            source_url,
            platform="weibo",
            source_type="weibo_user",
        )

    def fetch_entries(self, source_record, limit):
        metadata = source_record.get("metadata") or {}
        keywords = self.collector.normalize_keywords(metadata.get("keywords", []))
        fetched = self.collector.fetch_source_entries(
            source_record["source_url"],
            limit=limit,
            platform="weibo",
            source_type="weibo_user",
            metadata={"keywords": keywords},
        )
        entries = []
        for entry in fetched.get("entries", []):
            normalized = dict(entry)
            normalized.setdefault("platform", "weibo")
            normalized.setdefault("content_type", "post")
            entries.append(normalized)
        fetched["entries"] = entries
        fetched.setdefault("source_url", source_record["source_url"])
        fetched.setdefault("stats", {})
        fetched.setdefault("cursor", {})
        return fetched

    def get_request_headers(self, entry):
        return self.collector.get_platform_request_headers(
            "weibo",
            referer=(entry or {}).get("url"),
        )

    def to_dict(self):
        item = super().to_dict()
        item["platform_options"] = [
            {
                "adapter_id": self.adapter_id,
                "platform": self.platform,
                "source_type": "weibo_user",
                "label": "weibo",
            }
        ]
        return item


class XUserSourceAdapter(BaseSourceAdapter):
    adapter_id = "x_user"
    display_name = "X user"
    platform = "x"
    source_types = ("x_user",)

    def __init__(self, collector=None, config=None, x_adapter=None):
        super().__init__(collector=collector, config=config)
        self.x_adapter = x_adapter or XUserCollector()

    @property
    def default_limit(self):
        return app_config.platform_sync_limit("x")

    def normalize_source(self, source_url, platform=None, source_type=None, metadata=None):
        return self.x_adapter.normalize_user_source(source_url)["canonical_url"]

    def fetch_entries(self, source_record, limit):
        metadata = source_record.get("metadata") or {}
        keywords = self.x_adapter.normalize_keywords(metadata.get("keywords", []))
        fetched = self.x_adapter.fetch_user_posts(
            source_record["source_url"],
            limit=limit,
            keywords=keywords,
        )
        entries = []
        for entry in fetched.get("entries", []):
            normalized = dict(entry)
            normalized.setdefault("platform", "x")
            normalized.setdefault("content_type", "post")
            entries.append(normalized)
        fetched["entries"] = entries
        fetched.setdefault("source_url", source_record["source_url"])
        fetched.setdefault("stats", {})
        fetched.setdefault("cursor", {})
        return fetched

    def to_dict(self):
        item = super().to_dict()
        item["platform_options"] = [
            {
                "adapter_id": self.adapter_id,
                "platform": self.platform,
                "source_type": "x_user",
                "label": "x",
            }
        ]
        return item


class ConfigYtDlpSourceAdapter(YtDlpSourceAdapter):
    builtin = False
    configurable = True

    def __init__(self, collector=None, config=None):
        super().__init__(collector=collector, config=config)
        config = config or {}
        self.adapter_id = config["adapter_id"]
        self.display_name = config.get("display_name") or self.adapter_id
        self.platform = config["platform"]
        self.source_types = tuple(config["source_types"])
        self.enabled = bool(config.get("enabled", True))
        self.default_limit = config.get("default_limit")
        self.url_patterns = tuple(config.get("url_patterns") or [])

    def match(self, source_url, platform=None, source_type=None):
        if not self.enabled:
            return False
        source_type = str(source_type or "").lower()
        platform = str(platform or "").lower()
        if source_type and source_type not in self.source_types:
            return False
        if platform and platform not in {self.platform, self.adapter_id}:
            return False
        if self.url_patterns and source_url:
            return any(fnmatch.fnmatch(source_url, pattern) for pattern in self.url_patterns)
        return True

    def fetch_entries(self, source_record, limit):
        fetched = self.collector.fetch_source_entries(
            source_record["source_url"],
            limit=limit,
            platform=self.platform,
            source_type=source_record.get("source_type") or self.source_types[0],
            metadata=source_record.get("metadata") or {},
        )
        entries = []
        for entry in fetched.get("entries", []):
            normalized = dict(entry)
            normalized.setdefault("platform", self.platform)
            normalized.setdefault("content_type", "video")
            entries.append(normalized)
        return {
            "platform": self.platform,
            "title": fetched.get("title") or source_record.get("title", ""),
            "source_url": fetched.get("source_url") or source_record["source_url"],
            "entries": entries,
            "stats": fetched.get(
                "stats",
                {
                    "fetched_count": len(entries),
                    "matched_count": len(entries),
                    "filtered_count": 0,
                },
            ),
            "cursor": fetched.get("cursor", {}),
        }

    def to_dict(self):
        item = BaseSourceAdapter.to_dict(self)
        item["platform_options"] = [
            {
                "adapter_id": self.adapter_id,
                "platform": self.platform,
                "source_type": source_type,
                "label": self.display_name,
            }
            for source_type in self.source_types
        ]
        return item


class ModuleSourceAdapter(BaseSourceAdapter):
    builtin = False
    configurable = True

    def __init__(self, wrapped, config):
        self.wrapped = wrapped
        self.config = config
        self.adapter_id = config["adapter_id"]
        self.display_name = config.get("display_name") or self.adapter_id
        self.platform = config["platform"]
        self.source_types = tuple(config["source_types"])
        self.enabled = bool(config.get("enabled", True))
        self.default_limit = config.get("default_limit")
        self.url_patterns = tuple(config.get("url_patterns") or [])

    def match(self, source_url, platform=None, source_type=None):
        if not self.enabled:
            return False
        if hasattr(self.wrapped, "match"):
            return bool(self.wrapped.match(source_url, platform=platform, source_type=source_type))
        return ConfigYtDlpSourceAdapter.match(self, source_url, platform, source_type)

    def normalize_source(self, source_url, platform=None, source_type=None, metadata=None):
        if hasattr(self.wrapped, "normalize_source"):
            return self.wrapped.normalize_source(
                source_url,
                platform=platform,
                source_type=source_type,
                metadata=metadata,
            )
        return str(source_url or "").strip()

    def fetch_entries(self, source_record, limit):
        return self.wrapped.fetch_entries(source_record, limit)

    def get_request_headers(self, entry):
        if hasattr(self.wrapped, "get_request_headers"):
            return self.wrapped.get_request_headers(entry)
        return {}

    def to_dict(self):
        item = BaseSourceAdapter.to_dict(self)
        item["module_configured"] = True
        item["platform_options"] = [
            {
                "adapter_id": self.adapter_id,
                "platform": self.platform,
                "source_type": source_type,
                "label": self.display_name,
            }
            for source_type in self.source_types
        ]
        return item


def parse_adapter_config(raw, filename="adapter.json"):
    suffix = Path(filename or "").suffix.lower()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SourceAdapterError("YAML adapter configs require PyYAML to be installed.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SourceAdapterError("Adapter config must be a JSON object.")
    return validate_adapter_config(data)


def validate_adapter_config(config):
    config = dict(config or {})
    adapter_id = str(config.get("adapter_id") or "").strip()
    platform = str(config.get("platform") or "").strip().lower()
    display_name = str(config.get("display_name") or adapter_id).strip()
    source_types = config.get("source_types")
    if not ADAPTER_ID_RE.match(adapter_id):
        raise SourceAdapterError("adapter_id must be 2-64 letters, numbers, dashes, or underscores.")
    if not platform:
        raise SourceAdapterError("platform is required.")
    if not display_name:
        raise SourceAdapterError("display_name is required.")
    if not isinstance(source_types, list) or not source_types:
        raise SourceAdapterError("source_types must be a non-empty list.")
    normalized_types = []
    for item in source_types:
        cleaned = str(item or "").strip().lower()
        if not cleaned:
            raise SourceAdapterError("source_types cannot contain empty values.")
        if cleaned not in normalized_types:
            normalized_types.append(cleaned)

    url_patterns = config.get("url_patterns") or []
    if not isinstance(url_patterns, list):
        raise SourceAdapterError("url_patterns must be a list.")
    settings = config.get("settings") or {}
    if not isinstance(settings, dict):
        raise SourceAdapterError("settings must be an object.")
    module = str(config.get("module") or "").strip()
    default_limit = config.get("default_limit")
    if default_limit is not None:
        try:
            default_limit = int(default_limit)
        except (TypeError, ValueError) as exc:
            raise SourceAdapterError("default_limit must be an integer.") from exc
        if not 1 <= default_limit <= 100:
            raise SourceAdapterError("default_limit must be between 1 and 100.")

    return {
        "adapter_id": adapter_id,
        "display_name": display_name,
        "platform": platform,
        "enabled": bool(config.get("enabled", True)),
        "module": module,
        "source_types": normalized_types,
        "url_patterns": [str(item).strip() for item in url_patterns if str(item).strip()],
        "default_limit": default_limit,
        "settings": settings,
    }


class SourceAdapterRegistry:
    def __init__(self, collector):
        self.collector = collector
        self.adapters = {}
        self.reload()

    def reload(self):
        self.adapters = {}
        self._register(WeiboUserSourceAdapter(self.collector))
        self._register(XUserSourceAdapter(self.collector))
        self._register(YtDlpSourceAdapter(self.collector))
        self._load_config_adapters()

    def _register(self, adapter):
        if adapter.adapter_id in self.adapters:
            raise SourceAdapterError(f"Duplicate source adapter id: {adapter.adapter_id}")
        self.adapters[adapter.adapter_id] = adapter

    def _load_config_adapters(self):
        app_config.ensure_dirs()
        for path in sorted(app_config.source_adapter_config_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                config = validate_adapter_config(json.load(f))
            if config["adapter_id"] in self.adapters:
                raise SourceAdapterError(
                    f"Config adapter id conflicts with a built-in adapter: {config['adapter_id']}"
                )
            adapter = self._build_config_adapter(config)
            self._register(adapter)

    def _build_config_adapter(self, config):
        if config.get("module"):
            return ModuleSourceAdapter(self._load_module_adapter(config), config)
        return ConfigYtDlpSourceAdapter(self.collector, config)

    def _load_module_adapter(self, config):
        module_ref = config["module"]
        if ":" not in module_ref:
            raise SourceAdapterError("module must use the format module_name:ClassName.")
        module_name, class_name = module_ref.split(":", 1)
        if not ADAPTER_ID_RE.match(module_name):
            raise SourceAdapterError("module name must be a simple local Python module name.")
        module_path = (app_config.source_adapter_code_dir / f"{module_name}.py").resolve()
        try:
            module_path.relative_to(app_config.source_adapter_code_dir.resolve())
        except ValueError as exc:
            raise SourceAdapterError("module must resolve inside the local adapter module directory.") from exc
        if not module_path.exists():
            raise SourceAdapterError(f"Adapter module file not found: {module_path}")

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        adapter_cls = getattr(module, class_name, None)
        if adapter_cls is None:
            raise SourceAdapterError(f"Adapter class not found: {class_name}")
        try:
            return adapter_cls(config=config, collector=self.collector)
        except TypeError:
            return adapter_cls(config, self.collector)

    def list_adapters(self):
        return [adapter.to_dict() for adapter in self.adapters.values()]

    def list_platform_options(self):
        options = []
        for adapter in self.adapters.values():
            if not adapter.enabled:
                continue
            item = adapter.to_dict()
            adapter_options = item.get("platform_options") or [
                {
                    "adapter_id": adapter.adapter_id,
                    "platform": adapter.platform,
                    "source_type": adapter.source_types[0],
                    "label": adapter.display_name or adapter.adapter_id,
                }
            ]
            options.extend(adapter_options)
        return options

    def get(self, adapter_id):
        return self.adapters.get(str(adapter_id or ""))

    def select_adapter(self, source_url, platform=None, source_type=None, adapter_id=None):
        if adapter_id:
            adapter = self.get(adapter_id)
            if adapter is None or not adapter.enabled:
                raise SourceAdapterError(f"Source adapter is not available: {adapter_id}")
            if not adapter.match(source_url, platform=platform, source_type=source_type):
                raise SourceAdapterError(f"Source adapter does not support this source: {adapter_id}")
            return adapter

        configured = [item for item in self.adapters.values() if item.configurable]
        builtins = [item for item in self.adapters.values() if not item.configurable]
        for adapter in configured + builtins:
            if adapter.enabled and adapter.match(source_url, platform=platform, source_type=source_type):
                return adapter
        raise SourceAdapterError("No source adapter supports this source.")

    def save_config(self, config):
        config = validate_adapter_config(config)
        if config["adapter_id"] in self.adapters and self.adapters[config["adapter_id"]].builtin:
            raise SourceAdapterError("adapter_id conflicts with a built-in adapter.")
        app_config.ensure_dirs()
        path = app_config.source_adapter_config_dir / f"{config['adapter_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self.reload()
        return self.adapters[config["adapter_id"]].to_dict()

    def set_enabled(self, adapter_id, enabled):
        adapter = self.get(adapter_id)
        if adapter is None:
            raise SourceAdapterError("Source adapter not found.")
        if adapter.builtin:
            raise SourceAdapterError("Built-in adapters cannot be enabled or disabled.")
        path = app_config.source_adapter_config_dir / f"{adapter.adapter_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            config = validate_adapter_config(json.load(f))
        config["enabled"] = bool(enabled)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self.reload()
        return self.adapters[adapter_id].to_dict()

    def delete_config(self, adapter_id):
        adapter = self.get(adapter_id)
        if adapter is None:
            raise SourceAdapterError("Source adapter not found.")
        if adapter.builtin:
            raise SourceAdapterError("Built-in adapters cannot be deleted.")
        path = app_config.source_adapter_config_dir / f"{adapter.adapter_id}.json"
        if path.exists():
            path.unlink()
        self.reload()
        return adapter.to_dict()
