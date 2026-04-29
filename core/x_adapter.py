import re
from pathlib import Path
from urllib.parse import urlparse

from core.config import app_config


X_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
X_RESERVED_PATHS = {
    "home",
    "explore",
    "i",
    "intent",
    "messages",
    "notifications",
    "search",
    "settings",
    "share",
}


class XUserCollector:
    def __init__(self, bearer_token_file=None, client_cls=None, client=None):
        self.bearer_token_file = Path(
            bearer_token_file or app_config.x_bearer_token_path
        )
        self.client_cls = client_cls
        self.client = client

    def normalize_keywords(self, keywords):
        if keywords is None:
            return []
        raw_items = keywords if isinstance(keywords, list) else str(keywords).splitlines()
        normalized = []
        for item in raw_items:
            for part in str(item or "").replace("，", ",").split(","):
                cleaned = part.strip()
                if cleaned and cleaned not in normalized:
                    normalized.append(cleaned)
        return normalized

    def matches_keywords(self, text, keywords):
        keywords = self.normalize_keywords(keywords)
        if not keywords:
            return True
        haystack = str(text or "").lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    def load_bearer_token(self):
        if not app_config.platform_auth_enabled("x"):
            raise RuntimeError(
                "X bearer token authentication is disabled in the current configuration."
            )
        if not self.bearer_token_file.exists():
            raise RuntimeError(
                f"X bearer token file is missing: {self.bearer_token_file}"
            )
        token = self.bearer_token_file.read_text(encoding="utf-8").strip()
        if token.lower().startswith("bearer "):
            token = token.split(" ", 1)[1].strip()
        if not token:
            raise RuntimeError("X bearer token file is empty.")
        return token

    def _get_client(self):
        if self.client is not None:
            return self.client
        if self.client_cls is None:
            try:
                import tweepy
            except ImportError as exc:
                raise RuntimeError(
                    "tweepy is not installed. Install the tweepy package first."
                ) from exc
            self.client_cls = tweepy.Client
        self.client = self.client_cls(
            bearer_token=self.load_bearer_token(),
            wait_on_rate_limit=True,
        )
        return self.client

    def _extract_username(self, source):
        source = str(source or "").strip()
        if not source:
            raise RuntimeError("Missing X user source.")
        if source.startswith("@"):
            source = source[1:]

        parsed = urlparse(source)
        if not parsed.scheme and "/" in source:
            parsed = urlparse(f"https://{source}")

        if parsed.scheme:
            host = parsed.netloc.lower()
            if host not in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}:
                raise RuntimeError(f"Invalid X user source: {source}")
            parts = [part for part in parsed.path.split("/") if part]
            if not parts or parts[0].lower() in X_RESERVED_PATHS:
                raise RuntimeError(f"Invalid X user source: {source}")
            username = parts[0]
        else:
            username = source

        username = username.strip().lstrip("@")
        if not X_USERNAME_RE.match(username):
            raise RuntimeError(f"Invalid X username: {username}")
        return username

    def normalize_user_source(self, source):
        username = self._extract_username(source)
        return {
            "username": username,
            "canonical_url": f"https://x.com/{username}",
        }

    def _value(self, item, key, default=None):
        if item is None:
            return default
        if isinstance(item, dict):
            return item.get(key, default)
        data = getattr(item, "data", None)
        if isinstance(data, dict) and key in data:
            return data.get(key, default)
        return getattr(item, key, default)

    def _response_meta(self, response):
        meta = getattr(response, "meta", None)
        return meta if isinstance(meta, dict) else {}

    def _response_includes(self, response):
        includes = getattr(response, "includes", None)
        return includes if isinstance(includes, dict) else {}

    def _media_urls_by_key(self, response):
        media_by_key = {}
        for item in self._response_includes(response).get("media", []) or []:
            media_key = self._value(item, "media_key")
            media_type = self._value(item, "type")
            media_url = self._value(item, "url")
            if media_key and media_type == "photo" and media_url:
                media_by_key[str(media_key)] = str(media_url)
        return media_by_key

    def _tweet_media_urls(self, tweet, media_by_key):
        attachments = self._value(tweet, "attachments", {}) or {}
        media_keys = attachments.get("media_keys", []) if isinstance(attachments, dict) else []
        urls = []
        seen = set()
        for media_key in media_keys:
            url = media_by_key.get(str(media_key))
            if not url or url in seen:
                continue
            urls.append(url)
            seen.add(url)
        return urls

    def _public_metrics(self, tweet):
        metrics = self._value(tweet, "public_metrics", {}) or {}
        return metrics if isinstance(metrics, dict) else {}

    def fetch_user_posts(self, source, limit=None, keywords=None):
        if not app_config.platform_enabled("x"):
            raise RuntimeError("X collection is disabled in the current configuration.")
        limit = max(int(limit or app_config.platform_sync_limit("x")), 1)
        keywords = self.normalize_keywords(keywords)
        normalized_source = self.normalize_user_source(source)
        username = normalized_source["username"]
        client = self._get_client()

        try:
            user_response = client.get_user(
                username=username,
                user_fields=["id", "name", "username", "profile_image_url"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load X user data. Check whether the bearer token is valid "
                "and whether the user source is accessible."
            ) from exc

        user = getattr(user_response, "data", None)
        user_id = self._value(user, "id")
        if not user or not user_id:
            raise RuntimeError(
                "X user is unavailable or inaccessible. Check whether the username is valid."
            )

        entries = []
        fetched_count = 0
        matched_count = 0
        filtered_count = 0
        newest_tweet_id = ""
        newest_publish_time = ""
        pagination_token = None

        while len(entries) < limit:
            request_limit = max(5, min(100, limit - len(entries)))
            try:
                tweets_response = client.get_users_tweets(
                    id=user_id,
                    max_results=request_limit,
                    pagination_token=pagination_token,
                    expansions=["attachments.media_keys"],
                    media_fields=[
                        "url",
                        "preview_image_url",
                        "type",
                        "alt_text",
                        "width",
                        "height",
                    ],
                    tweet_fields=["attachments", "created_at", "public_metrics"],
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load X posts. Check whether the bearer token has access "
                    "to the user timeline endpoint and has remaining quota."
                ) from exc

            tweets = list(getattr(tweets_response, "data", None) or [])
            media_by_key = self._media_urls_by_key(tweets_response)
            if not tweets:
                break

            for tweet in tweets:
                fetched_count += 1
                tweet_id = str(self._value(tweet, "id", "") or "")
                created_at = str(self._value(tweet, "created_at", "") or "")
                text = str(self._value(tweet, "text", "") or "")
                if not newest_tweet_id:
                    newest_tweet_id = tweet_id
                    newest_publish_time = created_at

                image_urls = self._tweet_media_urls(tweet, media_by_key)
                if not image_urls or not self.matches_keywords(text, keywords):
                    filtered_count += 1
                    continue

                matched_count += 1
                metrics = self._public_metrics(tweet)
                entries.append(
                    {
                        "platform": "x",
                        "content_type": "post",
                        "external_id": tweet_id,
                        "title": text[:80].strip() or tweet_id,
                        "url": f"https://x.com/{username}/status/{tweet_id}",
                        "post_text": text.strip(),
                        "image_urls": image_urls,
                        "publish_time": created_at,
                        "metadata": {
                            "user_id": str(user_id),
                            "username": username,
                            "name": str(self._value(user, "name", "") or ""),
                            "retweet_count": int(metrics.get("retweet_count", 0) or 0),
                            "reply_count": int(metrics.get("reply_count", 0) or 0),
                            "like_count": int(metrics.get("like_count", 0) or 0),
                            "quote_count": int(metrics.get("quote_count", 0) or 0),
                        },
                    }
                )
                if len(entries) >= limit:
                    break

            meta = self._response_meta(tweets_response)
            pagination_token = meta.get("next_token")
            if not pagination_token:
                break

        return {
            "platform": "x",
            "title": str(self._value(user, "name", "") or username),
            "user_id": str(user_id),
            "source_url": normalized_source["canonical_url"],
            "entries": entries,
            "stats": {
                "fetched_count": fetched_count,
                "matched_count": matched_count,
                "filtered_count": filtered_count,
            },
            "cursor": {
                "last_seen_post_id": newest_tweet_id,
                "last_seen_publish_time": newest_publish_time,
            },
        }
