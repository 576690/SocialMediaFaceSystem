import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import requests
from lxml import etree

from core.config import app_config


WEIBO_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class WeiboSpiderRequestError(RuntimeError):
    pass


class WeiboSpiderParseError(RuntimeError):
    pass


class WeiboUserUnavailableError(RuntimeError):
    pass


class WeiboUserCollector:
    def __init__(
        self,
        cookie_file=None,
        retries=None,
        timeout_seconds=None,
        index_parser_cls=None,
        page_parser_cls=None,
    ):
        self.cookie_file = Path(cookie_file or app_config.weibo_cookie_path)
        self.retries = int(retries or app_config.platform_retry_count("weibo"))
        self.timeout_seconds = int(
            timeout_seconds or app_config.platform_timeout_seconds("weibo")
        )
        self.index_parser_cls = index_parser_cls
        self.page_parser_cls = page_parser_cls

    def normalize_keywords(self, keywords):
        if keywords is None:
            return []

        raw_items = []
        if isinstance(keywords, str):
            raw_items.extend(re.split(r"[\n,，]+", keywords))
        else:
            for item in keywords:
                raw_items.extend(re.split(r"[\n,，]+", str(item or "")))

        normalized = []
        for item in raw_items:
            cleaned = str(item or "").strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def load_cookie(self):
        if not app_config.platform_auth_enabled("weibo"):
            raise RuntimeError("Weibo cookies are disabled in the current configuration.")
        if not self.cookie_file.exists():
            raise RuntimeError(
                f"Weibo cookies file is missing: {self.cookie_file}"
            )

        cookie = self.cookie_file.read_text(encoding="utf-8").strip()
        if not cookie:
            raise RuntimeError("Weibo cookies file is empty.")
        if cookie.lower().startswith("cookie:"):
            cookie = cookie.split(":", 1)[1].strip()
        return cookie

    def build_request_headers(self, referer=None, cookie=None):
        headers = {
            "Cookie": cookie or self.load_cookie(),
            "User-Agent": WEIBO_USER_AGENT,
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _normalize_source_url_input(self, source):
        source = str(source or "").strip()
        if not source:
            raise RuntimeError("Missing Weibo user source.")

        if source.isdigit():
            return f"https://weibo.cn/{source}"

        parsed = urlparse(source)
        if parsed.scheme:
            return source

        host_prefixes = (
            "weibo.com/",
            "www.weibo.com/",
            "weibo.cn/",
            "www.weibo.cn/",
        )
        if source.lower().startswith(host_prefixes):
            return f"https://{source.lstrip('/')}"
        return f"https://weibo.cn/{source.lstrip('/')}"

    def _canonical_user_url(self, user_uri):
        normalized = str(user_uri or "").strip("/")
        if normalized.startswith("u/") and normalized[2:].isdigit():
            return f"https://weibo.cn/{normalized[2:]}"
        return f"https://weibo.cn/{normalized}"

    def _paging_user_uri(self, user_id):
        normalized = str(user_id or "").strip("/")
        if normalized.startswith("u/") and normalized[2:].isdigit():
            return normalized[2:]
        return normalized

    def normalize_user_source(self, source):
        source = self._normalize_source_url_input(source)
        parsed = urlparse(source)
        path = parsed.path.strip("/")
        if not path:
            raise RuntimeError(f"Invalid Weibo user source: {source}")

        parts = [part for part in path.split("/") if part]
        if not parts:
            raise RuntimeError(f"Invalid Weibo user source: {source}")

        if parts[0] in {"u", "n"} and len(parts) >= 2:
            user_uri = f"{parts[0]}/{parts[1]}"
        elif parts[0].isdigit():
            user_uri = f"u/{parts[0]}"
        else:
            user_uri = parts[0]

        return {
            "user_uri": user_uri,
            "canonical_url": self._canonical_user_url(user_uri),
        }

    def matches_keywords(self, text, keywords):
        keywords = self.normalize_keywords(keywords)
        if not keywords:
            return True
        haystack = str(text or "").lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    def normalize_picture_urls(self, picture_field):
        empty_markers = {"", "无", "鏃?"}
        if picture_field is None:
            return []
        if isinstance(picture_field, list):
            urls = picture_field
        else:
            urls = [item.strip() for item in str(picture_field).split(",")]
        return [item for item in urls if item and item not in empty_markers]

    def _request_weibo_page(self, cookie, url):
        try:
            response = requests.get(
                url,
                headers=self.build_request_headers(cookie=cookie),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise WeiboSpiderRequestError(
                f"Failed to fetch Weibo page: {url}"
            ) from exc

        if response.status_code != 200:
            raise WeiboSpiderRequestError(
                f"Weibo returned HTTP {response.status_code} for {url}"
            )
        if not response.content:
            raise WeiboSpiderRequestError(f"Weibo returned an empty response for {url}")

        selector = etree.HTML(response.content)
        if selector is None:
            raise WeiboSpiderParseError(f"Failed to parse Weibo page HTML for {url}")
        return selector

    def _build_fallback_user(self, selector, user_id):
        nickname = ""
        title_list = selector.xpath("//title/text()")
        if title_list:
            title = str(title_list[0] or "").strip()
            if title.endswith("的资料"):
                nickname = title[:-3]
            elif title.endswith("的微博"):
                nickname = title[:-3]

        if not nickname:
            header_bits = [
                str(item or "").strip()
                for item in selector.xpath("//div[@class='u']//div[@class='ut']/text()")
            ]
            nickname = "".join(bit for bit in header_bits if bit).strip()

        if not nickname:
            nickname = str(user_id)

        return SimpleNamespace(id=str(user_id), nickname=nickname)

    def _extract_user_id_from_selector(self, selector, fallback_uri):
        hrefs = selector.xpath("//div[@class='u']//a/@href")
        for href in hrefs:
            clean_href = str(href or "").split("?", 1)[0].strip()
            if not clean_href.endswith("/info"):
                continue
            user_id = clean_href.lstrip("/")[:-5].strip("/")
            if user_id.startswith("u/") and user_id[2:].isdigit():
                return user_id[2:]
            if user_id:
                return user_id

        fallback = str(fallback_uri or "").strip("/")
        if fallback.startswith("u/") and fallback[2:].isdigit():
            return fallback[2:]
        return fallback

    def _install_weibo_spider_patches(self):
        try:
            from weibo_spider.parser import IndexParser, PageParser
            from weibo_spider.parser import info_parser as info_parser_module
            from weibo_spider.parser import index_parser as index_parser_module
            from weibo_spider.parser import page_parser as page_parser_module
            from weibo_spider.parser import util as util_module
        except ImportError as exc:
            raise RuntimeError(
                "weibo-spider is not installed. Install the weibo-spider package first."
            ) from exc

        def patched_handle_html(cookie, url):
            return self._request_weibo_page(cookie, url)

        util_module.handle_html = patched_handle_html
        index_parser_module.handle_html = patched_handle_html
        info_parser_module.handle_html = patched_handle_html
        page_parser_module.handle_html = patched_handle_html

        collector = self

        class CompatIndexParser(IndexParser):
            def _get_user_id(self):
                if self.selector is None:
                    raise WeiboSpiderParseError("Weibo user page is empty or unreadable.")
                return collector._extract_user_id_from_selector(
                    self.selector,
                    self.user_uri,
                )

            def get_user(self):
                if self.selector is None:
                    raise WeiboSpiderParseError("Weibo user page is empty or unreadable.")

                user_id = self._get_user_id()
                if not user_id:
                    raise WeiboUserUnavailableError(
                        "Failed to resolve a Weibo user ID from the source page."
                    )

                try:
                    user = info_parser_module.InfoParser(
                        self.cookie,
                        user_id,
                    ).extract_user_info()
                except SystemExit as exc:
                    raise WeiboSpiderRequestError(
                        "Weibo cookies are invalid or expired."
                    ) from exc

                if user is None:
                    user = collector._build_fallback_user(self.selector, user_id)

                user.id = str(user_id)

                user_info = self.selector.xpath("//div[@class='tip2']/*/text()")
                if len(user_info) >= 3:
                    user.weibo_num = util_module.string_to_int(user_info[0][3:-1])
                    user.following = util_module.string_to_int(user_info[1][3:-1])
                    user.followers = util_module.string_to_int(user_info[2][3:-1])
                return user

            def get_page_num(self):
                if self.selector is None:
                    raise WeiboSpiderParseError("Weibo user page is empty or unreadable.")

                mp_list = self.selector.xpath("//input[@name='mp']")
                if not mp_list:
                    return 1

                value = mp_list[0].attrib.get("value", "").strip()
                if not value:
                    return 1

                try:
                    return int(value)
                except ValueError as exc:
                    raise WeiboSpiderParseError(
                        f"Failed to parse Weibo page count: {value}"
                    ) from exc

        return CompatIndexParser, PageParser

    def _resolve_parsers(self):
        if self.index_parser_cls is not None and self.page_parser_cls is not None:
            return self.index_parser_cls, self.page_parser_cls

        compat_index_parser_cls, compat_page_parser_cls = (
            self._install_weibo_spider_patches()
        )
        return (
            self.index_parser_cls or compat_index_parser_cls,
            self.page_parser_cls or compat_page_parser_cls,
        )

    def fetch_user_posts(self, source, limit=None, keywords=None):
        if not app_config.platform_enabled("weibo"):
            raise RuntimeError("Weibo collection is disabled in the current configuration.")
        limit = max(int(limit or app_config.platform_sync_limit("weibo")), 1)
        keywords = self.normalize_keywords(keywords)
        normalized_source = self.normalize_user_source(source)
        cookie = self.load_cookie()
        index_parser_cls, page_parser_cls = self._resolve_parsers()

        try:
            index_parser = index_parser_cls(cookie, normalized_source["user_uri"])
            user = index_parser.get_user()
            page_num = int(index_parser.get_page_num() or 1)
        except (WeiboSpiderRequestError, WeiboSpiderParseError) as exc:
            raise RuntimeError(
                "Failed to load Weibo user data. Check whether the cookies are still valid, "
                "whether Weibo is blocking the request, or whether the page could not be parsed."
            ) from exc
        except SystemExit as exc:
            raise RuntimeError(
                "Failed to load Weibo user data. Check whether the cookies are still valid."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Weibo user data. Check whether the user source is valid."
            ) from exc

        if not user or not getattr(user, "id", ""):
            raise RuntimeError(
                "Weibo user is unavailable or inaccessible. Check whether the user link is valid or the account is visible."
            )

        resolved_user_id = self._paging_user_uri(user.id)
        canonical_url = self._canonical_user_url(resolved_user_id)
        user_config = {
            "user_uri": resolved_user_id,
            "since_date": "2000-01-01",
            "end_date": "now",
        }
        entries = []
        seen_ids = []
        fetched_count = 0
        matched_count = 0
        filtered_count = 0
        newest_post_id = ""
        newest_publish_time = ""

        for page in range(1, max(page_num, 1) + 1):
            try:
                parser = page_parser_cls(cookie, user_config, page, 1)
                page_result = parser.get_one_page(seen_ids)
            except (WeiboSpiderRequestError, WeiboSpiderParseError) as exc:
                raise RuntimeError(
                    "Failed to load Weibo posts. Check whether the cookies are still valid, "
                    "whether Weibo is blocking the request, or whether the page could not be parsed."
                ) from exc
            except SystemExit as exc:
                raise RuntimeError(
                    "Failed to load Weibo posts. Check whether the cookies are still valid."
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load Weibo posts. Check whether the user source is valid."
                ) from exc

            if not page_result:
                raise RuntimeError(
                    "Failed to load Weibo posts. Check whether the page could not be parsed."
                )

            weibos, seen_ids, to_continue = page_result
            if not weibos:
                if not to_continue:
                    break
                continue

            for weibo in weibos:
                fetched_count += 1
                if not newest_post_id:
                    newest_post_id = str(weibo.id)
                    newest_publish_time = str(weibo.publish_time or "")

                image_urls = self.normalize_picture_urls(weibo.original_pictures)
                if not image_urls or not self.matches_keywords(weibo.content, keywords):
                    filtered_count += 1
                    continue

                matched_count += 1
                entries.append(
                    {
                        "platform": "weibo",
                        "external_id": str(weibo.id),
                        "title": str(weibo.content or "")[:80].strip() or str(weibo.id),
                        "url": f"https://weibo.cn/comment/{weibo.id}",
                        "post_text": str(weibo.content or "").strip(),
                        "image_urls": image_urls,
                        "publish_time": str(weibo.publish_time or ""),
                        "metadata": {
                            "publish_place": str(getattr(weibo, "publish_place", "") or ""),
                            "publish_tool": str(getattr(weibo, "publish_tool", "") or ""),
                            "up_num": int(getattr(weibo, "up_num", 0) or 0),
                            "retweet_num": int(getattr(weibo, "retweet_num", 0) or 0),
                            "comment_num": int(getattr(weibo, "comment_num", 0) or 0),
                            "article_url": str(getattr(weibo, "article_url", "") or ""),
                            "user_id": str(user.id),
                            "user_nickname": str(getattr(user, "nickname", "") or ""),
                        },
                    }
                )
                if len(entries) >= limit:
                    break

            if len(entries) >= limit or not to_continue:
                break

        return {
            "platform": "weibo",
            "title": str(getattr(user, "nickname", "") or user.id),
            "user_id": str(user.id),
            "source_url": canonical_url,
            "entries": entries,
            "stats": {
                "fetched_count": fetched_count,
                "matched_count": matched_count,
                "filtered_count": filtered_count,
            },
            "cursor": {
                "last_seen_post_id": newest_post_id,
                "last_seen_publish_time": newest_publish_time,
            },
        }
