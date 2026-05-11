import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse

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


class WeiboAuthenticationError(RuntimeError):
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
            raise RuntimeError("微博 Cookie 认证当前未启用。")
        if not self.cookie_file.exists():
            raise RuntimeError(
                f"缺少微博 Cookie 文件：{self.cookie_file}"
            )

        cookie = self.cookie_file.read_text(encoding="utf-8").strip()
        if not cookie:
            raise RuntimeError("微博 Cookie 文件为空。")
        if cookie.lower().startswith("cookie:"):
            cookie = cookie.split(":", 1)[1].strip()
        if "\t" in cookie:
            cookie = self._parse_netscape_cookie_file(cookie)
        return cookie

    def _parse_netscape_cookie_file(self, cookie_text):
        values = []
        for line in str(cookie_text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            name = parts[5].strip()
            value = parts[6].strip()
            if name:
                values.append(f"{name}={value}")
        if not values:
            raise RuntimeError("微博 Cookie 文件格式无效。")
        return "; ".join(values)

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
            raise RuntimeError("缺少微博用户源。")

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
            raise RuntimeError(f"微博用户源无效：{source}")

        parts = [part for part in path.split("/") if part]
        if not parts:
            raise RuntimeError(f"微博用户源无效：{source}")

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

    def _extract_single_post_id(self, source):
        source = str(source or "").strip()
        if not source:
            raise RuntimeError("缺少微博链接。")
        if source.isdigit():
            return source

        parsed = urlparse(source if urlparse(source).scheme else f"https://{source}")
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if not parts:
            raise RuntimeError(f"无法从微博链接解析微博 ID：{source}")

        for marker in ("comment", "detail", "status", "repost"):
            if marker in parts:
                marker_index = parts.index(marker)
                if len(parts) > marker_index + 1:
                    return parts[marker_index + 1].split("?", 1)[0].strip()

        if parts[0] in {"u", "n"} and len(parts) <= 2:
            raise RuntimeError(f"微博链接不是单条图文详情页：{source}")

        candidate = parts[-1].split("?", 1)[0].strip()
        if not re.match(r"^[A-Za-z0-9_-]{4,}$", candidate):
            raise RuntimeError(f"无法从微博链接解析微博 ID：{source}")
        return candidate

    def _single_post_url(self, post_id):
        return f"https://weibo.cn/comment/{post_id}"

    def _absolute_weibo_url(self, href):
        href = str(href or "").strip()
        if not href:
            return ""
        if href.startswith("//"):
            return f"https:{href}"
        return urljoin("https://weibo.cn", href)

    def _large_picture_url(self, image_url):
        image_url = self._absolute_weibo_url(image_url)
        if not image_url:
            return ""
        for marker in ("/thumb180/", "/wap180/", "/orj360/", "/mw690/", "/bmiddle/"):
            image_url = image_url.replace(marker, "/large/")
        return image_url

    def _find_single_post_node(self, selector, post_id):
        target_id = str(post_id or "").strip()
        for node in selector.xpath("//div[@class='c' and @id]"):
            node_id = str(node.attrib.get("id") or "").strip()
            if node_id == target_id or node_id == f"M_{target_id}":
                return node

        nodes = selector.xpath("body/div[@class='c' and @id][1]")
        return nodes[0] if nodes else None

    def _extract_single_post_text(self, node):
        text_items = node.xpath(".//span[@class='ctt']//text()")
        if not text_items:
            text_items = node.xpath(".//text()")
        text = " ".join(str(item or "").strip() for item in text_items)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lstrip(":：").strip()

    def _extract_single_post_publish_time(self, node):
        values = node.xpath(".//span[@class='ct']/text()")
        return str(values[0] or "").strip() if values else ""

    def _extract_single_post_image_urls(self, cookie, node):
        image_urls = []

        for image_url in node.xpath(".//img/@src"):
            normalized = self._large_picture_url(image_url)
            if normalized:
                image_urls.append(normalized)

        for href in node.xpath(".//a/@href"):
            href = self._absolute_weibo_url(href)
            if "/mblog/picAll/" not in href:
                continue
            try:
                selector = self._request_weibo_page(cookie, href)
            except Exception:
                continue
            for image_url in selector.xpath("//img/@src"):
                normalized = self._large_picture_url(image_url)
                if normalized:
                    image_urls.append(normalized)

        deduped = []
        seen = set()
        for image_url in self.normalize_picture_urls(image_urls):
            if image_url in seen:
                continue
            seen.add(image_url)
            deduped.append(image_url)
        return deduped

    def _is_weibo_auth_response(self, response, selector=None, text=None):
        final_url = str(getattr(response, "url", "") or "")
        final_host = urlparse(final_url).netloc.lower()
        if "login.sina.com.cn" in final_host or final_url.startswith("https://weibo.com/login"):
            return True

        page_text = text if text is not None else str(getattr(response, "text", "") or "")
        if "retcode=6102" in page_text or "passport.weibo" in page_text:
            return True

        if selector is not None:
            title = "".join(selector.xpath("//title/text()")).strip()
            if "新浪通行证" in title or "登录" in title:
                return True
        return False

    def _raise_weibo_auth_error(self):
        raise WeiboAuthenticationError(
            f"微博 Cookie 无效或已过期，请刷新 {self.cookie_file} 后重试。"
        )

    def _strip_html_text(self, value):
        value = str(value or "")
        if "<" in value and ">" in value:
            selector = etree.HTML(f"<div>{value}</div>")
            if selector is not None:
                value = " ".join(
                    str(item or "").strip() for item in selector.xpath("//text()")
                )
        return re.sub(r"\s+", " ", value).strip()

    def _extract_status_image_urls(self, status):
        image_urls = []
        for pic in status.get("pics") or []:
            if not isinstance(pic, dict):
                continue
            candidates = [
                ((pic.get("large") or {}).get("url") if isinstance(pic.get("large"), dict) else ""),
                ((pic.get("original") or {}).get("url") if isinstance(pic.get("original"), dict) else ""),
                pic.get("url"),
                pic.get("pid"),
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                if str(candidate).startswith("http"):
                    image_urls.append(self._large_picture_url(candidate))
                    break

        pic_infos = status.get("pic_infos") if isinstance(status.get("pic_infos"), dict) else {}
        pic_ids = status.get("pic_ids") if isinstance(status.get("pic_ids"), list) else []
        ordered_pic_ids = [str(pic_id) for pic_id in pic_ids if str(pic_id) in pic_infos]
        ordered_pic_ids.extend(
            str(pic_id) for pic_id in pic_infos.keys() if str(pic_id) not in ordered_pic_ids
        )
        for pic_id in ordered_pic_ids:
            pic_info = pic_infos.get(pic_id)
            if not isinstance(pic_info, dict):
                continue
            candidates = []
            for key in (
                "largest",
                "original",
                "large",
                "mw2000",
                "largecover",
                "bmiddle",
                "thumbnail",
            ):
                value = pic_info.get(key)
                if isinstance(value, dict):
                    candidates.append(value.get("url"))
                else:
                    candidates.append(value)
            for candidate in candidates:
                if candidate and str(candidate).startswith("http"):
                    image_urls.append(self._large_picture_url(candidate))
                    break

        page_info = status.get("page_info") if isinstance(status.get("page_info"), dict) else {}
        page_pic = page_info.get("page_pic") if page_info else ""
        if page_pic:
            image_urls.append(self._large_picture_url(page_pic))

        deduped = []
        seen = set()
        for image_url in self.normalize_picture_urls(image_urls):
            if image_url in seen:
                continue
            seen.add(image_url)
            deduped.append(image_url)
        return deduped

    def _post_from_status_json(self, post_id, status, source_url, import_source):
        if not isinstance(status, dict):
            raise WeiboSpiderParseError("微博 JSON 详情为空。")
        if int(status.get("ok", 1) or 1) == -100 or "login" in str(status.get("url", "")).lower():
            self._raise_weibo_auth_error()

        post_text = self._strip_html_text(status.get("text_raw") or status.get("text") or "")
        image_urls = self._extract_status_image_urls(status)
        return {
            "platform": "weibo",
            "content_type": "post",
            "external_id": str(status.get("id") or status.get("mid") or post_id),
            "title": post_text[:80].strip() or str(post_id),
            "url": source_url,
            "source_url": source_url,
            "post_text": post_text,
            "image_urls": image_urls,
            "publish_time": str(status.get("created_at") or ""),
            "metadata": {"import_source": str(import_source or "").strip()},
        }

    def _request_weibo_json(self, cookie, url, referer=None):
        try:
            response = requests.get(
                url,
                headers={
                    **self.build_request_headers(cookie=cookie, referer=referer),
                    "Accept": "application/json, text/plain, */*",
                    "MWeibo-Pwa": "1",
                },
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise WeiboSpiderRequestError(f"微博 JSON 请求失败：{url}") from exc

        text = response.text or ""
        if self._is_weibo_auth_response(response, text=text):
            self._raise_weibo_auth_error()
        if response.status_code != 200:
            raise WeiboSpiderRequestError(f"微博返回 HTTP {response.status_code}：{url}")

        try:
            data = response.json()
        except ValueError as exc:
            raise WeiboSpiderParseError(f"微博 JSON 解析失败：{url}") from exc

        if isinstance(data, dict) and (
            int(data.get("ok", 1) or 1) == -100
            or "login" in str(data.get("url", "")).lower()
        ):
            self._raise_weibo_auth_error()
        return data

    def _fetch_single_post_from_json(self, post_id, cookie, import_source):
        source_url = self._single_post_url(post_id)
        import_source = str(import_source or "").strip()
        parsed_source = urlparse(
            import_source if urlparse(import_source).scheme else f"https://{import_source}"
        )
        weibo_com_referer = (
            import_source
            if parsed_source.netloc.lower().endswith("weibo.com")
            else "https://weibo.com/"
        )
        candidates = [
            (f"https://weibo.com/ajax/statuses/show?id={post_id}", weibo_com_referer),
            (f"https://m.weibo.cn/statuses/show?id={post_id}", source_url),
        ]
        last_error = None
        auth_errors = []
        non_auth_errors = []
        for url, referer in candidates:
            try:
                payload = self._request_weibo_json(cookie, url, referer=referer)
                status = payload.get("data") if isinstance(payload, dict) else {}
                if not status and isinstance(payload, dict):
                    status = payload
                result = self._post_from_status_json(
                    post_id,
                    status,
                    source_url,
                    import_source,
                )
                if result.get("post_text") or result.get("image_urls"):
                    return result
                last_error = WeiboSpiderParseError("微博 JSON 详情为空。")
                non_auth_errors.append(last_error)
            except WeiboAuthenticationError as exc:
                auth_errors.append(exc)
                last_error = exc
            except Exception as exc:
                last_error = exc
                non_auth_errors.append(exc)
        if auth_errors and not non_auth_errors:
            raise auth_errors[-1]
        if last_error is not None:
            raise WeiboSpiderParseError(f"微博 JSON 详情解析失败：{last_error}") from last_error
        raise WeiboSpiderParseError("微博 JSON 详情解析失败。")

    def fetch_single_post(self, source):
        if not app_config.platform_enabled("weibo"):
            raise RuntimeError("微博采集当前未启用。")

        post_id = self._extract_single_post_id(source)
        cookie = self.load_cookie()
        self._install_weibo_spider_patches()
        source_url = self._single_post_url(post_id)

        try:
            selector = self._request_weibo_page(cookie, source_url)
        except WeiboAuthenticationError:
            raise
        except (WeiboSpiderRequestError, WeiboSpiderParseError) as exc:
            raise RuntimeError(
                "微博图文详情加载失败。请检查 Cookie 是否有效、请求是否被微博拦截，或链接是否可访问。"
            ) from exc

        node = self._find_single_post_node(selector, post_id)
        if node is None:
            return self._fetch_single_post_from_json(post_id, cookie, source)

        post_text = self._extract_single_post_text(node)
        image_urls = self._extract_single_post_image_urls(cookie, node)
        return {
            "platform": "weibo",
            "content_type": "post",
            "external_id": post_id,
            "title": post_text[:80].strip() or post_id,
            "url": source_url,
            "source_url": source_url,
            "post_text": post_text,
            "image_urls": image_urls,
            "publish_time": self._extract_single_post_publish_time(node),
            "metadata": {"import_source": str(source or "").strip()},
        }

    def _request_weibo_page(self, cookie, url):
        try:
            response = requests.get(
                url,
                headers=self.build_request_headers(cookie=cookie),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise WeiboSpiderRequestError(
                f"微博页面请求失败：{url}"
            ) from exc

        if response.status_code != 200:
            raise WeiboSpiderRequestError(
                f"微博返回 HTTP {response.status_code}：{url}"
            )
        if not response.content:
            raise WeiboSpiderRequestError(f"微博返回空响应：{url}")

        selector = etree.HTML(response.content)
        if selector is None:
            raise WeiboSpiderParseError(f"微博页面 HTML 解析失败：{url}")
        if self._is_weibo_auth_response(response, selector=selector):
            self._raise_weibo_auth_error()
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
            from weibo_spider.parser import comment_parser as comment_parser_module
            from weibo_spider.parser import info_parser as info_parser_module
            from weibo_spider.parser import index_parser as index_parser_module
            from weibo_spider.parser import mblog_picAll_parser as mblog_picAll_parser_module
            from weibo_spider.parser import page_parser as page_parser_module
            from weibo_spider.parser import util as util_module
        except ImportError as exc:
            raise RuntimeError(
                "未安装 weibo-spider，请先安装该依赖。"
            ) from exc

        def patched_handle_html(cookie, url):
            return self._request_weibo_page(cookie, url)

        util_module.handle_html = patched_handle_html
        comment_parser_module.handle_html = patched_handle_html
        index_parser_module.handle_html = patched_handle_html
        info_parser_module.handle_html = patched_handle_html
        mblog_picAll_parser_module.handle_html = patched_handle_html
        page_parser_module.handle_html = patched_handle_html

        collector = self

        class CompatIndexParser(IndexParser):
            def _get_user_id(self):
                if self.selector is None:
                    raise WeiboSpiderParseError("微博用户页面为空或无法读取。")
                return collector._extract_user_id_from_selector(
                    self.selector,
                    self.user_uri,
                )

            def get_user(self):
                if self.selector is None:
                    raise WeiboSpiderParseError("微博用户页面为空或无法读取。")

                user_id = self._get_user_id()
                if not user_id:
                    raise WeiboUserUnavailableError(
                        "无法从源页面解析微博用户 ID。"
                    )

                try:
                    user = info_parser_module.InfoParser(
                        self.cookie,
                        user_id,
                    ).extract_user_info()
                except SystemExit as exc:
                    raise WeiboSpiderRequestError(
                        "微博 Cookie 无效或已过期。"
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
                    raise WeiboSpiderParseError("微博用户页面为空或无法读取。")

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
                        f"微博页数解析失败：{value}"
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
            raise RuntimeError("微博采集当前未启用。")
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
                "微博用户数据加载失败。请检查 Cookie 是否有效、请求是否被微博拦截，或页面是否无法解析。"
            ) from exc
        except SystemExit as exc:
            raise RuntimeError(
                "微博用户数据加载失败。请检查 Cookie 是否仍然有效。"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "微博用户数据加载失败。请检查用户源是否有效。"
            ) from exc

        if not user or not getattr(user, "id", ""):
            raise RuntimeError(
                "微博用户不可用或无法访问，请检查用户链接是否有效、账号是否可见。"
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
                    "微博内容加载失败。请检查 Cookie 是否有效、请求是否被微博拦截，或页面是否无法解析。"
                ) from exc
            except SystemExit as exc:
                raise RuntimeError(
                    "微博内容加载失败。请检查 Cookie 是否仍然有效。"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    "微博内容加载失败。请检查用户源是否有效。"
                ) from exc

            if not page_result:
                raise RuntimeError(
                    "微博内容加载失败，页面可能无法解析。"
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
