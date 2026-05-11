import unittest

from lxml import etree

import core.weibo_adapter as weibo_adapter_module
from core.config import DEFAULT_CONFIG, app_config
from core.weibo_adapter import (
    WeiboAuthenticationError,
    WeiboSpiderParseError,
    WeiboSpiderRequestError,
    WeiboUserCollector,
)


class FakeUser:
    def __init__(self, user_id, nickname):
        self.id = user_id
        self.nickname = nickname


class FakeWeibo:
    def __init__(
        self,
        weibo_id,
        content,
        pictures,
        publish_time="2026-04-20 10:00",
    ):
        self.id = weibo_id
        self.content = content
        self.original_pictures = pictures
        self.publish_time = publish_time
        self.publish_place = "Beijing"
        self.publish_tool = "iPhone"
        self.up_num = 1
        self.retweet_num = 2
        self.comment_num = 3
        self.article_url = ""
        self.original = True


class FakeIndexParser:
    def __init__(self, cookie, user_uri):
        self.cookie = cookie
        self.user_uri = user_uri

    def get_user(self):
        return FakeUser("123456", "Test User")

    def get_page_num(self):
        return 2


class RecordingPageParser:
    seen_user_uris = []

    def __init__(self, cookie, user_config, page, filter_value):
        self.page = page
        self.__class__.seen_user_uris.append(user_config["user_uri"])

    def get_one_page(self, seen_ids):
        if self.page == 1:
            return (
                [
                    FakeWeibo("w1", "launch event in shanghai", "https://img/1.jpg"),
                    FakeWeibo("w2", "text only", "无"),
                ],
                seen_ids + ["w1", "w2"],
                True,
            )
        return (
            [FakeWeibo("w3", "dinner with a friend", "https://img/3a.jpg,https://img/3b.jpg")],
            seen_ids + ["w3"],
            False,
        )


class RequestFailingIndexParser:
    def __init__(self, cookie, user_uri):
        raise WeiboSpiderRequestError("blocked")


class FakeResponse:
    def __init__(self, url, body, status_code=200, json_data=None):
        self.url = url
        self.content = body.encode("utf-8")
        self.text = body
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError("not json")
        return self._json_data


class WeiboAdapterTests(unittest.TestCase):
    def test_collector_reads_platform_timeout_and_retry_defaults(self):
        original_data = app_config.data
        try:
            app_config.data = {
                **DEFAULT_CONFIG,
                "collection": {
                    **DEFAULT_CONFIG["collection"],
                    "platforms": {
                        **DEFAULT_CONFIG["collection"]["platforms"],
                        "weibo": {
                            **DEFAULT_CONFIG["collection"]["platforms"]["weibo"],
                            "timeout_seconds": 31,
                            "retry_count": 6,
                        },
                    },
                },
            }
            collector = WeiboUserCollector(cookie_file="dummy")

            self.assertEqual(collector.timeout_seconds, 31)
            self.assertEqual(collector.retries, 6)
        finally:
            app_config.data = original_data

    def test_normalize_keywords(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        self.assertEqual(
            collector.normalize_keywords("launch, shanghai\nfriend, shanghai"),
            ["launch", "shanghai", "friend"],
        )

    def test_normalize_user_source_supports_numeric_and_profile_urls(self):
        collector = WeiboUserCollector(cookie_file="dummy")

        self.assertEqual(
            collector.normalize_user_source("123456"),
            {
                "user_uri": "u/123456",
                "canonical_url": "https://weibo.cn/123456",
            },
        )
        self.assertEqual(
            collector.normalize_user_source("weibo.com/u/7473796836"),
            {
                "user_uri": "u/7473796836",
                "canonical_url": "https://weibo.cn/7473796836",
            },
        )
        self.assertEqual(
            collector.normalize_user_source("https://weibo.cn/7473796836"),
            {
                "user_uri": "u/7473796836",
                "canonical_url": "https://weibo.cn/7473796836",
            },
        )
        self.assertEqual(
            collector.normalize_user_source("https://weibo.cn/n/some_user"),
            {
                "user_uri": "n/some_user",
                "canonical_url": "https://weibo.cn/n/some_user",
            },
        )

    def test_single_post_id_parsing_supports_common_weibo_detail_urls(self):
        collector = WeiboUserCollector(cookie_file="dummy")

        self.assertEqual(
            collector._extract_single_post_id("https://weibo.cn/comment/5241373692531045"),
            "5241373692531045",
        )
        self.assertEqual(
            collector._extract_single_post_id("https://m.weibo.cn/detail/5241373692531045"),
            "5241373692531045",
        )
        self.assertEqual(
            collector._extract_single_post_id("https://weibo.com/123456/Nx2abcDEF"),
            "Nx2abcDEF",
        )

    def test_fetch_single_post_extracts_text_and_single_image(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None

        def fake_request(cookie, url):
            return etree.HTML(
                """
                <html><body>
                  <div class="c" id="M_5241373692531045">
                    <div>
                      <span class="ctt">微博正文
                        <a href="/mblog/pic/5241373692531045">
                          <img src="https://wx1.sinaimg.cn/wap180/demo.jpg"/>
                        </a>
                      </span>
                    </div>
                    <div><span class="ct">2026-05-11 10:00 来自 iPhone</span></div>
                  </div>
                </body></html>
                """
            )

        collector._request_weibo_page = fake_request

        result = collector.fetch_single_post(
            "https://weibo.cn/comment/5241373692531045"
        )

        self.assertEqual(result["content_type"], "post")
        self.assertEqual(result["external_id"], "5241373692531045")
        self.assertIn("微博正文", result["post_text"])
        self.assertEqual(
            result["image_urls"],
            ["https://wx1.sinaimg.cn/large/demo.jpg"],
        )

    def test_fetch_single_post_extracts_multi_image_picall_page(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None

        def fake_request(cookie, url):
            if "/mblog/picAll/" in url:
                return etree.HTML(
                    """
                    <html><body>
                      <img src="https://wx1.sinaimg.cn/thumb180/a.jpg"/>
                      <img src="https://wx2.sinaimg.cn/thumb180/b.jpg"/>
                    </body></html>
                    """
                )
            return etree.HTML(
                """
                <html><body>
                  <div class="c" id="M_5241373692531045">
                    <span class="ctt">组图正文</span>
                    <a href="/mblog/picAll/5241373692531045?rl=1">组图</a>
                  </div>
                </body></html>
                """
            )

        collector._request_weibo_page = fake_request

        result = collector.fetch_single_post(
            "https://m.weibo.cn/detail/5241373692531045"
        )

        self.assertEqual(
            result["image_urls"],
            [
                "https://wx1.sinaimg.cn/large/a.jpg",
                "https://wx2.sinaimg.cn/large/b.jpg",
            ],
        )

    def test_fetch_single_post_reports_missing_cookie(self):
        collector = WeiboUserCollector(cookie_file="missing")

        with self.assertRaisesRegex(RuntimeError, "Cookie"):
            collector.fetch_single_post("https://weibo.cn/comment/5241373692531045")

    def test_request_weibo_page_reports_login_redirect_as_cookie_error(self):
        collector = WeiboUserCollector(cookie_file="dummy")

        def fake_get(*args, **kwargs):
            return FakeResponse(
                "https://login.sina.com.cn/sso/login.php",
                "<html><title>新浪通行证</title><body>login</body></html>",
            )

        original_get = weibo_adapter_module.requests.get
        try:
            weibo_adapter_module.requests.get = fake_get
            with self.assertRaisesRegex(WeiboAuthenticationError, "微博 Cookie 无效"):
                collector._request_weibo_page("SUB=demo;", "https://weibo.cn/comment/1")
        finally:
            weibo_adapter_module.requests.get = original_get

    def test_request_weibo_page_reports_retcode_6102_as_cookie_error(self):
        collector = WeiboUserCollector(cookie_file="dummy")

        def fake_get(*args, **kwargs):
            return FakeResponse(
                "https://weibo.cn/comment/1",
                '<html><title>新浪通行证</title><body>location.replace("https://weibo.cn/comment/1?display=0&amp;retcode=6102");</body></html>',
            )

        original_get = weibo_adapter_module.requests.get
        try:
            weibo_adapter_module.requests.get = fake_get
            with self.assertRaisesRegex(WeiboAuthenticationError, "微博 Cookie 无效"):
                collector._request_weibo_page("SUB=demo;", "https://weibo.cn/comment/1")
        finally:
            weibo_adapter_module.requests.get = original_get

    def test_fetch_single_post_falls_back_to_status_json(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None
        calls = []

        def fake_request_page(cookie, url):
            return etree.HTML("<html><body><div class='c'>not detail</div></body></html>")

        def fake_request_json(cookie, url, referer=None):
            calls.append((url, referer))
            return {
                "ok": 1,
                "data": {
                    "id": "5241373692531045",
                    "text": "<span>JSON 正文</span>",
                    "pics": [
                        {"large": {"url": "https://wx1.sinaimg.cn/mw690/a.jpg"}},
                        {"original": {"url": "https://wx2.sinaimg.cn/orj360/b.jpg"}},
                    ],
                    "page_info": {"page_pic": "https://wx3.sinaimg.cn/thumb180/c.jpg"},
                    "created_at": "Mon May 11 10:00:00 +0800 2026",
                },
            }

        collector._request_weibo_page = fake_request_page
        collector._request_weibo_json = fake_request_json

        result = collector.fetch_single_post(
            "https://weibo.com/1793285524/5241373692531045"
        )

        self.assertEqual(result["post_text"], "JSON 正文")
        self.assertEqual(
            result["image_urls"],
            [
                "https://wx1.sinaimg.cn/large/a.jpg",
                "https://wx2.sinaimg.cn/large/b.jpg",
                "https://wx3.sinaimg.cn/large/c.jpg",
            ],
        )
        self.assertEqual(
            calls,
            [
                (
                    "https://weibo.com/ajax/statuses/show?id=5241373692531045",
                    "https://weibo.com/1793285524/5241373692531045",
                )
            ],
        )

    def test_fetch_single_post_continues_after_json_candidate_auth_error(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None

        def fake_request_page(cookie, url):
            return etree.HTML("<html><body><div class='c'>not detail</div></body></html>")

        def fake_request_json(cookie, url, referer=None):
            if "weibo.com/ajax/statuses/show" in url:
                raise WeiboAuthenticationError("ajax auth failed")
            return {
                "ok": 1,
                "data": {
                    "id": "5241373692531045",
                    "text_raw": "m.weibo 正文",
                    "pics": [{"large": {"url": "https://wx1.sinaimg.cn/mw690/a.jpg"}}],
                },
            }

        collector._request_weibo_page = fake_request_page
        collector._request_weibo_json = fake_request_json

        result = collector.fetch_single_post(
            "https://weibo.com/1793285524/5241373692531045"
        )

        self.assertEqual(result["post_text"], "m.weibo 正文")
        self.assertEqual(result["image_urls"], ["https://wx1.sinaimg.cn/large/a.jpg"])

    def test_fetch_single_post_extracts_ajax_pic_infos(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None

        def fake_request_page(cookie, url):
            return etree.HTML("<html><body><div class='c'>not detail</div></body></html>")

        def fake_request_json(cookie, url, referer=None):
            return {
                "id": "5241373692531045",
                "text_raw": "ajax 正文",
                "pic_ids": ["pic-b", "pic-a"],
                "pic_infos": {
                    "pic-a": {
                        "large": {"url": "https://wx1.sinaimg.cn/mw690/a.jpg"},
                    },
                    "pic-b": {
                        "largest": {"url": "https://wx2.sinaimg.cn/orj360/b.jpg"},
                    },
                },
            }

        collector._request_weibo_page = fake_request_page
        collector._request_weibo_json = fake_request_json

        result = collector.fetch_single_post(
            "https://weibo.com/1793285524/5241373692531045"
        )

        self.assertEqual(result["post_text"], "ajax 正文")
        self.assertEqual(
            result["image_urls"],
            [
                "https://wx2.sinaimg.cn/large/b.jpg",
                "https://wx1.sinaimg.cn/large/a.jpg",
            ],
        )

    def test_fetch_single_post_reports_auth_error_only_when_all_json_candidates_auth_fail(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None

        def fake_request_page(cookie, url):
            return etree.HTML("<html><body><div class='c'>not detail</div></body></html>")

        def fake_request_json(cookie, url, referer=None):
            raise WeiboAuthenticationError("candidate auth failed")

        collector._request_weibo_page = fake_request_page
        collector._request_weibo_json = fake_request_json

        with self.assertRaisesRegex(WeiboAuthenticationError, "candidate auth failed"):
            collector.fetch_single_post(
                "https://weibo.com/1793285524/5241373692531045"
            )

    def test_fetch_single_post_reports_parse_error_when_json_candidates_mix_auth_and_parse_failures(self):
        collector = WeiboUserCollector(cookie_file="dummy")
        collector.load_cookie = lambda: "SUB=demo;"
        collector._install_weibo_spider_patches = lambda: None

        def fake_request_page(cookie, url):
            return etree.HTML("<html><body><div class='c'>not detail</div></body></html>")

        def fake_request_json(cookie, url, referer=None):
            if "weibo.com/ajax/statuses/show" in url:
                raise WeiboAuthenticationError("ajax auth failed")
            return {"ok": 1, "data": {}}

        collector._request_weibo_page = fake_request_page
        collector._request_weibo_json = fake_request_json

        with self.assertRaisesRegex(WeiboSpiderParseError, "微博 JSON 详情解析失败"):
            collector.fetch_single_post(
                "https://weibo.com/1793285524/5241373692531045"
            )

    def test_fetch_user_posts_uses_numeric_user_uri_for_page_fetches(self):
        RecordingPageParser.seen_user_uris = []
        collector = WeiboUserCollector(
            cookie_file="dummy",
            index_parser_cls=FakeIndexParser,
            page_parser_cls=RecordingPageParser,
        )
        collector.load_cookie = lambda: "SUB=demo;"

        result = collector.fetch_user_posts(
            "weibo.com/u/123456",
            limit=10,
            keywords=["shanghai", "friend"],
        )

        self.assertEqual(result["title"], "Test User")
        self.assertEqual(result["user_id"], "123456")
        self.assertEqual(result["source_url"], "https://weibo.cn/123456")
        self.assertEqual(len(result["entries"]), 2)
        self.assertEqual(result["entries"][0]["external_id"], "w1")
        self.assertEqual(
            result["entries"][1]["image_urls"],
            ["https://img/3a.jpg", "https://img/3b.jpg"],
        )
        self.assertEqual(result["stats"]["fetched_count"], 3)
        self.assertEqual(result["stats"]["matched_count"], 2)
        self.assertEqual(result["stats"]["filtered_count"], 1)
        self.assertEqual(RecordingPageParser.seen_user_uris, ["123456", "123456"])

    def test_fetch_user_posts_reports_request_failures_clearly(self):
        collector = WeiboUserCollector(
            cookie_file="dummy",
            index_parser_cls=RequestFailingIndexParser,
            page_parser_cls=RecordingPageParser,
        )
        collector.load_cookie = lambda: "SUB=demo;"

        with self.assertRaisesRegex(
            RuntimeError,
            r"微博用户数据加载失败",
        ):
            collector.fetch_user_posts("weibo.com/u/123456", limit=1)


if __name__ == "__main__":
    unittest.main()
