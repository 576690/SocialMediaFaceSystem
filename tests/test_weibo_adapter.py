import unittest

from core.weibo_adapter import WeiboSpiderRequestError, WeiboUserCollector


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


class WeiboAdapterTests(unittest.TestCase):
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
            r"Failed to load Weibo user data",
        ):
            collector.fetch_user_posts("weibo.com/u/123456", limit=1)


if __name__ == "__main__":
    unittest.main()
