"""Web fetch's page shaping and its guard against private addresses (inference/web.py)."""

from inference.web import _page_text, _public_url


def test_html_becomes_readable_text_without_page_chrome() -> None:
    html = b"""<html><head><title>Fall 2026 Calendar</title><style>p{}</style></head>
    <body><nav>Home | About</nav><h1>Academic Calendar</h1>
    <p>Last day to drop with no grade: <b>Sept 9</b>.</p><ul><li>Classes begin Aug 31</li></ul>
    <script>track()</script><footer>Copyright</footer></body></html>"""
    title, text = _page_text(html, "text/html; charset=utf-8")
    assert title == "Fall 2026 Calendar"
    assert "Last day to drop with no grade: Sept 9." in text
    assert "- Classes begin Aug 31" in text
    assert "# Academic Calendar" in text
    for chrome in ("Home | About", "track()", "Copyright", "p{}"):
        assert chrome not in text


def test_fetch_refuses_local_and_private_hosts_and_other_schemes() -> None:
    for url in ("http://localhost:8080/agents", "http://127.0.0.1/", "http://10.89.0.5:5432/", "http://192.168.1.1/", "file:///etc/passwd", "ftp://example.com/"):
        assert _public_url(url) is not None, url
