"""Web search and fetch for Hath's web tools, shaped so an agent's prompt gets only what it can use.

Search runs Anthropic's web search server tool in a call of its own on a small model and
returns one compact line per result (title, url, age, and the model's few-word account of
the page), never the raw result blocks: one search's blocks ran 18.5k characters on
2026-10-19. Fetch gets pages directly, in parallel, as plain text, and with a question has
the small model answer it from each page on its own, so a page's text reaches the caller
whole (to store) while the agent sees only the answer.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import anthropic
import httpx

from config import Web, get_config
from errors import DwarError, TransportError
from inference.types import Usage
from logutil import caller_var, log_extra, request_id_var

import logging

logger = logging.getLogger("dwar")

SEARCH_SYSTEM = (
    "You run exactly one web search for the query you are given, then report what it found. "
    "Do not answer the query yourself and do not search again."
)

SEARCH_INSTRUCTION = (
    "Search once for: {query}\n\n"
    "Then, for each result in the order the search returned them, write one line and nothing else: "
    "`<n>. <what that page says that bears on the query, at most 30 words>`. "
    "Write only what the page itself says; if a page says nothing about the query, write `<n>. (nothing on this)`."
)

ANSWER_SYSTEM = (
    "You answer one question from one web page, using only that page's text. "
    "Quote the exact words that answer it, in quotation marks, with as little else as possible. "
    "If the page does not answer it, say `Not on this page.` and, in a few words, what the page is about. "
    "Never use outside knowledge."
)


def _client(web: Web) -> anthropic.Anthropic:
    cfg = get_config()
    if not cfg.env.anthropic_api_key:
        raise DwarError(503, "provider_unconfigured", "ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=cfg.env.anthropic_api_key, max_retries=0, timeout=cfg.retry.timeout_seconds)


def _create(client: anthropic.Anthropic, kwargs: dict[str, Any]) -> Any:
    try:
        return client.messages.create(**kwargs)
    except anthropic.RateLimitError as exc:
        raise TransportError(429, "rate_limit", str(exc)) from exc
    except anthropic.APITimeoutError as exc:
        raise TransportError(504, "upstream_timeout", str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        raise TransportError(502, "upstream_unreachable", str(exc)) from exc
    except anthropic.APIStatusError as exc:
        if exc.status_code == 429:
            raise TransportError(429, "rate_limit", str(exc)) from exc
        if exc.status_code >= 500:
            raise TransportError(502, "provider", str(exc)) from exc
        raise DwarError(502, "provider", str(exc)) from exc


def _with_retry(call):
    retry = get_config().retry
    for attempt in range(retry.attempts):
        try:
            return call()
        except TransportError as exc:
            if attempt + 1 == retry.attempts:
                raise DwarError(exc.status_code, exc.type, exc.message) from exc
            time.sleep(retry.backoff_seconds[min(attempt, len(retry.backoff_seconds) - 1)])


def _usage(response: Any) -> Usage:
    usage = response.usage
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )


def _log(route: str, model: str, started: float, **fields: object) -> None:
    logger.info(
        "inference",
        extra=log_extra(
            request_id=request_id_var.get(),
            caller=caller_var.get(),
            route=route,
            provider="anthropic",
            model=model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            **fields,
        ),
    )


_LINE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")


def search(query: str, max_results: int, allowed_domains: list[str] | None, blocked_domains: list[str] | None) -> dict[str, Any]:
    """search runs one web search and returns its results as compact rows, each with the model's short account of the page."""
    web = get_config().web
    tool: dict[str, Any] = {"type": web.search_tool, "name": "web_search", "max_uses": 1}
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    if blocked_domains:
        tool["blocked_domains"] = blocked_domains
    kwargs = {
        "model": web.model,
        "max_tokens": web.search_max_tokens,
        "system": SEARCH_SYSTEM,
        "tools": [tool],
        "messages": [{"role": "user", "content": SEARCH_INSTRUCTION.format(query=query)}],
    }
    client = _client(web)
    started = time.perf_counter()
    response = _with_retry(lambda: _create(client, kwargs))
    results: list[dict[str, Any]] = []
    notes: dict[int, str] = {}
    for block in response.content:
        if block.type == "web_search_tool_result":
            content = block.content
            if not isinstance(content, list):
                code = getattr(content, "error_code", "unknown")
                raise DwarError(502, "search_failed", f"web search failed: {code}")
            for item in content:
                results.append(
                    {"title": item.title, "url": item.url, "page_age": getattr(item, "page_age", None)}
                )
        elif block.type == "text":
            for line in block.text.splitlines():
                match = _LINE.match(line)
                if match:
                    notes[int(match.group(1))] = match.group(2)
    for index, row in enumerate(results, start=1):
        note = notes.get(index)
        row["about"] = None if note is None or note == "(nothing on this)" else note
    usage = _usage(response)
    searches = getattr(getattr(response.usage, "server_tool_use", None), "web_search_requests", 0) or 0
    _log("web.search", web.model, started, searches=searches, **usage.model_dump())
    return {"results": results[:max_results], "searches": searches, "usage": usage.model_dump()}


class _TextExtractor(HTMLParser):
    """_TextExtractor turns HTML into readable text: block elements become line breaks, and
    script, style, navigation, header and footer chrome is left out."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "form", "iframe", "template"}
    BLOCK = {"p", "div", "section", "article", "main", "li", "ul", "ol", "tr", "table", "br", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")
            if tag in {"h1", "h2", "h3"}:
                self.parts.append("## " if tag != "h1" else "# ")
            elif tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip > 0:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._skip == 0:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r" *\n *", "\n", joined)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def _page_text(body: bytes, content_type: str) -> tuple[str, str]:
    """_page_text is a page's title and readable text, from HTML or a plain-text type."""
    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    if match:
        charset = match.group(1)
    decoded = body.decode(charset, errors="replace")
    if "html" in content_type or decoded.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        parser = _TextExtractor()
        parser.feed(decoded)
        return parser.title.strip(), parser.text()
    return "", decoded.strip()


def _public_url(url: str) -> str | None:
    """_public_url is why `url` must not be fetched, or None when it may: only http(s) to a
    host whose every address is public, so a page cannot steer an agent into the box's own
    services (hath, nas, the databases) or anything else on the private network."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "only public http(s) pages can be fetched"
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        return "the host does not resolve"
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return "the host is on a private or local network"
    return None


def _get(web: Web, url: str) -> dict[str, Any]:
    """_get fetches one public page and returns its text, or the error that stopped it.
    Redirects are followed by hand so every hop is checked by _public_url."""
    row: dict[str, Any] = {"url": url, "final_url": url, "status": None, "title": "", "text": "", "error": None}
    try:
        with httpx.Client(follow_redirects=False, timeout=web.fetch_timeout_seconds, headers={"User-Agent": web.user_agent}) as client:
            current = url
            for _ in range(web.max_redirects + 1):
                refusal = _public_url(current)
                if refusal is not None:
                    row["error"] = refusal
                    return row
                response = client.get(current)
                if response.is_redirect and "location" in response.headers:
                    current = urljoin(current, response.headers["location"])
                    continue
                break
            else:
                row["error"] = "too many redirects"
                return row
            row["status"] = response.status_code
            row["final_url"] = current
            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400:
                row["error"] = f"HTTP {response.status_code}"
                return row
            if content_type and not any(kind in content_type for kind in ("html", "text", "json", "xml")):
                row["error"] = f"not a text page ({content_type.split(';')[0]})"
                return row
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > web.max_page_bytes:
                row["error"] = f"page is larger than {web.max_page_bytes} bytes"
                return row
            body = response.content[: web.max_page_bytes]
        title, text = _page_text(bytes(body), content_type)
        row["title"] = title
        row["text"] = text[: web.max_page_chars]
        if not text:
            row["error"] = "no readable text (the page may need a browser)"
    except httpx.TimeoutException:
        row["error"] = "timed out"
    except httpx.HTTPError as exc:
        row["error"] = f"could not fetch: {exc.__class__.__name__}"
    return row


def _answer(client: anthropic.Anthropic, web: Web, row: dict[str, Any], question: str) -> tuple[str | None, Usage | None]:
    """_answer has the small model answer `question` from one fetched page's text alone."""
    if row["error"] is not None:
        return None, None
    page = row["text"][: web.answer_page_chars]
    kwargs = {
        "model": web.model,
        "max_tokens": web.answer_max_tokens,
        "system": ANSWER_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": f"Page: {row['final_url']}\nTitle: {row['title']}\n\n<page>\n{page}\n</page>\n\nQuestion: {question}",
            }
        ],
    }
    response = _with_retry(lambda: _create(client, kwargs))
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return (text or "Not on this page."), _usage(response)


def fetch(urls: list[str], question: str | None) -> dict[str, Any]:
    """fetch gets `urls` in parallel as text and, with a question, answers it from each page on its own."""
    web = get_config().web
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(urls), web.max_parallel)) as pool:
        rows = list(pool.map(lambda url: _get(web, url), urls))
    totals = Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    if question:
        client = _client(web)
        with ThreadPoolExecutor(max_workers=min(len(rows), web.max_parallel)) as pool:
            answers = list(pool.map(lambda row: _answer(client, web, row, question), rows))
        for row, (answer, usage) in zip(rows, answers):
            row["answer"] = answer
            if usage is not None:
                totals = Usage(
                    input_tokens=totals.input_tokens + usage.input_tokens,
                    output_tokens=totals.output_tokens + usage.output_tokens,
                    cache_read_input_tokens=totals.cache_read_input_tokens + usage.cache_read_input_tokens,
                    cache_creation_input_tokens=totals.cache_creation_input_tokens + usage.cache_creation_input_tokens,
                )
    _log("web.fetch", web.model, started, pages=len(rows), answered=bool(question), **totals.model_dump())
    return {"pages": rows, "usage": totals.model_dump()}
