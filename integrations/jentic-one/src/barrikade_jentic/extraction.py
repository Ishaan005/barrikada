"""Bounded response and OpenAPI text extraction with content-safe locators."""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass
from typing import Any, Literal

from barrikade_jentic.generated.models import TextSegment

_JSON_CONTENT_TYPE = re.compile(r"^(application|text)/(?:[\w.+-]+\+)?json$")
_TEXT_CONTENT_TYPES = {
    "application/xml": "xml",
    "application/yaml": "yaml",
    "application/x-yaml": "yaml",
    "application/problem+xml": "problem",
}
_SPEC_SOURCES = {
    "title": "specification_title",
    "summary": "specification_summary",
    "description": "specification_description",
    "default": "specification_default",
    "example": "specification_example",
    "examples": "specification_example",
    "externalDocs": "specification_external_docs",
    "tags": "specification_tags",
}
_MAX_NESTING_DEPTH = 64
_MAX_CONTAINER_NODES = 10_000


class ExtractionLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractionResult:
    applicability: Literal["eligible", "not_applicable", "incomplete"]
    content_sha256: str
    segments: list[TextSegment]
    body: bytes
    headers: dict[str, str]
    reason: str | None = None


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _source_for(media_type: str) -> str | None:
    if _JSON_CONTENT_TYPE.match(media_type):
        return "problem" if "problem" in media_type else "json_value"
    if media_type.startswith("text/"):
        if media_type == "text/html":
            return "html"
        if media_type in {"text/xml"}:
            return "xml"
        if media_type in {"text/yaml", "text/x-yaml"}:
            return "yaml"
        return "text"
    return _TEXT_CONTENT_TYPES.get(media_type)


def _decompress(
    body: bytes,
    encoding: str,
    *,
    max_bytes: int,
    ratio_limit: int,
) -> bytes:
    if len(body) > max_bytes:
        raise ExtractionLimitError("compressed_body_limit")

    def inflate(window_bits: int) -> bytes:
        inflater = zlib.decompressobj(window_bits)
        decoded = inflater.decompress(body, max_bytes + 1)
        if len(decoded) > max_bytes or inflater.unconsumed_tail:
            raise ExtractionLimitError("decompressed_body_limit")
        remaining = max_bytes + 1 - len(decoded)
        if remaining > 0:
            decoded += inflater.flush(remaining)
        if len(decoded) > max_bytes:
            raise ExtractionLimitError("decompressed_body_limit")
        if not inflater.eof:
            raise ExtractionLimitError("invalid_compressed_body")
        return decoded

    try:
        if encoding == "gzip":
            decoded = inflate(zlib.MAX_WBITS | 16)
        elif encoding == "deflate":
            try:
                decoded = inflate(zlib.MAX_WBITS)
            except zlib.error:
                decoded = inflate(-zlib.MAX_WBITS)
        else:
            raise ExtractionLimitError("unsupported_content_encoding")
    except (EOFError, zlib.error) as exc:
        raise ExtractionLimitError("invalid_compressed_body") from exc
    if len(decoded) > max_bytes:
        raise ExtractionLimitError("decompressed_body_limit")
    if body and len(decoded) / len(body) > ratio_limit:
        raise ExtractionLimitError("decompression_ratio_limit")
    return decoded


def _normalized_body(
    body: bytes,
    headers: dict[str, str],
    *,
    max_bytes: int,
    ratio_limit: int,
) -> tuple[bytes, dict[str, str]]:
    encoding = next(
        (
            value.strip().lower()
            for key, value in headers.items()
            if key.lower() == "content-encoding"
        ),
        "identity",
    )
    if encoding in {"", "identity"}:
        return body, dict(headers)
    decoded = _decompress(body, encoding, max_bytes=max_bytes, ratio_limit=ratio_limit)
    normalized_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() not in {"content-encoding", "content-length"}
    }
    return decoded, normalized_headers


def _safe_locator(path: str) -> str:
    if len(path) <= 512 and path.startswith("$"):
        return path
    if re.fullmatch(r"locator_sha256:[0-9a-f]{64}", path):
        return path
    return f"locator_sha256:{hashlib.sha256(path.encode()).hexdigest()}"


def _key_locator(parent: str, key: str, kind: str) -> str:
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _safe_locator(f"{parent}.key_{key_hash}.{kind}")


def _text_windows(text: str, max_bytes: int, overlap_chars: int = 256):
    start = 0
    while start < len(text):
        low = start + 1
        high = len(text)
        best = start
        while low <= high:
            middle = (low + high) // 2
            if len(text[start:middle].encode("utf-8")) <= max_bytes:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == start:
            raise ExtractionLimitError("segment_encoding_limit")
        yield text[start:best]
        if best == len(text):
            break
        start = max(start + 1, best - overlap_chars)


class _SegmentCollector:
    def __init__(self, max_segments: int, max_segment_bytes: int, max_text_bytes: int) -> None:
        self.max_segments = max_segments
        self.max_segment_bytes = max_segment_bytes
        self.max_text_bytes = max_text_bytes
        self.total_bytes = 0
        self.segments: list[TextSegment] = []

    def add(self, text: str, source: str, locator: str) -> None:
        if not text:
            return
        for window_index, window in enumerate(_text_windows(text, self.max_segment_bytes)):
            encoded = window.encode("utf-8")
            if len(self.segments) >= self.max_segments:
                raise ExtractionLimitError("segment_count_limit")
            if self.total_bytes + len(encoded) > self.max_text_bytes:
                raise ExtractionLimitError("aggregate_text_limit")
            segment_id = f"seg-{len(self.segments) + 1:04d}"
            window_locator = locator if window_index == 0 else f"{locator}.window_{window_index}"
            self.segments.append(
                TextSegment(
                    id=segment_id,
                    text=window,
                    source=source,
                    locator=_safe_locator(window_locator),
                    sha256=hashlib.sha256(encoded).hexdigest(),
                )
            )
            self.total_bytes += len(encoded)


def _collect_json(
    value: Any,
    collector: _SegmentCollector,
    path: str = "$",
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
) -> None:
    if depth > _MAX_NESTING_DEPTH:
        raise ExtractionLimitError("json_nesting_limit")
    if node_count is None:
        node_count = [0]
    node_count[0] += 1
    if node_count[0] > _MAX_CONTAINER_NODES:
        raise ExtractionLimitError("json_node_limit")
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            locator = _key_locator(path, key_text, "key")
            collector.add(key_text, "json_key", locator)
            _collect_json(
                child,
                collector,
                _key_locator(path, key_text, "value"),
                depth=depth + 1,
                node_count=node_count,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_json(
                child,
                collector,
                f"{path}[{index}]",
                depth=depth + 1,
                node_count=node_count,
            )
    elif isinstance(value, str):
        collector.add(value, "json_value", path)


def extract_response(
    *,
    body: bytes,
    headers: dict[str, str],
    content_type: str | None,
    max_text_bytes: int,
    max_segment_bytes: int,
    max_segments: int,
    decompression_ratio_limit: int,
) -> ExtractionResult:
    media_type = _media_type(content_type)
    source = _source_for(media_type)
    digest = hashlib.sha256(body).hexdigest()
    if not body or source is None:
        return ExtractionResult("not_applicable", digest, [], body, dict(headers))

    try:
        normalized, normalized_headers = _normalized_body(
            body,
            headers,
            max_bytes=max_text_bytes,
            ratio_limit=decompression_ratio_limit,
        )
        digest = hashlib.sha256(normalized).hexdigest()
        if len(normalized) > max_text_bytes:
            raise ExtractionLimitError("eligible_body_limit")
        text = normalized.decode("utf-8", errors="strict")
        if not text:
            return ExtractionResult("not_applicable", digest, [], normalized, normalized_headers)
        collector = _SegmentCollector(max_segments, max_segment_bytes, max_text_bytes)
        if _JSON_CONTENT_TYPE.match(media_type):
            try:
                document = json.loads(text)
            except (json.JSONDecodeError, RecursionError):
                collector.add(text, source, "$raw")
            else:
                _collect_json(document, collector)
        else:
            collector.add(text, source, "$body")
        if not collector.segments:
            return ExtractionResult("not_applicable", digest, [], normalized, normalized_headers)
        return ExtractionResult(
            "eligible", digest, collector.segments, normalized, normalized_headers
        )
    except (ExtractionLimitError, UnicodeDecodeError) as exc:
        return ExtractionResult(
            "incomplete",
            digest,
            [],
            body,
            dict(headers),
            reason=str(exc),
        )


def canonical_specification(specification: dict[str, Any]) -> tuple[bytes, str]:
    encoded = json.dumps(
        specification,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return encoded, hashlib.sha256(encoded).hexdigest()


def extract_specification(
    specification: dict[str, Any],
    *,
    max_text_bytes: int,
    max_segment_bytes: int,
    max_segments: int,
) -> tuple[str, list[TextSegment]]:
    _, digest = canonical_specification(specification)
    collector = _SegmentCollector(max_segments, max_segment_bytes, max_text_bytes)

    node_count = [0]

    def walk(value: Any, path: str, selected_source: str | None = None, depth: int = 0) -> None:
        if depth > _MAX_NESTING_DEPTH:
            raise ExtractionLimitError("specification_nesting_limit")
        node_count[0] += 1
        if node_count[0] > _MAX_CONTAINER_NODES:
            raise ExtractionLimitError("specification_node_limit")
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                child_source = _SPEC_SOURCES.get(key_text, selected_source)
                child_path = _key_locator(path, key_text, "field")
                walk(child, child_path, child_source, depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", selected_source, depth + 1)
        elif selected_source and isinstance(value, str):
            collector.add(value, selected_source, path)

    walk(specification, "$")
    return digest, collector.segments
