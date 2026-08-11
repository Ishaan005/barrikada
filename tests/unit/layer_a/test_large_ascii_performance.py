import time

from core.layer_a.pipeline import analyze_text


def test_large_ascii_response_uses_bounded_fast_path():
    text = ("A routine weather observation for Dublin. " * 2000)[: 64 * 1024]
    started = time.perf_counter()
    result = analyze_text(text)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert result.processed_text == text
    assert result.flags == []
    assert elapsed_ms < 250
