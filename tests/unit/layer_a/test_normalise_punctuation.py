import pytest

from core.layer_a.normalise_punctuation import (
    collapse_separated_characters,
    normalise_punctuation_and_whitespace,
)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("I g n o r e", "Ignore"),
        ("I-g-n-o-r-e", "Ignore"),
        ("I.g.n.o.r.e", "Ignore"),
    ],
)
def test_collapse_separated_characters(original, expected):
    assert collapse_separated_characters(original) == expected


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("“double” and ‘single’", "\"double\" and 'single'"),
        ("«quoted»", '"quoted"'),
        ("en–dash em—dash minus−sign", "en-dash em-dash minus-sign"),
    ],
)
def test_normalizes_punctuation(original, expected):
    result = normalise_punctuation_and_whitespace(original)
    assert result == {"original": original, "normalised": expected}


def test_normalizes_newlines_and_horizontal_whitespace():
    original = "  first\r\n\r\n second\t\tvalue  "
    result = normalise_punctuation_and_whitespace(original)
    assert result == {"original": original, "normalised": "first\nsecond value"}
