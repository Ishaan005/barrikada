from core.layer_a.strip.strip import strip_suspicious_characters
from core.layer_a.strip.utils import detect_control_characters, detect_homoglyphs


def test_normal_text_unchanged():
    text = "Hello, world! This is normal text."
    assert strip_suspicious_characters(text) == text


def test_invisible_and_directional_characters_removed():
    text = "abc\u200b\u200c\u200d\u202edef\u202cghi\ufeff"
    result = strip_suspicious_characters(text)
    assert result == "abcdefghi"


def test_cyrillic_greek_and_mathematical_homoglyphs_normalized():
    assert strip_suspicious_characters("аdmіn pаssword") == "admin password"
    assert strip_suspicious_characters("αdmin ραssword") == "admin password"
    assert strip_suspicious_characters("ℓog_іnfo") == "log_info"


def test_control_characters_removed():
    result = strip_suspicious_characters("hello\x00world\x1f\x7ftest")
    assert result == "helloworldtest"


def test_combined_attack_is_normalized():
    text = "аdmіn\u200b\u202epassword\x00\ufeffsystem"
    assert strip_suspicious_characters(text) == "adminpasswordsystem"


def test_legitimate_unicode_and_whitespace_preserved():
    for text in ("", "   \t  \n  ", "Café naïve résumé", "Hello 🌍 World 🎉"):
        assert strip_suspicious_characters(text) == text


def test_detect_homoglyphs_metadata():
    result, metadata = detect_homoglyphs("аdmіn", normalize=True)
    assert result == "admin"
    assert metadata["homoglyph_count"] == 2
    assert metadata["was_normalized"] is True
    assert len(metadata["homoglyphs_found"]) == 2


def test_detect_homoglyphs_without_normalizing():
    text = "аdmіn"
    result, metadata = detect_homoglyphs(text, normalize=False)
    assert result == text
    assert metadata["homoglyph_count"] == 2
    assert metadata["was_normalized"] is False


def test_detect_control_characters_metadata():
    result, metadata = detect_control_characters("hello\x00\x1fworld", strip_controls=True)
    assert result == "helloworld"
    assert metadata["control_count"] == 2
    assert metadata["was_cleaned"] is True
    assert len(metadata["control_chars"]) == 2


def test_detect_control_characters_without_stripping():
    text = "hello\x00world"
    result, metadata = detect_control_characters(text, strip_controls=False)
    assert result == text
    assert metadata["control_count"] == 1
    assert metadata["was_cleaned"] is False
