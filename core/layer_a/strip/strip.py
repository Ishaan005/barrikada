"""Remove suspicious Unicode and control characters from input text."""

from .utils import SUSPICIOUS_CHARS_RE, detect_control_characters, detect_homoglyphs


def strip_suspicious_characters(text: str) -> str:
    """Return text with suspicious formatting, homoglyphs, and controls normalized."""
    text = SUSPICIOUS_CHARS_RE.sub("", text)
    text, _ = detect_homoglyphs(text, normalize=True)
    text, _ = detect_control_characters(text, strip_controls=True)
    return text
