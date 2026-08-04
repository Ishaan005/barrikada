"""Unicode helpers used by the Layer A suspicious-character stripper."""

import re
import unicodedata


SUSPICIOUS_CHARS_RE = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F"  # Zero-width and directional markers
    r"\u202A-\u202E"  # LRE, RLE, PDF, LRO, RLO
    r"\u2066-\u2069"  # LRI, RLI, FSI, PDI
    r"\uFEFF"  # BOM
    r"\u00A0"  # Non-breaking space
    r"\u1680"  # Ogham space mark
    r"\u2000-\u200A"  # En quad through hair space
    r"\u2028\u2029"  # Line and paragraph separators
    r"\u202F"  # Narrow no-break space
    r"\u205F"  # Medium mathematical space
    r"\u3000"  # Ideographic space
    r"\uFFF9-\uFFFB]"  # Interlinear annotation characters
)

HOMOGLYPH_PATTERNS = {
    # Cyrillic lookalikes
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    # Greek lookalikes
    "α": "a",
    "β": "B",
    "γ": "y",
    "ε": "e",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
    # Mathematical and punctuation lookalikes
    "ℓ": "l",
    "𝓁": "l",
    "𝐥": "l",
    "𝑙": "l",
    "⸻": "-",
    "–": "-",
    "—": "-",
}

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def detect_homoglyphs(text: str, normalize: bool = True) -> tuple[str, dict]:
    """Detect common homoglyphs and optionally replace them with ASCII equivalents."""
    homoglyphs_found = []
    normalized_text = text

    for position, char in enumerate(text):
        if char not in HOMOGLYPH_PATTERNS:
            continue

        replacement = HOMOGLYPH_PATTERNS[char]
        homoglyphs_found.append(
            {
                "char": char,
                "position": position,
                "replacement": replacement,
                "unicode_name": unicodedata.name(char, "UNKNOWN"),
            }
        )
        if normalize:
            normalized_text = normalized_text.replace(char, replacement)

    metadata = {
        "homoglyphs_found": homoglyphs_found,
        "homoglyph_count": len(homoglyphs_found),
        "was_normalized": normalized_text != text,
    }
    return normalized_text, metadata


def detect_control_characters(text: str, strip_controls: bool = True) -> tuple[str, dict]:
    """Detect control characters and optionally strip them from text."""
    control_chars = []
    for match in CONTROL_CHARS_RE.finditer(text):
        char = match.group()
        control_chars.append(
            {
                "char": repr(char),
                "position": match.start(),
                "ord": ord(char),
                "description": f"Control character 0x{ord(char):02X}",
            }
        )

    cleaned_text = CONTROL_CHARS_RE.sub("", text) if strip_controls else text
    return cleaned_text, {
        "control_chars": control_chars,
        "control_count": len(control_chars),
        "was_cleaned": cleaned_text != text,
    }
