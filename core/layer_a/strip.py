import re

SUSPICIOUS_CHARS_RE = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F"  # Zero-width & directional markers
    r"\u202A-\u202E"                    # LRE, RLE, PDF, LRO, RLO
    r"\u2066-\u2069"                    # LRI, RLI, FSI, PDI
    r"\uFEFF]"                          # BOM
)

def strip_suspicious_characters(text: str, replace_with: str = ''):
    matches = SUSPICIOUS_CHARS_RE.findall(text)
    suspicious_count = len(matches)
    clean_text = SUSPICIOUS_CHARS_RE.sub(replace_with, text)
    return clean_text, {
        "suspicious_count": suspicious_count,
        "was_modified": clean_text != text
    }

