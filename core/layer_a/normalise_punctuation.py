import re

import ftfy


def collapse_separated_characters(text):
    """
    Collapse text where individual characters are separated by single spaces.
    e.g. "I g n o r e" -> "Ignore"

    Requires at least 4 consecutive spaced characters to avoid collapsing
    legitimate short words.
    """

    # Collapse spaced-out characters
    # "I g n o r e" -> "Ignore"
    text = re.sub(r"(?:\S ){4,}\S", lambda m: m.group().replace(" ", ""), text)

    # Collapse other separators between chars (dots, dashes, underscores)
    # "i.g.n.o.r.e" or "i-g-n-o-r-e"
    text = re.sub(r"(?:\w[._\-]){4,}\w", lambda m: re.sub(r"[._\-]", "", m.group()), text)

    return text


def normalise_punctuation_and_whitespace(text):
    """Use ftfy to fix punctuation and whitespace issues."""

    # Step 1: Canonicalize newlines
    fixed = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2: ftfy fixes mojibake and other Unicode issues
    if not fixed.isascii():
        fixed = ftfy.fix_text(fixed)

    # Step 3: Normalize quote characters to ASCII equivalents
    quotes_map = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "«": '"',
        "»": '"',
    }

    for k, v in quotes_map.items():
        fixed = fixed.replace(k, v)

    # Step 4: Normalize dash/minus variants
    dashes_map = {
        "–": "-",  # en-dash
        "—": "-",  # em-dash
        "−": "-",  # minus
    }

    for k, v in dashes_map.items():
        fixed = fixed.replace(k, v)

    # Step 5: Collapse separated characters (e.g. "I g n o r e" -> "Ignore")
    fixed = collapse_separated_characters(fixed)

    # Step 6: Normalize whitespace while preserving line structure
    # First, collapse multiple newlines to single newline
    fixed = re.sub(r"\n+", "\n", fixed)
    # Then collapse horizontal whitespace (spaces/tabs) to single space
    fixed = re.sub(r"[^\S\n]+", " ", fixed)
    # Clean up spaces around newlines
    fixed = re.sub(r" *\n *", "\n", fixed)
    fixed = fixed.strip()

    return {
        "original": text,
        "normalised": fixed,
    }
