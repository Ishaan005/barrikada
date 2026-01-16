import ftfy
import re

def normalise_punctuation_and_whitespace(text:str):
    """Use ftfy to fix punctuation and whitespace issues."""
    
    # Step 1: Canonicalize newlines
    fixed = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Step 2: ftfy fixes mojibake and other Unicode issues
    fixed = ftfy.fix_text(fixed)

    # Step 3: Normalize quote characters to ASCII equivalents
    quotes_map = {
        """: '"', """: '"',
        "'": "'", "'": "'",
        "«": '"', "»": '"',
    }

    for k, v in quotes_map.items():
        fixed = fixed.replace(k, v)

    # Step 4: Normalize dash/minus variants
    dashes_map = {
        "–": "-",   # en-dash
        "—": "-",   # em-dash
        "−": "-",   # minus
    }

    for k, v in dashes_map.items():
        fixed = fixed.replace(k, v)

    # Step 5: Normalize whitespace while preserving line structure
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
