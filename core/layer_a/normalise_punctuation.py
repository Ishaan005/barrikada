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

    # Step 5: Collapse whitespace (spaces/tabs/newlines) into single space
    fixed = re.sub(r"\s+", " ", fixed)
    fixed = fixed.strip()

    return {
        "original": text,
        "normalised": fixed,
    }
