import ftfy
import re

def normalise_punctuation_and_whitespace(text:str):
    """Use ftfy to fix punctuation and whitespace issues."""
    #Step 1
    fixed = ftfy.fix_text(text)

    #Step 2
    quotes_map = {
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "«": '"', "»": '"',
    }

    for k, v in quotes_map.items():
        fixed = fixed.replace(k, v)

    #Step 3
    dashes_map = {
        "–": "-",   # en-dash
        "—": "-",   # em-dash
        "−": "-",   # minus
    }

    for k, v in dashes_map.items():
        fixed = fixed.replace(k, v)

    #Step 4
    fixed = re.sub(r"\s+", " ", fixed)       # collapse spaces/tabs/newlines into one space
    fixed = fixed.strip()

    #step 5
    canonical_newlines = text.replace("\r\n", "\n").replace("\r", "\n")

    return {
        "original": text,
        "normalised": fixed,
        "normalised_newlines": canonical_newlines,
    }