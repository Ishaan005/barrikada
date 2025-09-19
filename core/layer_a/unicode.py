import unicodedata

def normalize_uniccode(text: str, form: str = "NFKC") -> str:
    """Normalize unicode text to NFKC form."""
    return unicodedata.normalize(form, text)#type:ignore