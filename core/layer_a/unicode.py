import unicodedata


def normalize_uniccode(text, form="NFKC"):
    """Normalize unicode text to NFKC form."""
    if text.isascii():
        return text
    if form == "NFC":
        return unicodedata.normalize("NFC", text)
    if form == "NFD":
        return unicodedata.normalize("NFD", text)
    if form == "NFKD":
        return unicodedata.normalize("NFKD", text)
    return unicodedata.normalize("NFKC", text)
