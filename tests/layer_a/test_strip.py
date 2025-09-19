from core.layer_a.strip import strip_suspicious_characters

samples = [
    "normal text",
    "hello\u200Bworld",                # zero width space
    "abc\u202Edef",                    # right-to-left override
    "safe\uFEFFdata",                  # BOM inside string
]

for s in samples:
    clean, meta = strip_suspicious_characters(s)
    print("Raw:     ", repr(s))
    print("Cleaned: ", repr(clean))
    print("Meta:    ", meta)
    print()
