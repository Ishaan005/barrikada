from core.layer_a.confusables import detect_confusables

samples = [
    "password",            # plain Latin
    "раssword",            # first letter is Cyrillic 'р' (looks like Latin p)
    "ΡΑSSWORD",            # Greek rho + alpha
    "سلام",                # Arabic
]

for s in samples:
    result = detect_confusables(s, expected_script="Latin", threshold=0.3)
    print("Input: ", repr(s))
    print("Meta:  ", result)
    print()