import regex
from collections import Counter
from confusable_homoglyphs import confusables
from typing import Dict, Any

# Reference the paper 
def script_distribution(text: str):
    scripts = Counter()
    for ch in text:
        match = regex.match(r"\p{Script=Latin}|\p{Script=Cyrillic}|\p{Script=Greek}|\p{Script=Arabic}|\p{Script=Han}", ch)
        if match:
            if match.group(0):
                if regex.match(r"\p{Script=Latin}", ch):
                    scripts["Latin"] += 1
                elif regex.match(r"\p{Script=Cyrillic}", ch):
                    scripts["Cyrillic"] += 1
                elif regex.match(r"\p{Script=Greek}", ch):
                    scripts["Greek"] += 1
                elif regex.match(r"\p{Script=Arabic}", ch):
                    scripts["Arabic"] += 1
                elif regex.match(r"\p{Script=Han}", ch):
                    scripts["Han"] += 1
        else:
            scripts["Other"] += 1
    return dict(scripts)

def detect_confusables(text: str, expected_script = "Latin", threshold = 0.1) -> Dict[str, Any]:
    scripts = script_distribution(text)
    total = sum(scripts.values())
    expected_count = scripts.get(expected_script, 0)
    non_expected = total - expected_count

    if total:
        percent_non_expected = (non_expected / total) 
    else:
        percent_non_expected = 0.0

    amogus_sus = percent_non_expected > threshold

    # Check for dangerous confusable characters
    is_dangerous = False
    is_mixed = False
    confusable_details = None
    
    try:
        is_dangerous = confusables.is_dangerous(text)
        is_mixed = confusables.is_mixed_script(text)
        # Get detailed confusable information (check if text has confusables)
        confusable_details = confusables.is_confusable(text)
    except Exception as e:
        # Handle any errors in confusable detection
        pass

    if is_dangerous or is_mixed or confusable_details:
        amogus_sus = True

    return {
        "script_counts": scripts,
        "percent_non_expected": percent_non_expected,
        "is_dangerous": is_dangerous,
        "is_mixed_script": is_mixed,
        "has_confusables": bool(confusable_details),
        "confusable_details": confusable_details,
        "suspicious": amogus_sus,
    }