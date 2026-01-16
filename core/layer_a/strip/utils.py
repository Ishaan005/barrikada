import re
import unicodedata

# The below corpus was enhanced using AI
# suspicious character detection for prompt injection attacks
SUSPICIOUS_CHARS_RE = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F"  # Zero-width & directional markers
    r"\u202A-\u202E"                    # LRE, RLE, PDF, LRO, RLO
    r"\u2066-\u2069"                    # LRI, RLI, FSI, PDI
    r"\uFEFF"                           # BOM
    r"\u00A0"                           # Non-breaking space
    r"\u1680"                           # Ogham space mark
    r"\u2000-\u200A"                    # En quad through hair space
    r"\u2028\u2029"                     # Line/paragraph separator
    r"\u202F"                           # Narrow no-break space
    r"\u205F"                           # Medium mathematical space
    r"\u3000"                           # Ideographic space
    r"\uFFF9-\uFFFB]"                   # Interlinear annotation chars
)

# Homoglyph detection patterns - common substitutions in attacks
HOMOGLYPH_PATTERNS = {
    # Latin lookalikes
    'а': 'a',  # Cyrillic 'a'
    'е': 'e',  # Cyrillic 'e'  
    'о': 'o',  # Cyrillic 'o'
    'р': 'p',  # Cyrillic 'p'
    'с': 'c',  # Cyrillic 'c'
    'у': 'y',  # Cyrillic 'y'
    'х': 'x',  # Cyrillic 'x'
    'і': 'i',  # Cyrillic 'i'
    'ј': 'j',  # Cyrillic 'j'
    'ѕ': 's',  # Cyrillic 's'
    
    # Greek lookalikes
    'α': 'a',  # Greek alpha
    'β': 'B',  # Greek beta (looks like B)
    'γ': 'y',  # Greek gamma
    'ε': 'e',  # Greek epsilon
    'ο': 'o',  # Greek omicron
    'ρ': 'p',  # Greek rho
    'τ': 't',  # Greek tau (looks like t)
    'υ': 'u',  # Greek upsilon
    'χ': 'x',  # Greek chi
    
    # Mathematical and other lookalikes
    'ℓ': 'l',  # Script small l
    '𝓁': 'l',  # Mathematical script small l
    '𝐥': 'l',  # Mathematical bold small l
    '𝑙': 'l',  # Mathematical italic small l
    '⸻': '-',  # Two-em dash (looks like hyphen)
    '–': '-',  # En dash
    '—': '-',  # Em dash
}

# Control character detection
CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')

# Suspicious Unicode categories that might hide attacks
SUSPICIOUS_CATEGORIES = {
    'Cf',  # Other, format (invisible formatting)
    'Mn',  # Mark, nonspacing (combining chars that modify appearance)
    'Me',  # Mark, enclosing (combining chars that enclose)
}

def detect_homoglyphs(text: str, normalize: bool = True):
    """
    Detect and optionally normalize homoglyph characters that could be used in attacks.
    
    Args:
        text: Input text to analyze
        normalize: If True, replace homoglyphs with ASCII equivalents
        
    Returns:
        tuple: (normalized_text, metadata)
    """
    homoglyphs_found = []
    normalized_text = text

    
    for i, char in enumerate(text):
        if char in HOMOGLYPH_PATTERNS:
            unicode_name = unicodedata.name(char, 'UNKNOWN')
            homoglyphs_found.append({
                'char': char,
                'position': i,
                'replacement': HOMOGLYPH_PATTERNS[char],
                'unicode_name': unicode_name
            })
            
            if normalize:
                normalized_text = normalized_text.replace(char, HOMOGLYPH_PATTERNS[char])
    
    metadata = {
        'homoglyphs_found': homoglyphs_found,
        'homoglyph_count': len(homoglyphs_found),
        'was_normalized': normalized_text != text
    }
    return normalized_text, metadata

# def detect_suspicious_unicode_categories(text: str):
#     """
#     Detect characters in suspicious Unicode categories that could hide attacks.
#     """
#     suspicious_chars = []
    
#     for i, char in enumerate(text):
#         category = unicodedata.category(char)
#         if category in SUSPICIOUS_CATEGORIES:
#             suspicious_chars.append({
#                 'char': char,
#                 'position': i,
#                 'category': category,
#                 'unicode_name': unicodedata.name(char, 'UNKNOWN')
#             })
    
#     return suspicious_chars

def detect_control_characters(text: str, strip_controls: bool = True):
    """
    Detect and optionally strip control characters.
    
    Args:
        text: Input text to analyze
        strip_controls: If True, remove control characters
        
    Returns:
        tuple: (cleaned_text, metadata)
    """
    control_matches = list(CONTROL_CHARS_RE.finditer(text))
    control_chars = []
    
    for match in control_matches:
        char = match.group()
        control_chars.append({
            'char': repr(char),  # Safe representation
            'position': match.start(),
            'ord': ord(char),
            'description': f'Control character 0x{ord(char):02X}'
        })
    
    cleaned_text = CONTROL_CHARS_RE.sub('', text) if strip_controls else text
    
    return cleaned_text, {
        'control_chars': control_chars,
        'control_count': len(control_chars),
        'was_cleaned': cleaned_text != text
    }

# #TODO: Evaluation required
# def analyze_text_structure(text: str):
#     """
#     Analyze text structure for potential injection patterns.
#     """
#     analysis = {
#         'length': len(text),
#         'line_count': text.count('\n') + 1,
#         'word_count': len(text.split()),
#         'char_counts': {},
#         #'suspicious_patterns': []
#     }
    
#     # Character frequency analysis
#     for char in text:
#         category = unicodedata.category(char)
#         analysis['char_counts'][category] = analysis['char_counts'].get(category, 0) + 1
        
#     #TODO: Evaluate whether to include this. Prompts should be allowed to be long. Overly long propts would be blocked anyway?

#     # # Look for suspicious patterns
#     # if text.count('\n') > 50:
#     #     analysis['suspicious_patterns'].append('excessive_newlines')
    
#     # if len(text) > 10000:
#     #     analysis['suspicious_patterns'].append('excessive_length')
    
#     # Look for repeated suspicious chars
#     for char in ['\u200B', '\u200C', '\u200D', '\uFEFF']:
#         if text.count(char) > 5:
#             analysis['suspicious_patterns'].append(f'repeated_{unicodedata.name(char, "unknown").lower()}')
    
#     return analysis