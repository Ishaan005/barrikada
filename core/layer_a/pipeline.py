# Using all the modules I built to detect suspicious text
from core.layer_a.safe_decode import safe_decode
from core.layer_a.unicode import normalize_uniccode  
from core.layer_a.confusables import detect_confusables
from core.layer_a.strip import strip_suspicious_characters
from core.layer_a.normalise_punctuation import normalise_punctuation_and_whitespace
from core.layer_a.detect_encodings import detect_and_decode_embedded

import base64
import re

# Simple list to track what we find
flags = []

def add_flag(name):
    flags.append(name)
    print(f"SUSPICIOUS: {name}")

# Simple function to check for weird unicode directions
def has_direction_override(text):
    bad_chars = ['\u202D', '\u202E', '\u2066', '\u2067', '\u2068']
    for char in bad_chars:
        if char in text:
            return True
    return False

# Check if text might be base64 encoded
def might_be_base64(text):
    if len(text) < 32:  # Increased minimum length to reduce false positives
        return False
    
    clean = text.replace(' ', '').replace('\n', '').replace('\t', '')
    
    # Must look like base64 (only valid chars)
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
    if not all(c in valid_chars for c in clean):
        return False
    
    # Must be proper length (multiple of 4)
    if len(clean) % 4 != 0:
        return False
    
    # Skip if it looks like a common benign pattern
    benign_patterns = [
        # JWT header patterns (typically short)
        r'^eyJ[A-Za-z0-9+/]+={0,2}$',  # JWT header
        # Very short base64 strings
        r'^[A-Za-z0-9+/]{8,24}={0,2}$',  # Short tokens
    ]
    
    for pattern in benign_patterns:
        if re.match(pattern, clean):
            return False
        
    try:
        decoded = base64.b64decode(clean)
        # Decoded content should be reasonable length and mostly printable
        if len(decoded) < 8:  # Increased minimum decoded length
            return False
        try:
            decoded_str = decoded.decode('utf-8', errors='strict')
            
            # Additional check: look for suspicious keywords in decoded content
            suspicious_keywords = ['ignore', 'previous', 'instruction', 'system', 'admin', 'password', 'secret']
            decoded_lower = decoded_str.lower()
            has_suspicious = any(keyword in decoded_lower for keyword in suspicious_keywords)
            
            # Only flag if it has suspicious content or is unusually long
            return has_suspicious or len(decoded) > 100
            
        except:
            return False  # Invalid UTF-8 suggests not meaningful base64
    except:
        return False

# Main function to process and analyze text
def analyze_text(input_bytes):
    global flags
    flags = []  # reset
    
    print("Starting Layer A analysis...")
    
    # Step 1: Decode the bytes safely
    text, decode_info = safe_decode(input_bytes)
    print(f"Decoded text: {repr(text)}")
    
    # Step 2: Fix unicode normalization  
    normalized = normalize_uniccode(text)
    
    # Step 3: Clean up punctuation and whitespace
    punct_result = normalise_punctuation_and_whitespace(normalized)
    cleaned_punct = punct_result['normalised']
    
    # Step 4: Remove suspicious characters
    cleaned, strip_info = strip_suspicious_characters(cleaned_punct)
    
    # Step 5: Look for embedded encodings
    embedded_result = detect_and_decode_embedded(cleaned)
    if embedded_result.get('found_encodings'):
        add_flag('embedded_encodings')
    
    # Step 6: Check for direction override attacks
    if has_direction_override(text):  # Check original text before cleaning
        add_flag('direction_override')
    
    # Step 7: Check if it looks like base64
    if might_be_base64(cleaned):
        add_flag('possible_base64')
    
    # Step 9: Check for confusable characters
    confusable_info = detect_confusables(cleaned, threshold=0.15) 

    # Only flag if it's actually dangerous or mixed script, not just having confusables
    if confusable_info.get('is_dangerous') or confusable_info.get('is_mixed_script'):
        add_flag('confusable_chars')
    
    # Final canonical version
    final_text = normalize_uniccode(cleaned)
    
    result = {
        'original': text,
        'final': final_text,
        'flags': flags,
        'suspicious': len(flags) > 0,
        'decode_info': decode_info,
        'confusables': confusable_info,
        'embedded': embedded_result
    }
    
    return result

# Test generated using AI (temporary)
def main():
    # Some test cases to try
    tests = [
        b"Hello world",  # normal text, should be clean
        b"\xd1\x80assword",  # cyrillic p looks like latin p: dangerous confusable
        b"VGhpcyBpcyBhIGhpZGRlbiBtZXNzYWdl",  # base64 encoded "This is a hidden message"
        "Hello\u202eworld".encode('utf-8'),  # direction override attack  
        b"Mixed \xd0\xa1yrillic text",  # mixed scripts: Cyrillic C in English
        b"Just normal English text here",  # another clean example
    ]
    
    print("Testing my prompt injection detector!\n")
    
    for i, test in enumerate(tests, 1):
        print(f"--- Test {i} ---")
        print(f"Input: {test}")
        
        result = analyze_text(test)
        
        print(f"Final text: {result['final']}")
        print(f"Flags found: {result['flags']}")
        print(f"Is suspicious: {result['suspicious']}")
        print()

# if __name__ == "__main__":
#     main()