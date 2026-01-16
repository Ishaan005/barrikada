import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Using all the modules I built to detect suspicious text
from core.layer_a.safe_decode import safe_decode
from core.layer_a.unicode import normalize_uniccode  
from core.layer_a.confusables import detect_confusables
from core.layer_a.strip.strip import strip_suspicious_characters
from core.layer_a.normalise_punctuation import normalise_punctuation_and_whitespace
from core.layer_a.detect_encodings import detect_and_decode_embedded

from models.LayerAResult import LayerAResult

import re
import time

# Check for bidirectional text override attacks
# Complete list per Unicode Bidirectional Algorithm
BIDI_OVERRIDE_CHARS = {
    '\u202A',  # LRE - Left-to-Right Embedding
    '\u202B',  # RLE - Right-to-Left Embedding  
    '\u202C',  # PDF - Pop Directional Formatting
    '\u202D',  # LRO - Left-to-Right Override
    '\u202E',  # RLO - Right-to-Left Override
    '\u2066',  # LRI - Left-to-Right Isolate
    '\u2067',  # RLI - Right-to-Left Isolate
    '\u2068',  # FSI - First Strong Isolate
    '\u2069',  # PDI - Pop Directional Isolate
    '\u061C',  # ALM - Arabic Letter Mark
}

def has_direction_override(text):
    """Check if text contains bidirectional override characters."""
    return any(char in text for char in BIDI_OVERRIDE_CHARS)


# Main function to process and analyze text
def analyze_text(input_bytes):
    """
    Analyze text for suspicious patterns and preprocessing
    
    Args:
        input_bytes: Raw input as bytes or string
        
    Returns:
        LayerAResult: Standardized result object
    """
    # Thread-safe local flags list
    flags = []
    
    def add_flag(name):
        flags.append(name)
    
    start_time = time.time()
  
    # Handle both bytes and string input
    if isinstance(input_bytes, str):
        input_bytes = input_bytes.encode('utf-8')
    
    # Step 1: Decode the bytes safely
    text, decode_info = safe_decode(input_bytes)
    
    # Flag suspicious encoding issues (replacement chars, low confidence)
    if decode_info.get('suspicious'):
        add_flag('suspicious_encoding')
    
    # Step 2: Fix unicode normalization  
    normalized = normalize_uniccode(text)
    
    # Step 3: Clean up punctuation and whitespace
    punct_result = normalise_punctuation_and_whitespace(normalized)
    cleaned_punct = punct_result['normalised']
    
    # Step 4: Check for confusable characters before stripping
    confusable_info = detect_confusables(cleaned_punct, threshold=0.15) 
    if confusable_info.get('is_dangerous') or confusable_info.get('is_mixed_script'):
        add_flag('confusable_chars')
    
    # Step 5: Remove suspicious characters (zero-width, bidi, homoglyphs, control chars)
    cleaned = strip_suspicious_characters(cleaned_punct)
    
    # Step 6: Look for embedded encodings (base64, hex, url, html)
    embedded_result = detect_and_decode_embedded(cleaned)
    if embedded_result.get('suspicious') or len(embedded_result.get('findings', [])) > 0:
        add_flag('embedded_encodings')
    
    # Step 7: Check for direction override attacks on ORIGINAL text
    # (before cleaning removed them)
    if has_direction_override(text):
        add_flag('direction_override')
    
    # Final canonical version (cleaned text is already normalized)
    final_text = cleaned
    
    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Calculate confidence score based on flag types
    confidence_score = _calculate_confidence(flags)
    
    # Return standardized result
    return LayerAResult(
        original_text=text,
        processed_text=final_text,
        flags=flags,
        suspicious=len(flags) > 0,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
        decode_info=decode_info,
        confusables=confusable_info,
        embedded=embedded_result
    )

def _calculate_confidence(flags: list) -> float:
    """
    Calculate confidence score based on detected flags
    
    High confidence detections: direction_override, embedded_encodings
    Medium confidence: confusable_chars, suspicious_encoding
    """
    if not flags:
        return 1.0  # High confidence in clean text
    
    high_confidence_flags = ['direction_override', 'embedded_encodings']
    medium_confidence_flags = ['confusable_chars', 'suspicious_encoding']
    
    has_high = any(flag in flags for flag in high_confidence_flags)
    has_medium = any(flag in flags for flag in medium_confidence_flags)
    
    if has_high:
        return 0.85  # High confidence in detection
    elif has_medium:
        return 0.70  # Medium confidence
    else:
        return 0.60  # Other flags