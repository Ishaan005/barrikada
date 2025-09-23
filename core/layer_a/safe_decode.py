import chardet
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safe_decoder")

def safe_decode(raw_bytes: bytes, decode_errors: str = "replace", suspicious_threshold: int = 0, 
                preferred_encodings: Optional[list[str]] = None, confidence_threshold: float = 0.7):
    """
    Safely decode raw bytes to Unicode string with robust encoding detection.
    
    Args:
        raw_bytes (bytes): Input byte sequence.
        decode_errors (str): Error handler ('replace', 'ignore', 'surrogateescape').
        suspicious_threshold (int): Number of replacement chars before logging as suspicious.
        preferred_encodings (list): List of encodings to try in order before detection.
        confidence_threshold (float): Minimum confidence for chardet detection (0.0-1.0).
    
    Returns:
        tuple: (decoded_text, meta_info)
            decoded_text (str): Unicode text
            meta_info (dict): {
                "encoding_used": str,
                "decode_replacements": int,
                "suspicious": bool,
                "detection_confidence": float,
                "attempted_encodings": list,
                "utf8_decode_errors": int,
            }
    """
    if preferred_encodings is None:
        preferred_encodings = ["utf-8", "windows-1252", "iso-8859-1", "ascii"]
    
    attempted_encodings = []
    best_text = None
    best_encoding = None
    best_replacements = float('inf')
    detection_confidence = 0.0
    utf8_decode_errors = 0
    
    # First, always try UTF-8 to detect actual decode errors
    try:
        utf8_strict = raw_bytes.decode("utf-8", errors="strict")
        # If this succeeds, UTF-8 is perfect
        utf8_decode_errors = 0
    except UnicodeDecodeError as e:
        # Count how many bytes would cause errors
        utf8_decode_errors = len([b for b in raw_bytes if b > 127])  # Rough estimate
        logger.debug(f"UTF-8 strict decode failed: {utf8_decode_errors} potential error bytes")
    
    # Always run chardet detection for confidence scoring
    chardet_result = None
    if len(raw_bytes) > 0:
        try:
            chardet_result = chardet.detect(raw_bytes)
            detection_confidence = chardet_result.get("confidence", 0.0)
            logger.debug(f"Chardet detected: {chardet_result.get('encoding')} (confidence: {detection_confidence:.2f})")
        except Exception as e:
            logger.warning(f"Chardet detection failed: {e}")
            detection_confidence = 0.0
    
    # Step 1: Try preferred encodings in order
    for encoding in preferred_encodings:
            attempted_encodings.append(encoding)
            text = raw_bytes.decode(encoding, errors=decode_errors)
            replacements = text.count("�")  # replacement character
            
            logger.debug(f"Tried {encoding}: {replacements} replacements")
            
            # Keep track of best result so far
            if replacements < best_replacements:
                best_text = text
                best_encoding = encoding
                best_replacements = replacements
            
            # For UTF-8, if no replacements and no decode errors, it's perfect
            if encoding.lower() == "utf-8" and replacements == 0 and utf8_decode_errors == 0:
                break
            
    
    # Step 2: Try chardet suggestion if confidence is high enough
    if chardet_result and detection_confidence >= confidence_threshold:
        guessed_enc = chardet_result.get("encoding")
        
        # Only use detection if encoding is different from what we tried
        if (guessed_enc and 
            guessed_enc.lower() not in [enc.lower() for enc in attempted_encodings]):
                attempted_encodings.append(guessed_enc)
                text = raw_bytes.decode(guessed_enc, errors=decode_errors)
                replacements = text.count("�")
                
                logger.debug(f"Chardet {guessed_enc}: {replacements} replacements")
                
                # Use detected encoding if it's better
                if replacements < best_replacements:
                    best_text = text
                    best_encoding = guessed_enc
                    best_replacements = replacements
                    
    # Fallback: if somehow we have no result, force UTF-8 with replacement
    if best_text is None:
        logger.warning("All decoding attempts failed, forcing UTF-8 with replacement")
        best_text = raw_bytes.decode("utf-8", errors="replace")
        best_encoding = "utf-8"
        best_replacements = best_text.count("�")
        attempted_encodings.append("utf-8")
    
    # Enhanced suspicious detection
    suspicious = (
        best_replacements > suspicious_threshold or 
        utf8_decode_errors > suspicious_threshold or
        (detection_confidence > 0 and detection_confidence < 0.3)  # Very low confidence is suspicious
    )
    
    if suspicious:
        reasons = []
        if best_replacements > suspicious_threshold:
            reasons.append(f"{best_replacements} replacement chars")
        if utf8_decode_errors > suspicious_threshold:
            reasons.append(f"{utf8_decode_errors} UTF-8 decode errors")
        if detection_confidence > 0 and detection_confidence < 0.3:
            reasons.append(f"low detection confidence ({detection_confidence:.2f})")
        
        logger.warning(f"Suspicious decode: {', '.join(reasons)}, encoding={best_encoding}")
    
    if best_replacements > 0:
        logger.info(f"Decode had {best_replacements} replacement characters (encoding: {best_encoding})")
    
    return best_text, {
        "encoding_used": best_encoding,
        "decode_replacements": best_replacements,
        "suspicious": suspicious,
        "detection_confidence": detection_confidence,
        "attempted_encodings": attempted_encodings,
        "utf8_decode_errors": utf8_decode_errors,
    }

