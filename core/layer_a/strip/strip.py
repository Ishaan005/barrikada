from .utils import *

def strip_suspicious_characters(
        text: str, 
        replace_with: str = '', 
        normalize_homoglyphs: bool = True,
        strip_controls: bool = True,
        detailed_analysis: bool = True):
    """
    suspicious character detection and removal for prompt injection mitigation.
    
    Args:
        text: Input text to process
        replace_with: String to replace suspicious characters with
        normalize_homoglyphs: If True, normalize homoglyphs to ASCII equivalents
        strip_controls: If True, remove control characters
        detailed_analysis: If True, perform comprehensive analysis
        
    Returns:
        tuple: (cleaned_text, comprehensive_metadata)
    """
    original_text = text
    metadata = {}
    processing_steps = []
    
    # Step 1: Handle suspicious Unicode formatting characters
    matches = SUSPICIOUS_CHARS_RE.findall(text)
    suspicious_count = len(matches)
    text = SUSPICIOUS_CHARS_RE.sub(replace_with, text)
    
    metadata['suspicious_formatting'] = {
        'count': suspicious_count,
        'characters_found': list(set(matches)),
        'was_modified': text != original_text
    }
    processing_steps.append('suspicious_formatting')
    
    # Step 2: Handle homoglyphs
    if normalize_homoglyphs:
        text, homoglyph_meta = detect_homoglyphs(text, normalize=True)
        metadata['homoglyphs'] = homoglyph_meta
        processing_steps.append('homoglyph_normalization')
    
    # Step 3: Handle control characters
    if strip_controls:
        text, control_meta = detect_control_characters(text, strip_controls=True)
        metadata['control_chars'] = control_meta
        processing_steps.append('control_character_removal')
    
    # Step 4: Detect suspicious Unicode categories
    suspicious_unicode = detect_suspicious_unicode_categories(text)
    metadata['suspicious_unicode_categories'] = {
        'characters': suspicious_unicode,
        'count': len(suspicious_unicode)
    }
    
    # Step 5: Comprehensive analysis (if requested)
    if detailed_analysis:
        metadata['text_analysis'] = analyze_text_structure(text)
        processing_steps.append('structural_analysis')
    
    # Add processing steps to metadata
    metadata['processing_steps'] = processing_steps
    
    # Summary
    total_suspicious = (
        suspicious_count + 
        metadata.get('homoglyphs', {}).get('homoglyph_count', 0) +
        metadata.get('control_chars', {}).get('control_count', 0) +
        len(suspicious_unicode)
    )
    
    metadata['summary'] = {
        'original_length': len(original_text),
        'final_length': len(text),
        'total_suspicious_elements': total_suspicious,
        'was_modified': text != original_text,
        'risk_level': (
            'high' if total_suspicious > 10 else
            'medium' if total_suspicious > 3 else
            'low'
        )
    }
    
    return text, metadata

