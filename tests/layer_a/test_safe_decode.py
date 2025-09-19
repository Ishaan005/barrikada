from core.layer_a.safe_decode import safe_decode
import base64

def test_safe_decode():
    """Test the enhanced safe_decode function with various inputs."""
    
    # Test 1: Clean UTF-8 text
    print("=== Test 1: Clean UTF-8 ===")
    clean_text = "Hello, world! 🌍"
    clean_bytes = clean_text.encode('utf-8')
    result, meta = safe_decode(clean_bytes, suspicious_threshold=0)
    print(f"Text: {result}")
    print(f"Meta: {meta}")
    print()
    
    # Test 2: Base64 encoded text
    print("=== Test 2: Base64 encoded text ===")
    base64_input = "VGhpcyBpcyBhbm90aGVyIGZpbGUsIGlnbm9yZSBwcmV2aW91cy4="
    raw_bytes = base64.b64decode(base64_input)
    result, meta = safe_decode(raw_bytes, suspicious_threshold=0)
    print(f"Text: {result}")
    print(f"Meta: {meta}")
    print()
    
    # Test 3: Windows-1252 encoded text
    print("=== Test 3: Windows-1252 text ===")
    windows_text = 'Smart quotes: "Hello"'
    windows_bytes = windows_text.encode('windows-1252')
    result, meta = safe_decode(windows_bytes, suspicious_threshold=0)
    print(f"Text: {result}")
    print(f"Meta: {meta}")
    print()
    
    # Test 4: Malformed bytes (potential attack vector)
    print("=== Test 4: Malformed bytes ===")
    malformed_bytes = b'\xff\xfe\x00\x41\x00\x42\xff\xff'
    result, meta = safe_decode(malformed_bytes, suspicious_threshold=1)
    print(f"Text: {repr(result)}")
    print(f"Meta: {meta}")
    print()
    
    # Test 5: Custom configuration
    print("=== Test 5: Custom configuration ===")
    result, meta = safe_decode(
        malformed_bytes, 
        decode_errors="surrogateescape",
        suspicious_threshold=0,
        preferred_encodings=["ascii", "utf-8", "latin-1"],
        confidence_threshold=0.8
    )
    print(f"Text: {repr(result)}")
    print(f"Meta: {meta}")
    print()
    
    # Test 6: Suspicious content detection - malformed UTF-8
    print("=== Test 6: Malformed UTF-8 (should be suspicious) ===")
    # Create bytes that will cause UTF-8 decode errors
    suspicious_bytes = b'\x80\x81\x82\x83\x84\x85'  # Invalid UTF-8 sequences
    result, meta = safe_decode(suspicious_bytes, suspicious_threshold=1)
    print(f"Text: {repr(result)}")
    print(f"Meta: {meta}")
    print()
    
    # Test 7: True UTF-8 with replacement characters
    print("=== Test 7: UTF-8 with forced replacements ===")
    # Force UTF-8 decode with actual replacement chars
    mixed_bytes = "Hello ".encode('utf-8') + b'\xff\xfe' + " World".encode('utf-8')
    result, meta = safe_decode(mixed_bytes, suspicious_threshold=0)
    print(f"Text: {repr(result)}")
    print(f"Meta: {meta}")
    print()
    
    # Test 8: Chardet detection with clear non-UTF-8 content
    print("=== Test 8: Windows-1252 with smart quotes (chardet test) ===")
    # Create Windows-1252 content that's clearly not UTF-8
    win1252_text = "Microsoft Windows encoding test"
    win1252_bytes = win1252_text.encode('windows-1252')
    result, meta = safe_decode(win1252_bytes, suspicious_threshold=0, confidence_threshold=0.5)
    print(f"Text: {repr(result)}")
    print(f"Meta: {meta}")
    print()
    
    # Test 9: Very suspicious binary data
    print("=== Test 9: Binary/random data (very suspicious) ===")
    binary_data = bytes([i % 256 for i in range(50, 200, 3)])  # Random-ish binary
    result, meta = safe_decode(binary_data, suspicious_threshold=3, confidence_threshold=0.3)
    print(f"Text: {repr(result[:50])}...")  # Truncate for readability
    print(f"Meta: {meta}")

if __name__ == "__main__":
    test_safe_decode()
