# Layer B Tests

Minimal test suite for the signature detection engine.

## Test Files

1. **`test_simple.py`** - Basic functionality test (run directly)
2. **`test_signature_engine.py`** - Unit tests (run with unittest)

## Running Tests

```bash
# Simple test
python tests/layer_b/test_simple.py

# Unit tests  
python tests/layer_b/test_signature_engine.py

# Or run from project root
python -m unittest tests.layer_b.test_signature_engine
```

## What's Tested

- ✅ Basic signature detection (allow/flag/block)
- ✅ High severity blocking
- ✅ Performance (sub-millisecond detection)
- ✅ Engine initialization

That's it! Minimal and focused.