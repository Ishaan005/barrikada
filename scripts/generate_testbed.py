#!/usr/bin/env python3
"""
Generate tool document testbed using LLM.

Usage:
    python3 generate_testbed.py              # 50 benign + 50 malicious
    python3 generate_testbed.py 100 100      # Custom amounts
"""

import sys
from core.tool_hijacker.tool_generator import TestbedGenerator


def main():
    # Default values
    num_benign = 50
    num_malicious = 50
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        num_benign = int(sys.argv[1])
        num_malicious = int(sys.argv[2])
    
    # Generate testbed
    try:
        generator = TestbedGenerator()
        filename = generator.generate_dataset(num_benign, num_malicious)
        
        print(f"\nSuccess!")
        print(f"File: {filename}")

    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
