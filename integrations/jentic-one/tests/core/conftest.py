"""Make Core-facing integration tests runnable from either project root."""

import sys
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

for path in (REPOSITORY_ROOT, INTEGRATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
