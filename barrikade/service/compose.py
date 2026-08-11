"""Internal Compose entry point; a false integration toggle leaves Core stopped."""

from __future__ import annotations

import os


def main() -> int:
    enabled = os.getenv("JENTIC__BARRIKADE__ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return 0
    import uvicorn  # noqa: PLC0415

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
