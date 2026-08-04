---
paths:
  - "**/*.py"
---

## Python code style

Formatting is `ruff format` — PEP 8-aligned, line length 100 (project standard; PEP 8 recommends 79). `ruff check` enforces our selected rule subset (`E4`, `E7`, `E9`, `F`, `I`, `PLC0415`, `PLC2701`, and modern typing rules `UP006`, `UP007`, `UP035`, `UP045`) — see `[tool.ruff.lint]` in `pyproject.toml`. Run manual formatting or rely on the automatic PostToolUse hook before completing tasks.

- **Top-level imports by default.** Ordinary imports belong at module scope. A local import is
  allowed only for intentional lazy loading, optional-dependency fallback, required initialization
  order, or a circular dependency. Explain the exception and add `# noqa: PLC0415` to that import.

- **Don't import other modules' private names.** Symbols prefixed with `_` are module-private. If another module needs one, promote it to public (rename without the leading underscore) rather than cross-importing `_foo`. Enforced by ruff `PLC2701`.

- **Modern type syntax for Python 3.10+.** Prefer `list[str]`, `dict[str, int]` (PEP 585) and `X | None` (PEP 604) over `typing.List` / `typing.Optional`.
