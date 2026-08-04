# Python Code Style for AI Agents & Developers

To maintain a clean and consistent Python codebase in Barrikade, all developers and **AI agents** (Claude Code, Cursor, Copilot, Antigravity) must adhere to these code style rules.

> [!TIP]
> These styling rules are automatically checked and formatted by Ruff inside:
> 1. The Claude PostToolUse hook (`.claude/hooks/ruff-fix.sh`).
> 2. Manual execution of `ruff check` and `ruff format`.

## 1. Code Formatting & Formatting Standards

- **Standard**: Strictly aligned with [PEP 8](https://peps.python.org/pep-0008/).
- **Formatter**: Use `ruff format`.
- **Line Length**: Set to exactly **100 characters** (instead of standard PEP 8's 79). This project standard allows better readability of complex model and array code while keeping lines manageable.

---

## 2. Code Linting & Selected Rules

We enforce a highly optimized subset of Ruff lint rules. In `pyproject.toml`, our standard linter configuration enables:
- **`E` / `F`**: Pycodestyle and Pyflakes (standard syntax and styling errors).
- **`I`**: Isort (alphabetical, organized import block styling).
- **`PLC0415`**: Enforces top-level imports.
- **`PLC2701`**: Restricts importing private names from other modules.
- **`UP006`, `UP007`, `UP035`, `UP045`**: Enforce Python 3.10+ collection and union typing syntax.

---

## 3. Core Importing Standards

### Top-level Imports by Default (Enforced by `PLC0415`)
- Place ordinary imports at module scope, above class and function definitions.
- A local import is allowed only when it preserves intentional lazy loading, optional-dependency
  fallback, required initialization order, or resolves a circular dependency.
- Every exception must have a one-line explanation and a narrow `# noqa: PLC0415` on the import;
  module-wide exemptions are not allowed.

### No Private Name Imports (Enforced by `PLC2701`)
- Symbols prefixed with a single leading underscore (e.g., `_my_private_function`) are module-private.
- Do NOT import private symbols from one module into another. If another module needs access to a private symbol, refactor that symbol to be public (remove the leading underscore) rather than cross-importing private names.

---

## 4. Modern Type Syntax

Since this codebase targets **Python 3.10+**, you must use modern Python typing syntax rather than obsolete `typing` module wrappers:
- **Generic Collections ([PEP 585](https://peps.python.org/pep-0585/))**: Use native collections for type hints instead of `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Set`.
  - **Do**: `list[str]`, `dict[str, int]`, `tuple[int, float]`
  - **Don't**: `List[str]`, `Dict[str, int]`, `Tuple[int, float]`
- **Union Types ([PEP 604](https://peps.python.org/pep-0604/))**: Use the `|` operator for Union and Optional types instead of `typing.Union` and `typing.Optional`.
  - **Do**: `str | int`, `Path | None`
  - **Don't**: `Union[str, int]`, `Optional[Path]`
