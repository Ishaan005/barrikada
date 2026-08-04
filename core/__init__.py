"""Core package initialization."""

import os
import platform


def _configure_safe_runtime() -> None:
    if platform.system() != "Darwin":
        return

    if os.getenv("BARRIKADA_SAFE_RUNTIME") == "0":
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env_var, "1")

    try:
        # Keep torch optional and avoid importing it on non-macOS runtimes.
        import torch  # noqa: PLC0415
    except ImportError:
        return

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ValueError, RuntimeError):
        pass


_configure_safe_runtime()
