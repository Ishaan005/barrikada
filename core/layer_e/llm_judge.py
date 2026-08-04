"""Compatibility export for the local Qwen3Guard judge."""

from .local_judge import Qwen3GuardJudge as LLMJudge


__all__ = ["LLMJudge"]
