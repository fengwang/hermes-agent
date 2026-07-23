"""Read-only config accessors for the skill-review gate (default OFF).

Mirrors ``tools.write_approval.write_approval_enabled``: a lazy import of
``hermes_cli.config``, a safe nested read via ``cfg_get``, and a fail-safe that
returns the default on any error. It reads ONLY the ``skills.review_gate.*``
namespace — it is independent of ``skills.write_approval`` (INV-3) — and requires
no edit to ``hermes_cli/config.py`` (an absent key resolves to the default).

This is a skeleton: nothing here is consulted by the live ``skill_manage`` write
path in S1. Wiring the gate into ``_apply_skill_write_gate`` is S4.
"""
from __future__ import annotations

from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a config value to bool, matching write_approval's tolerant parsing."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSY:
            return False
        return default
    if value is None:
        return default
    return bool(value)


def _read(*keys: str, default: Any) -> Any:
    """Read a nested key from the live config, defaulting on any error/absence."""
    try:
        from hermes_cli.config import cfg_get, load_config

        return cfg_get(load_config(), *keys, default=default)
    except Exception:
        return default


def review_gate_enabled() -> bool:
    """Whether the skill-review gate is enabled (``skills.review_gate.enabled``). Default: off."""
    return _as_bool(_read("skills", "review_gate", "enabled", default=False), default=False)


def reviewer_enabled(reviewer_id: str, *, default: bool = True) -> bool:
    """Whether a specific reviewer is enabled (``skills.review_gate.reviewers.<id>``).

    Defaults to on when the gate is enabled; individual reviewers can be toggled off.
    """
    return _as_bool(
        _read("skills", "review_gate", "reviewers", reviewer_id, default=default),
        default=default,
    )
