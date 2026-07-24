"""Rejection-record writer + observability counters for the live skill-review gate (S4).

Two responsibilities, both deliberately kept out of the frozen panel/reviewers:

  * **Rejection record** — a best-effort, *non-recursing* audit artifact written to
    ``<HERMES_HOME>/skill_review/rejections/<key>.json`` when the gate blocks an
    agent write. It is written by direct file I/O only — never through ``skill_manage``,
    the review gate, or the memory tool — so it cannot re-enter the gate (R7). It is
    keyed by ``sha256(name + reviewable)`` so identical rejected writes dedupe to one
    file, and carries a ``reason`` + ``quality_signal`` so a future consumer (or #412)
    can distinguish a real quality veto from an infra failure (INV-8). It carries **no
    wall-clock** — the record content is deterministic; file mtime carries recency.

  * **Counters** — a tiny thread-safe tally (the background fork and the curator are
    separate daemon threads) so operators can detect a learning-stall (FR12). Counters
    never influence the decision.

INV-5: this module writes an *audit* artifact only; it never writes the skill or touches
lifecycle state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Rejection key (pure Calculation)
# --------------------------------------------------------------------------- #
def rejection_key(target, reviewable: str) -> str:
    """A stable 16-hex key for a rejected write: ``sha256(name \\0 reviewable)``.

    Keying on the reviewed content (not the whole payload) means a regenerated
    identical skill maps to the same record — the dedupe/suppression handle.
    """
    h = hashlib.sha256()
    h.update((getattr(target, "name", "") or "").encode("utf-8"))
    h.update(b"\0")
    h.update((reviewable or "").encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Rejection record (Action — best-effort, non-recursing)
# --------------------------------------------------------------------------- #
def _rejections_dir():
    return get_hermes_home() / "skill_review" / "rejections"


def write_rejection(record, *, reason: str, quality_signal: bool, reviewable: str) -> str | None:
    """Persist a rejection record. Best-effort: any failure is logged and swallowed
    (the block decision has already been made and must not depend on this write).

    Returns the key on success, ``None`` on failure.
    """
    try:
        key = rejection_key(record.target, reviewable)
        payload = {
            "key": key,
            "schema_version": record.schema_version,
            "reason": reason,
            "quality_signal": quality_signal,
            "target": record.target.to_dict(),
            "decision": record.decision.value,
            "verdicts": [v.to_dict() for v in record.verdicts],
        }
        d = _rejections_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return key
    except Exception as e:  # best-effort, non-recursing
        logger.warning("skill-review: rejection record write failed: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Counters (thread-safe)
# --------------------------------------------------------------------------- #
_lock = threading.Lock()


def _fresh() -> dict:
    return {
        "seen": 0,
        "allowed": 0,
        "blocked_veto": 0,
        "blocked_unavailable": 0,
        "deadline": 0,
        "vetoed": {},
    }


_counters = _fresh()


def bump(name: str, n: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


def bump_reviewer(reviewer_id: str) -> None:
    with _lock:
        _counters["vetoed"][reviewer_id] = _counters["vetoed"].get(reviewer_id, 0) + 1


def snapshot() -> dict:
    with _lock:
        snap = dict(_counters)
        snap["vetoed"] = dict(_counters["vetoed"])
        return snap


def reset() -> None:
    """Test hook: clear all counters."""
    global _counters
    with _lock:
        _counters = _fresh()
