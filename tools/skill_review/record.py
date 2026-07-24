"""Rejection-record writer + observability counters for the live skill-review gate (S4).

Two responsibilities, both deliberately kept out of the frozen panel/reviewers:

  * **Rejection record** — a best-effort, *non-recursing* audit artifact written to
    ``<HERMES_HOME>/skill_review/rejections/<key>.json`` when the gate blocks an agent write.
    It is written by direct, **atomic** file I/O (temp + ``os.replace``) — never through
    ``skill_manage``, the review gate, or the memory tool — so it cannot re-enter the gate
    (R7) and a concurrent same-key block cannot leave a partial file (F10). It is keyed by
    ``sha256(name + reviewable)`` so identical rejected writes dedupe to one file, and carries
    the contract ``reason`` (``blocked-by-veto`` / ``blocked-by-reviewer-unavailable``, INV-8)
    plus a finer ``subreason`` and a ``quality_signal``. Persisted rationale/evidence is
    length-capped and secret-redacted (the reviewed skill is untrusted, F8). It carries **no
    wall-clock** — the record content is deterministic; file mtime carries recency.

  * **Counters** — a tiny thread-safe tally (the background fork and the curator are separate
    daemon threads) so operators can detect a learning-stall (FR12). Counters never influence
    the decision.

INV-5: this module writes an *audit* artifact only; it never writes the skill or touches
lifecycle state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Persisted-detail bounds (F8): the reviewed skill is untrusted, and reviewer rationale/evidence
# can echo attacker-influenced text. Cap length and redact obvious secret shapes before persist.
_MAX_DETAIL = 500
_SECRET_RE = re.compile(
    r"AKIA[0-9A-Z]{16}"                                   # AWS access-key id
    r"|(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}"      # GitHub token
    r"|(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])",  # long base64-ish blob
)


def redact_secrets(text: str) -> str:
    """Mask obvious secret-shaped substrings (used for both the agent message and the record)."""
    return _SECRET_RE.sub("[REDACTED]", text or "")


def _cap_detail(text: str) -> str:
    return redact_secrets(text or "")[:_MAX_DETAIL]


# --------------------------------------------------------------------------- #
# Rejection key (pure Calculation)
# --------------------------------------------------------------------------- #
def rejection_key(target, reviewable: str) -> str:
    """A stable 16-hex key for a rejected write: ``sha256(name \\0 reviewable)``.

    Keying on the reviewed content (not the whole payload) means a regenerated identical skill
    maps to the same record — the dedupe/suppression handle.
    """
    h = hashlib.sha256()
    h.update((getattr(target, "name", "") or "").encode("utf-8"))
    h.update(b"\0")
    h.update((reviewable or "").encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Rejection record (Action — best-effort, non-recursing, atomic)
# --------------------------------------------------------------------------- #
def _rejections_dir():
    return get_hermes_home() / "skill_review" / "rejections"


def _capped_verdicts(record) -> list:
    """Verdicts with rationale + evidence detail length-capped and secret-redacted (F8)."""
    out = []
    for v in record.verdicts:
        d = v.to_dict()
        d["rationale"] = _cap_detail(d.get("rationale", ""))
        d["evidence"] = [
            {"locator": str(e.get("locator", "")), "detail": _cap_detail(str(e.get("detail", "")))}
            for e in d.get("evidence", [])
        ]
        out.append(d)
    return out


def write_rejection(record, *, reason: str, subreason: str, quality_signal: bool,
                    reviewable: str) -> str | None:
    """Persist a rejection record atomically. Best-effort: any failure is logged and swallowed
    (the block decision has already been made and must not depend on this write).

    Returns the key on success, ``None`` on failure.
    """
    try:
        key = rejection_key(record.target, reviewable)
        payload = {
            "key": key,
            "schema_version": record.schema_version,
            "reason": reason,               # contract label (blocked-by-veto / …-unavailable)
            "subreason": subreason,         # finer detail (veto / reviewer_unavailable / …_error)
            "quality_signal": quality_signal,
            "target": record.target.to_dict(),
            "decision": record.decision.value,
            "verdicts": _capped_verdicts(record),
        }
        d = _rejections_dir()
        d.mkdir(parents=True, exist_ok=True)
        final = d / f"{key}.json"
        tmp = d / f".{key}.{uuid.uuid4().hex}.tmp"   # unique per writer ⇒ no concurrent-write race
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, final)                        # atomic publish (F10)
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
        "retries": 0,
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
