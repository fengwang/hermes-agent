"""Frozen verdict / decision-record schema for the skill-review gate (S1 freeze).

This module is the initiative's forward-compatibility surface toward weighted
voting (issue #412). It is intentionally built on the standard library only
(``dataclasses`` + ``enum``), per project invariant INV-10, so the wire format is
fully under our control.

Freeze contract (see docs/session_1/design.md §5):
  * ``SCHEMA_VERSION`` is bumped only for breaking changes; after S1, changes are
    additive-only.
  * The enums ``Decision`` / ``Severity`` / ``Depth`` are closed and stable.
    Forward-compatibility is provided by **additive optional fields**, which
    ``from_dict`` ignores when unknown (a future reviewer field must never break
    an older reader).
  * Records/verdicts carry no wall-clock or ordering state, so deterministic
    reviewers are byte-identical across runs (metric M3). A logging envelope, if
    any, is added by the caller (S4), not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.0"


class SchemaError(ValueError):
    """Raised when a verdict/record is malformed or carries an unknown enum value."""


class Decision(str, Enum):
    PASS = "pass"
    VETO = "veto"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Depth(str, Enum):
    """Analysis depth marker (risk R2): a full check vs. a static-shallow one."""

    FULL = "full"
    STATIC = "static"


class GraderType(str, Enum):
    """How a reviewer grades. Drives panel ordering (deterministic-first).

    Not a field of ``Verdict`` — it is a property of the reviewer. Kept here so
    the taxonomy has a single home for later phases (red-team / human graders).
    """

    DETERMINISTIC = "deterministic"
    LLM = "llm"


def _as_enum(enum_cls: type[Enum], value: Any) -> Any:
    """Coerce ``value`` to ``enum_cls`` (accepting an instance or its string value).

    Raises ``SchemaError`` on an unknown value — enum values are additive-only and
    a new value requires a ``SCHEMA_VERSION`` bump; we never silently coerce.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise SchemaError(f"invalid {enum_cls.__name__} value: {value!r}") from exc


def _as_confidence(value: Any) -> float:
    """Coerce/validate a confidence to a float in [0, 1]; raise ``SchemaError`` otherwise."""
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("confidence must be a number in [0, 1]") from exc
    if not (0.0 <= confidence <= 1.0):
        raise SchemaError(f"confidence out of range [0, 1]: {confidence}")
    return confidence


def _as_sequence(value: Any, what: str) -> tuple:
    """Coerce a list/tuple to a tuple; reject a bare string or non-sequence.

    Guards the frozen list-typed fields: a bare string must not be silently
    char-split into per-item scopes, and a scalar must raise ``SchemaError`` rather
    than leak a ``TypeError`` (the documented failure type is ``SchemaError``).
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise SchemaError(f"{what} must be a list or tuple, got {type(value).__name__}")


def _require_mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise SchemaError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Evidence:
    """A single machine-actionable pointer: WHERE (``locator``) and WHAT (``detail``)."""

    locator: str
    detail: str

    def to_dict(self) -> dict:
        return {"locator": self.locator, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Any) -> "Evidence":
        d = _require_mapping(data, "evidence item")
        return cls(locator=str(d.get("locator", "")), detail=str(d.get("detail", "")))


@dataclass(frozen=True)
class Verdict:
    """One reviewer's machine-actionable judgement of a single skill write."""

    reviewer: str
    decision: Decision
    severity: Severity
    confidence: float
    evidence: tuple[Evidence, ...] = ()
    impacted_scope: tuple[str, ...] = ()
    rationale: str = ""
    depth: Depth = Depth.FULL

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _as_enum(Decision, self.decision))
        object.__setattr__(self, "severity", _as_enum(Severity, self.severity))
        object.__setattr__(self, "depth", _as_enum(Depth, self.depth))
        object.__setattr__(self, "confidence", _as_confidence(self.confidence))
        object.__setattr__(self, "evidence", _as_sequence(self.evidence, "evidence"))
        object.__setattr__(
            self,
            "impacted_scope",
            tuple(str(s) for s in _as_sequence(self.impacted_scope, "impacted_scope")),
        )
        if not isinstance(self.reviewer, str) or not self.reviewer:
            raise SchemaError("reviewer must be a non-empty string")
        for item in self.evidence:
            if not isinstance(item, Evidence):
                raise SchemaError("evidence items must be Evidence instances")

    def to_dict(self) -> dict:
        return {
            "reviewer": self.reviewer,
            "decision": self.decision.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "impacted_scope": list(self.impacted_scope),
            "rationale": self.rationale,
            "depth": self.depth.value,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Verdict":
        d = _require_mapping(data, "verdict")
        return cls(
            reviewer=str(d.get("reviewer", "")),
            decision=_as_enum(Decision, d.get("decision")),
            severity=_as_enum(Severity, d.get("severity")),
            confidence=_as_confidence(d.get("confidence")),
            evidence=tuple(Evidence.from_dict(e) for e in _as_sequence(d.get("evidence", ()), "evidence")),
            impacted_scope=_as_sequence(d.get("impacted_scope", ()), "impacted_scope"),
            rationale=str(d.get("rationale", "")),
            depth=_as_enum(Depth, d.get("depth", Depth.FULL.value)),
        )


@dataclass(frozen=True)
class WriteTarget:
    """Identifies the gated write a record is about (the record's subject)."""

    action: str
    name: str
    origin: str
    file_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "name": self.name,
            "origin": self.origin,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "WriteTarget":
        d = _require_mapping(data, "target")
        return cls(
            action=str(d.get("action", "")),
            name=str(d.get("name", "")),
            origin=str(d.get("origin", "")),
            file_path=d.get("file_path"),
        )


@dataclass(frozen=True)
class DecisionRecord:
    """The panel's aggregate result for one write: a hard-veto decision + verdicts."""

    schema_version: str
    target: WriteTarget
    decision: Decision
    verdicts: tuple[Verdict, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _as_enum(Decision, self.decision))
        object.__setattr__(self, "verdicts", _as_sequence(self.verdicts, "verdicts"))
        if not isinstance(self.target, WriteTarget):
            raise SchemaError("target must be a WriteTarget")
        for v in self.verdicts:
            if not isinstance(v, Verdict):
                raise SchemaError("verdicts must be Verdict instances")

    @property
    def is_blocked(self) -> bool:
        return self.decision is Decision.VETO

    def blocking_verdicts(self) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.decision is Decision.VETO)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "target": self.target.to_dict(),
            "decision": self.decision.value,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DecisionRecord":
        d = _require_mapping(data, "decision record")
        return cls(
            schema_version=str(d.get("schema_version", "")),
            target=WriteTarget.from_dict(d.get("target")),
            decision=_as_enum(Decision, d.get("decision")),
            verdicts=tuple(Verdict.from_dict(v) for v in _as_sequence(d.get("verdicts", ()), "verdicts")),
        )
