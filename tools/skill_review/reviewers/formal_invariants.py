"""The deterministic formal-invariants reviewer (static depth).

Formalizes two *purely-encodable* Hermes invariants as read-only verdicts (INV-4), without
re-implementing or weakening the existing guards (INV-6):

  * **permission-monotonicity** — a ``patch`` whose new ``allowed-tools`` set widens vs the old
    one (adds a tool, or becomes a wildcard where the old was specific). The patch delta is the
    ONLY monotonicity signal available in the frozen ``SkillWrite``; ``create``/``edit`` carry no
    prior version, so they are not monotonicity-checked (documented static limitation, R15).
  * **identity-consistency** — a ``create``/``edit`` whose frontmatter ``name`` differs from the
    write's ``name``. This is net-new: the contract reviewer checks that ``name`` *exists* and
    validates the write-name *format*, but never that the two are *equal* — so there is no
    double-penalty (E1).

State-dependent invariants — owned-skill (``created_by``), pinned protection, and name-collision
— depend on sidecar/filesystem state a pure reviewer cannot read and that the existing guards
(``_background_review_write_guard``, ``_create_skill``'s collision check) already enforce. This
reviewer DEFERS them; S3's ``test_skill_review_formal.py::TestGuardsIntact`` proves those guards
still fire independently. See docs/skill_review_static_limitations.md and docs/session_3/design.md
§5-§8/§13.

Depth is ``STATIC`` and every verdict carries a machine-actionable ``deferred-dynamic-checks``
note (R2/R15). Pure Calculation: ``review()`` performs no I/O.
"""
from __future__ import annotations

import re

import yaml

# Pure, LLM-agnostic primitives (importing llm.py runs no model and does no I/O). A deterministic
# reviewer importing from a module named ``llm`` is a known smell; a future non-S3 refactor should
# hoist these shared pure helpers into a ``reviewers/_common.py`` (out of this session's blast radius).
from tools.skill_review.llm import Finding, veto_finding
from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import (
    Decision,
    Depth,
    Evidence,
    GraderType,
    Severity,
    Verdict,
)

_ALLOWED_TOOLS_RE = re.compile(r"allowed-tools\s*:\s*(.+)", re.I)
_FRONTMATTER_CLOSE_RE = re.compile(r"\n---\s*\n")

_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}

# Appended to EVERY verdict (pass and veto) so downstream consumers never over-trust static depth (R2).
_DEFERRED_NOTE = Evidence(
    locator="deferred-dynamic-checks",
    detail=(
        "formal-invariants ran at STATIC depth; NOT checked here: cross-version permission "
        "monotonicity for create/edit (no prior version in the frozen SkillWrite), owned-skill "
        "provenance, pinned protection, and name collision (all enforced by the existing guards). "
        "See docs/skill_review_static_limitations.md."
    ),
)


def _parse_tool_list(value: str) -> frozenset[str]:
    """Parse an ``allowed-tools`` value into a lowercased set. Handles ``[a, b]``, ``a, b``, ``*``."""
    v = value.strip().strip("[]").strip()
    if not v:
        return frozenset()
    return frozenset(t.strip().strip("\"'").lower() for t in v.split(",") if t.strip())


def _extract_allowed_tools(text: str | None) -> frozenset[str] | None:
    """Return the declared ``allowed-tools`` set found in ``text``, or ``None`` if none is declared."""
    if not text:
        return None
    match = _ALLOWED_TOOLS_RE.search(text)
    if not match:
        return None
    return _parse_tool_list(match.group(1))


def _frontmatter_name(content: str | None) -> str | None:
    """Best-effort parse of the frontmatter ``name`` value, else ``None``."""
    text = (content or "").lstrip("\ufeff")
    if not text.startswith("---"):
        return None
    close = _FRONTMATTER_CLOSE_RE.search(text[3:])
    if not close:
        return None
    try:
        parsed = yaml.safe_load(text[3 : close.start() + 3])
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None
    name = parsed["name"]
    return None if name is None else str(name).strip()


def _check_monotonicity(write: SkillWrite) -> list[Finding]:
    """Veto a ``patch`` that widens the ``allowed-tools`` set vs the prior (delta-only, I3)."""
    if write.action != "patch":
        return []
    old = _extract_allowed_tools(write.old_string)
    new = _extract_allowed_tools(write.new_string)
    if old is None or new is None:
        # No baseline in the delta ⇒ widening cannot be proven ⇒ do not veto (M2-safe).
        return []
    if "*" in old:
        return []  # old already unrestricted ⇒ new cannot widen it
    if "*" in new:
        return [veto_finding(Severity.HIGH, "permission-monotonicity",
                             "Patch widens allowed-tools to a wildcard '*' (prior set was specific).",
                             "allowed-tools")]
    added = new - old
    if added:
        return [veto_finding(Severity.HIGH, "permission-monotonicity",
                             f"Patch widens allowed-tools: adds {sorted(added)} not in the prior set.",
                             "allowed-tools")]
    return []


def _check_identity(write: SkillWrite) -> list[Finding]:
    """Veto a ``create``/``edit`` whose frontmatter ``name`` ≠ the write's ``name`` (net-new; E1-clean)."""
    if write.action not in ("create", "edit"):
        return []
    declared = _frontmatter_name(write.content)
    if not declared:
        return []  # existence is the contract reviewer's job — not an identity veto (E1)
    # Case-insensitive: the write name is authoritative (lowercased by VALID_NAME_RE); a case-only
    # frontmatter difference is a benign slip, not an identity mismatch (M2).
    if declared.casefold() != str(write.name).strip().casefold():
        return [veto_finding(Severity.MEDIUM, "identity",
                             f"Frontmatter name {declared!r} does not match the write name "
                             f"{write.name!r}; the declared identity must equal the requested identity.",
                             "name")]
    return []


def _formal_findings(write: SkillWrite) -> list[Finding]:
    return _check_monotonicity(write) + _check_identity(write)


def _aggregate(findings: list[Finding]) -> Verdict:
    blocking = [f for f in findings if f.blocking]
    decision = Decision.VETO if blocking else Decision.PASS
    severity = (max((f.severity for f in findings), key=lambda s: _SEVERITY_RANK[s])
                if findings else Severity.INFO)
    evidence = tuple(Evidence(locator=f.locator, detail=f.detail) for f in findings) + (_DEFERRED_NOTE,)
    scope = tuple(dict.fromkeys([f.scope for f in findings] + ["static-depth"]))
    rationale = ("Formal-invariants veto: " + "; ".join(f.detail for f in blocking)
                 if blocking else "Formal-invariants checks passed (static depth).")
    return Verdict(
        reviewer=FormalInvariantsReviewer.id,
        decision=decision,
        severity=severity,
        confidence=1.0,
        evidence=evidence,
        impacted_scope=scope,
        rationale=rationale,
        depth=Depth.STATIC,
    )


class FormalInvariantsReviewer(Reviewer):
    """Deterministic reviewer formalizing permission-monotonicity + identity-consistency (static)."""

    id = "formal"
    grader_type = GraderType.DETERMINISTIC

    def review(self, write: SkillWrite) -> Verdict:
        return _aggregate(_formal_findings(write))
