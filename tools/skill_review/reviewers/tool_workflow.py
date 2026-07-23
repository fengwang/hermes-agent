"""The deterministic tool/workflow-integrity reviewer (static depth).

Static lint for the workflow-integrity classes the other reviewers do NOT own (E1), operating
ONLY on structured surfaces (fenced code blocks + numbered step lists), never freeform prose,
with an M2 exemption on each detector:

  * **non-idempotent retry** — a fenced/step unit that retries a non-idempotent *network/state*
    mutation with no idempotency guard.
  * **use-before-create ordering** — a numbered step list that uses a resource (by physical step
    position) before the step that creates it.
  * **missing precondition** — a numbered step list with a destructive workflow step and no
    verify/backup/guard token in that step or an earlier one.

DEFERRED to the dynamic phase (NOT statically vetoed):
  * **hallucinated tool (SEED-TW3)** — the Hermes tool universe (plugins / MCP / configurable
    toolsets / dynamically-registered families) cannot be statically enumerated, so any
    snapshot-denylist would false-veto legitimate skills (M2, unrecoverable under static-wins).
    Grounding a tool reference needs the LIVE tool set / sandbox (PRD §9 "tool/workflow →
    sandboxed replay"). See docs/session_3/design.md §7/§13.
  * **unsafe-shell / RCE** — OWNED by the security reviewer (E1 no double-penalty); not re-flagged.

Patterns are tunable defaults calibrated in S5 (like S1/S2 statics). Depth is ``STATIC`` and every
verdict carries a ``deferred-dynamic-checks`` note (R2/R15). Pure Calculation: ``review()`` does
no I/O. See docs/skill_review_static_limitations.md.
"""
from __future__ import annotations

import re

# Pure, LLM-agnostic primitives (importing llm.py runs no model and does no I/O). See the note in
# formal_invariants.py: a future non-S3 refactor should hoist these into ``reviewers/_common.py``.
from tools.skill_review.llm import Finding, reviewable_text, veto_finding
from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import (
    Decision,
    Depth,
    Evidence,
    GraderType,
    Severity,
    Verdict,
)

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_STEP_RE = re.compile(r"^[ \t]*(\d+)[.)]\s+(.*)$", re.M)
_BACKTICK = re.compile(r"`([a-zA-Z][\w-]*)`")
_CREATE_TOKEN = re.compile(
    r"\b(?:create|creating|creates|define|defines|provision|provisions|generate|generates|"
    r"initialize|initialise|set up|make|makes)\b[^`\n]*`([a-zA-Z][\w-]*)`", re.I)

# workflow-integrity signals (deliberately disjoint from security._UNSAFE_SHELL/_RCE and
# safety._DESTRUCTIVE — E1). Tunable defaults (S5). The mutating set is scoped to non-idempotent
# NETWORK/STATE mutations; idempotent-by-nature ops (local `write`, `delete`) are excluded so a
# benign retry is not false-vetoed (M2).
_RETRY = re.compile(
    r"\b(?:retry|retries|retrying|for attempt|while not done|until (?:it )?succe|"
    r"again on (?:failure|error))\b", re.I)
_MUTATING = re.compile(
    r"\b(?:POST|PUT|create|creates|insert|charge|deploy|send|push)\b", re.I)
_IDEMPOTENT = re.compile(
    r"\b(?:idempoten\w*|idempotency[- ]key|if[- ]not[- ]exists|already exists|--if-match|dedup\w*)\b",
    re.I)
_WF_MUTATE = re.compile(
    r"\b(?:overwrite|force[- ]?push|truncate|wipe|purge|reset --hard|drop table)\b", re.I)
_GUARD = re.compile(
    r"\b(?:verify|check|ensure|confirm|if exists|back(?: |-)?up|dry[- ]run|validate|precondition|"
    r"make sure)\b", re.I)

_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}

_DEFERRED_NOTE = Evidence(
    locator="deferred-dynamic-checks",
    detail=(
        "tool_workflow ran at STATIC depth; NOT checked here: HALLUCINATED-tool grounding against "
        "the live tool set (the tool universe — plugins/MCP/toolsets — cannot be statically "
        "enumerated; needs the sandbox/replay depth), body-level tool-invocation grounding, true "
        "idempotence/ordering across a real run, and endpoint reachability. Unsafe-shell/RCE is "
        "owned by the security reviewer (not re-flagged here). See "
        "docs/skill_review_static_limitations.md."
    ),
)


def _fenced_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _FENCE_RE.finditer(text or "")]


def _numbered_steps(text: str) -> list[str]:
    """The body of each numbered step, in the order they appear in the source (physical order)."""
    return [m.group(2) for m in _STEP_RE.finditer(text or "")]


def _check_retry(text: str) -> list[Finding]:
    """Veto a single unit (a fenced block or a single step body) that retries a non-idempotent
    mutation with no idempotency guard. Co-occurrence is scoped to ONE unit so a retry in one step
    and an unrelated mutation in another do not combine into a false veto (M2)."""
    for unit in _fenced_blocks(text) + _numbered_steps(text):
        if _RETRY.search(unit) and _MUTATING.search(unit) and not _IDEMPOTENT.search(unit):
            return [veto_finding(Severity.MEDIUM, "non-idempotent-retry",
                                 "Retries a non-idempotent mutation with no idempotency guard "
                                 "(idempotency key / if-not-exists); a retry may duplicate the effect.",
                                 "workflow")]
    return []


def _check_ordering(text: str) -> list[Finding]:
    """Veto a step list that uses a resource before the step that creates it (by physical order)."""
    steps = _numbered_steps(text)
    if len(steps) < 2:
        return []
    created_at: dict[str, int] = {}
    used_at: dict[str, int] = {}
    for position, body in enumerate(steps):
        for tok in _BACKTICK.findall(body):
            used_at.setdefault(tok.lower(), position)
        for m in _CREATE_TOKEN.finditer(body):
            created_at.setdefault(m.group(1).lower(), position)
    for tok in sorted(created_at):
        first_use = used_at.get(tok)
        if first_use is not None and first_use < created_at[tok]:
            return [veto_finding(Severity.MEDIUM, "tool-ordering",
                                 f"Resource '{tok}' is used at step position {first_use + 1} before it "
                                 f"is created at step position {created_at[tok] + 1} "
                                 "(use-before-create ordering).", "workflow")]
    return []


def _check_precondition(text: str) -> list[Finding]:
    """Veto a destructive workflow step with no verify/backup/guard token in it or an earlier step."""
    steps = _numbered_steps(text)
    if not steps:
        return []
    mutating = next((pos for pos, body in enumerate(steps) if _WF_MUTATE.search(body)), None)
    if mutating is None:
        return []
    # A guard in the mutating step itself (or an earlier one) exempts it (M2).
    if any(_GUARD.search(body) for pos, body in enumerate(steps) if pos <= mutating):
        return []
    return [veto_finding(Severity.MEDIUM, "missing-precondition",
                         f"Destructive workflow step at position {mutating + 1} runs with no "
                         "verify/backup/precondition step.", "workflow")]


def _workflow_findings(write: SkillWrite) -> list[Finding]:
    text = reviewable_text(write)
    return _check_retry(text) + _check_ordering(text) + _check_precondition(text)


def _aggregate(findings: list[Finding]) -> Verdict:
    blocking = [f for f in findings if f.blocking]
    decision = Decision.VETO if blocking else Decision.PASS
    severity = (max((f.severity for f in findings), key=lambda s: _SEVERITY_RANK[s])
                if findings else Severity.INFO)
    evidence = tuple(Evidence(locator=f.locator, detail=f.detail) for f in findings) + (_DEFERRED_NOTE,)
    scope = tuple(dict.fromkeys([f.scope for f in findings] + ["static-depth"]))
    rationale = ("Tool/workflow-integrity veto: " + "; ".join(f.detail for f in blocking)
                 if blocking else "Tool/workflow-integrity checks passed (static depth).")
    return Verdict(
        reviewer=ToolWorkflowReviewer.id,
        decision=decision,
        severity=severity,
        confidence=1.0,
        evidence=evidence,
        impacted_scope=scope,
        rationale=rationale,
        depth=Depth.STATIC,
    )


class ToolWorkflowReviewer(Reviewer):
    """Deterministic reviewer for retry/ordering/precondition workflow lint (static depth)."""

    id = "tool_workflow"
    grader_type = GraderType.DETERMINISTIC

    def review(self, write: SkillWrite) -> Verdict:
        return _aggregate(_workflow_findings(write))
