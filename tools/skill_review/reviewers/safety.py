"""The safety/policy reviewer: deterministic static pre-filter, then an LLM rubric pass.

Per design D6, plain-text destructive commands are a NON-blocking signal to the LLM (so
SEED-OK4 "documents danger as a warning" is not statically false-vetoed under static-wins);
only *obfuscated* destructive payloads are a deterministic static veto (the base64
decode-and-rescan is shared with the security reviewer, so the security-critical rule is
identical). The LLM rubric judges harmful/disallowed behavior and missing termination/fallback.
"""
from __future__ import annotations

import re

from tools.skill_review.llm import (
    Finding,
    check_obfuscated_payload,
    load_rubric,
    matches_any,
    reviewable_text,
    review_with_llm,
    signal_finding,
    strip_zero_width,
)
from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import GraderType, Severity, Verdict

_RUBRIC = load_rubric("safety")   # read once at import; review() does no file I/O (INV-4)

_DESTRUCTIVE = (
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\bmkfs\b", re.I),
    re.compile(r"\bdd\s+(?:if|of)=", re.I),
    re.compile(r"git\s+push\s+(?:--force|-f)\b[^\n]*\b(?:main|master)\b", re.I),
    re.compile(r"\b(?:DROP|TRUNCATE)\s+TABLE\b", re.I),
    re.compile(r":\(\)\s*\{"),   # fork bomb
)


def _static_findings(write: SkillWrite) -> list[Finding]:
    text = strip_zero_width(reviewable_text(write))   # F3: zero-width can't split a keyword
    findings: list[Finding] = []
    findings += check_obfuscated_payload(text)   # obfuscated destructive = STATIC veto (D6)
    if matches_any(text, _DESTRUCTIVE):
        # Plain-text destructive = NON-blocking signal; the LLM judges instructs-vs-documents (OK4).
        findings.append(signal_finding(
            "destructive",
            "Mentions a destructive/irreversible operation; the LLM judges intent (instructs vs documents).",
            "body", Severity.HIGH))
    return findings


class SafetyReviewer(Reviewer):
    """Safety/policy hard-veto reviewer (static pre-filter → LLM rubric; static-wins)."""

    id = "safety"
    grader_type = GraderType.LLM

    def review(self, write: SkillWrite) -> Verdict:
        return review_with_llm(
            write,
            reviewer_id=self.id,
            rubric=_RUBRIC,
            task="skill_review_safety",
            static_findings=_static_findings(write),
        )
