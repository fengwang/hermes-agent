"""Skill-review quality-gate subsystem (default-off skeleton).

A small panel of orthogonal reviewers that inspects an *agent-created* skill
write and returns a machine-actionable verdict/decision-record. In this session
(S1) the package is a skeleton: the schema is frozen, the read-only ``Reviewer``
interface and the hard-veto ``Panel`` exist, and the deterministic contract/schema
reviewer is implemented. Nothing here is consulted by the live ``skill_manage``
write path yet — wiring into ``_apply_skill_write_gate`` is S4.

Keep import side effects minimal; callers may import concrete submodules directly.
"""
from tools.skill_review.config import review_gate_enabled, reviewer_enabled
from tools.skill_review.panel import Panel, PanelMode
from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.reviewers.contract import ContractReviewer
from tools.skill_review.schema import (
    SCHEMA_VERSION,
    Decision,
    DecisionRecord,
    Depth,
    Evidence,
    GraderType,
    SchemaError,
    Severity,
    Verdict,
    WriteTarget,
)

__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "Decision",
    "Severity",
    "Depth",
    "GraderType",
    "Evidence",
    "Verdict",
    "WriteTarget",
    "DecisionRecord",
    "Reviewer",
    "SkillWrite",
    "ContractReviewer",
    "Panel",
    "PanelMode",
    "review_gate_enabled",
    "reviewer_enabled",
]
