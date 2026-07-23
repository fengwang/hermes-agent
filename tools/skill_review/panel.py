"""The reviewer panel: order deterministic-first, run, aggregate hard-veto.

The panel is a pure calculation over its reviewers. Two run modes (project_contract
§4):
  * ``GATE`` — short-circuit on the first veto (save tokens at gate time).
  * ``EVAL`` — run every reviewer to completion (independent scoring at eval time).

Aggregation is hard-veto: the record's decision is ``VETO`` iff any executed
reviewer vetoes; there is no scoring/weighting in the MVP.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import (
    SCHEMA_VERSION,
    Decision,
    DecisionRecord,
    GraderType,
    Verdict,
    WriteTarget,
)


class PanelMode(str, Enum):
    GATE = "gate"
    EVAL = "eval"


# Lower sorts first. Deterministic reviewers run before LLM reviewers so the cheap,
# un-talk-out-able checks can veto (and short-circuit) before any model is called.
_GRADER_ORDER = {GraderType.DETERMINISTIC: 0, GraderType.LLM: 1}


class Panel:
    def __init__(self, reviewers: Iterable[Reviewer]):
        self._reviewers: tuple[Reviewer, ...] = tuple(reviewers)

    def _ordered(self) -> list[Reviewer]:
        # ``sorted`` is stable, so insertion order is preserved within a grader group.
        return sorted(
            self._reviewers, key=lambda r: _GRADER_ORDER.get(r.grader_type, 99)
        )

    def review(self, write: SkillWrite, mode: PanelMode = PanelMode.GATE) -> DecisionRecord:
        verdicts: list[Verdict] = []
        for reviewer in self._ordered():
            verdict = reviewer.review(write)
            verdicts.append(verdict)
            if mode is PanelMode.GATE and verdict.decision is Decision.VETO:
                break

        decision = (
            Decision.VETO
            if any(v.decision is Decision.VETO for v in verdicts)
            else Decision.PASS
        )
        target = WriteTarget(
            action=write.action,
            name=write.name,
            origin=write.origin,
            file_path=write.file_path,
        )
        return DecisionRecord(
            schema_version=SCHEMA_VERSION,
            target=target,
            decision=decision,
            verdicts=tuple(verdicts),
        )
