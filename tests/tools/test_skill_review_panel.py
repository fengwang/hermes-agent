"""Panel tests: hard-veto aggregation, deterministic-first ordering, gate/eval modes.

See docs/session_1/design.md §7 (Requirement: Hard-veto panel with ordering and
modes) and project_contract.md §4 (gate-time short-circuit vs eval-time run-all).
"""
from tools.skill_review.panel import Panel, PanelMode
from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import (
    Decision,
    Depth,
    DecisionRecord,
    GraderType,
    Severity,
    Verdict,
)


class FakeReviewer(Reviewer):
    """A reviewer that records that it ran and returns a preset decision."""

    def __init__(self, rid, grader_type, decision, calls):
        self.id = rid
        self.grader_type = grader_type
        self._decision = decision
        self._calls = calls

    def review(self, write: SkillWrite) -> Verdict:
        self._calls.append(self.id)
        return Verdict(
            reviewer=self.id,
            decision=self._decision,
            severity=Severity.HIGH if self._decision is Decision.VETO else Severity.INFO,
            confidence=1.0,
            evidence=(),
            impacted_scope=(),
            rationale="fake",
            depth=Depth.FULL,
        )


def _write(**kw) -> SkillWrite:
    base = dict(action="create", name="foo", origin="background_review")
    base.update(kw)
    return SkillWrite(**base)


def _det(rid, decision, calls):
    return FakeReviewer(rid, GraderType.DETERMINISTIC, decision, calls)


def _llm(rid, decision, calls):
    return FakeReviewer(rid, GraderType.LLM, decision, calls)


class TestHardVeto:
    def test_all_pass_is_pass(self):
        calls = []
        panel = Panel([_det("a", Decision.PASS, calls), _det("b", Decision.PASS, calls)])
        record = panel.review(_write(), PanelMode.EVAL)
        assert isinstance(record, DecisionRecord)
        assert record.decision is Decision.PASS
        assert record.is_blocked is False

    def test_any_veto_blocks(self):
        calls = []
        panel = Panel([_det("a", Decision.PASS, calls), _det("b", Decision.VETO, calls)])
        record = panel.review(_write(), PanelMode.EVAL)
        assert record.decision is Decision.VETO
        assert record.is_blocked is True
        assert record.blocking_verdicts()[0].reviewer == "b"

    def test_empty_panel_passes(self):
        record = Panel([]).review(_write(), PanelMode.EVAL)
        assert record.decision is Decision.PASS
        assert record.verdicts == ()


class TestOrdering:
    def test_deterministic_reviewers_run_before_llm(self):
        calls = []
        # LLM reviewer listed first, deterministic second — panel must reorder.
        panel = Panel([_llm("llm", Decision.PASS, calls), _det("det", Decision.PASS, calls)])
        panel.review(_write(), PanelMode.EVAL)
        assert calls == ["det", "llm"]

    def test_stable_within_grader_group(self):
        calls = []
        panel = Panel(
            [_det("d1", Decision.PASS, calls), _det("d2", Decision.PASS, calls),
             _llm("l1", Decision.PASS, calls)]
        )
        panel.review(_write(), PanelMode.EVAL)
        assert calls == ["d1", "d2", "l1"]


class TestModes:
    def test_gate_mode_short_circuits_on_first_veto(self):
        calls = []
        # deterministic veto should stop the panel before the LLM reviewer runs.
        panel = Panel([_det("det", Decision.VETO, calls), _llm("llm", Decision.PASS, calls)])
        record = panel.review(_write(), PanelMode.GATE)
        assert calls == ["det"]  # llm never ran
        assert record.decision is Decision.VETO
        assert [v.reviewer for v in record.verdicts] == ["det"]

    def test_eval_mode_runs_all_reviewers(self):
        calls = []
        panel = Panel([_det("det", Decision.VETO, calls), _llm("llm", Decision.PASS, calls)])
        record = panel.review(_write(), PanelMode.EVAL)
        assert calls == ["det", "llm"]  # both ran despite the early veto
        assert record.decision is Decision.VETO
        assert [v.reviewer for v in record.verdicts] == ["det", "llm"]

    def test_gate_is_default_mode(self):
        calls = []
        panel = Panel([_det("det", Decision.VETO, calls), _llm("llm", Decision.PASS, calls)])
        panel.review(_write())  # no mode arg
        assert calls == ["det"]


class TestRecordTarget:
    def test_target_reflects_write(self):
        calls = []
        panel = Panel([_det("a", Decision.PASS, calls)])
        record = panel.review(
            _write(action="write_file", name="bar", file_path="references/x.md"),
            PanelMode.EVAL,
        )
        assert record.target.action == "write_file"
        assert record.target.name == "bar"
        assert record.target.origin == "background_review"
        assert record.target.file_path == "references/x.md"
        assert record.schema_version == "1.0"
