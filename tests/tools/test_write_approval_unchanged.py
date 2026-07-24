"""S4 independence/parity: the review gate never changes write_approval behavior (INV-3, R8).

  * gate OFF  ⇒ write_approval staging is byte-identical to pre-feature.
  * gate ON, veto ⇒ blocks BEFORE staging (review-first); no pending record is created.
  * gate ON, pass ⇒ a passing review still stages for human approval (non-interference).
  * both OFF ⇒ the pure pre-feature proceed path (returns None).
"""
from __future__ import annotations

import json

import pytest

import tools.skill_provenance as prov
import tools.write_approval as wa
from tools import skill_manager_tool as smt
from tools.skill_review import gate, record
from tools.skill_review.panel import Panel
from tools.skill_review.reviewers.base import Reviewer
from tools.skill_review.schema import Decision, GraderType, Severity, Verdict


class _FixedReviewer(Reviewer):
    id = "fake"
    grader_type = GraderType.DETERMINISTIC

    def __init__(self, verdict):
        self._verdict = verdict

    def review(self, write):
        return self._verdict


def _pass():
    return Panel([_FixedReviewer(Verdict("fake", Decision.PASS, Severity.INFO, 1.0))])


def _veto():
    return Panel([_FixedReviewer(Verdict(
        "security", Decision.VETO, Severity.CRITICAL, 1.0,
        rationale="bad"))])


@pytest.fixture
def bg_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
    record.reset()
    tok = prov.set_current_write_origin("background_review")
    try:
        yield tmp_path
    finally:
        prov.reset_current_write_origin(tok)


def _pending_dir(tmp_path):
    return tmp_path / "pending" / "skills"


class TestIndependence:
    def test_both_off_proceeds(self, bg_tmp, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: False)
        monkeypatch.setattr(wa, "write_approval_enabled", lambda sub: False)
        assert smt._apply_skill_write_gate("create", "s", content="x") is None

    def test_disabled_gate_has_no_side_effects(self, bg_tmp, monkeypatch):
        # DoD-1: with the gate off, review_skill_write is a true no-op — no counter is
        # bumped and no rejection record is written (parity: the disabled path adds nothing).
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: False)
        record.reset()
        assert gate.review_skill_write("create", "s", content="x").allow is True
        snap = record.snapshot()
        assert snap["seen"] == 0 and snap["allowed"] == 0 and snap["blocked_veto"] == 0
        assert not (bg_tmp / "skill_review" / "rejections").exists()

    def test_gate_off_write_approval_on_still_stages(self, bg_tmp, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: False)
        monkeypatch.setattr(wa, "write_approval_enabled", lambda sub: True)
        result = smt._apply_skill_write_gate("create", "s", content="x")
        payload = json.loads(result)
        assert payload.get("staged") is True  # unchanged write_approval staging
        # the disabled review gate injects/removes NO field in the staged payload (parity)
        assert set(payload.keys()) == {"success", "staged", "pending_id", "gist", "message"}

    def test_gate_on_veto_blocks_before_staging(self, bg_tmp, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)
        monkeypatch.setattr(gate, "_deadline_seconds", lambda: 0.0)
        monkeypatch.setattr(gate, "_build_panel", _veto)
        monkeypatch.setattr(wa, "write_approval_enabled", lambda sub: True)
        result = smt._apply_skill_write_gate("create", "bad", content="danger")
        assert json.loads(result)["success"] is False  # blocked, not staged
        # write_approval never staged it (review ran first)
        assert not _pending_dir(bg_tmp).exists() or not list(_pending_dir(bg_tmp).glob("*.json"))

    def test_gate_on_pass_write_approval_on_still_stages(self, bg_tmp, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)
        monkeypatch.setattr(gate, "_deadline_seconds", lambda: 0.0)
        monkeypatch.setattr(gate, "_build_panel", _pass)
        monkeypatch.setattr(wa, "write_approval_enabled", lambda sub: True)
        result = smt._apply_skill_write_gate("create", "s", content="clean")
        payload = json.loads(result)
        assert payload.get("staged") is True  # a passing review still defers to human staging
