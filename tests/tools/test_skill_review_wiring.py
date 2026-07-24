"""S4 seam tests: the review-gate branch inside
``skill_manager_tool._apply_skill_write_gate`` maps GateDecision → tool_error / fall-through,
runs after the bypass, before write_approval, and only for agent-origin gated actions."""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch as _patch

import pytest

import tools.skill_provenance as prov
from tools import skill_manager_tool as smt
from tools.skill_review import gate, record


@contextmanager
def _skill_dir(tmp_path):
    """Point skill_manage's storage at a tmp dir (mirrors test_skill_manager_tool._skill_dir)."""
    with _patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         _patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


_VALID_SKILL = (
    "---\nname: e2e-skill\ndescription: A reusable class-level capability.\n---\n\n# E2E\nBody.\n"
)


@pytest.fixture
def clean_counters():
    record.reset()
    yield


# --------------------------------------------------------------------------- #
# Seam mapping (isolated: review_skill_write is faked)
# --------------------------------------------------------------------------- #
class TestSeamMapping:
    def test_blocked_maps_to_tool_error(self, monkeypatch):
        monkeypatch.setattr(gate, "review_skill_write",
                            lambda *a, **k: gate._wa.GateDecision(blocked=True, message="NOPE"))
        result = smt._apply_skill_write_gate("create", "s", content="x")
        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload["success"] is False
        assert "NOPE" in json.dumps(payload)

    def test_allow_falls_through(self, monkeypatch):
        # review gate allows; write_approval is off by default ⇒ the seam returns None (proceed)
        monkeypatch.setattr(gate, "review_skill_write",
                            lambda *a, **k: gate._wa.GateDecision(allow=True))
        assert smt._apply_skill_write_gate("create", "s", content="x") is None

    def test_bypass_skips_review(self, monkeypatch):
        def _must_not_run(*a, **k):
            raise AssertionError("review gate must not run during approved-pending replay")

        monkeypatch.setattr(gate, "review_skill_write", _must_not_run)
        tok = smt._skill_gate_bypass.set(True)
        try:
            assert smt._apply_skill_write_gate("create", "s", content="x") is None
        finally:
            smt._skill_gate_bypass.reset(tok)

    def test_review_runs_before_write_approval(self, monkeypatch):
        # A block must short-circuit before the write_approval consult is reached.
        def _wa_must_not_run(*a, **k):
            raise AssertionError("write_approval must not be consulted after a review block")

        monkeypatch.setattr(gate, "review_skill_write",
                            lambda *a, **k: gate._wa.GateDecision(blocked=True, message="X"))
        monkeypatch.setattr(smt, "_apply_skill_write_gate",
                            smt._apply_skill_write_gate)  # keep the real seam
        # patch write_approval.evaluate_gate via the module the seam imports
        import tools.write_approval as wa
        monkeypatch.setattr(wa, "evaluate_gate", _wa_must_not_run)
        assert json.loads(smt._apply_skill_write_gate("create", "s", content="x"))["success"] is False


# --------------------------------------------------------------------------- #
# Seam integration (real review_skill_write with a faked panel)
# --------------------------------------------------------------------------- #
def _veto_panel():
    from tests.tools.test_skill_review_gate import _FixedReviewer, _verdict
    from tools.skill_review.panel import Panel
    return Panel([_FixedReviewer(_verdict("security", "shell"))])


class TestSeamIntegration:
    def test_enabled_background_veto_blocks(self, clean_counters, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)
        monkeypatch.setattr(gate, "_deadline_seconds", lambda: 0.0)
        monkeypatch.setattr(gate, "_build_panel", _veto_panel)
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        tok = prov.set_current_write_origin("background_review")
        try:
            result = smt._apply_skill_write_gate("create", "bad", content="danger")
        finally:
            prov.reset_current_write_origin(tok)
        assert isinstance(result, str)
        assert json.loads(result)["success"] is False

    def test_enabled_foreground_falls_through(self, clean_counters, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)

        def _boom():
            raise AssertionError("panel must not be built for a foreground write")

        monkeypatch.setattr(gate, "_build_panel", _boom)
        # default origin is foreground; write_approval off ⇒ seam returns None
        assert smt._apply_skill_write_gate("create", "s", content="x") is None


# --------------------------------------------------------------------------- #
# Codex F2 — gate import/run failure fails CLOSED when enabled (not fail-open)
# --------------------------------------------------------------------------- #
class TestGateUnavailable:
    def test_fails_closed_when_enabled(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("gate package broken")

        monkeypatch.setattr(gate, "review_skill_write", _boom)
        monkeypatch.setattr(smt, "_review_gate_enabled_fallback", lambda: True)
        tok = prov.set_current_write_origin("background_review")
        try:
            result = smt._apply_skill_write_gate("create", "s", content="x")
        finally:
            prov.reset_current_write_origin(tok)
        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload["success"] is False
        assert "unavailable" in json.dumps(payload).lower()

    def test_feature_absent_when_disabled(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("gate package broken")

        monkeypatch.setattr(gate, "review_skill_write", _boom)
        monkeypatch.setattr(smt, "_review_gate_enabled_fallback", lambda: False)
        tok = prov.set_current_write_origin("background_review")
        try:
            # disabled ⇒ degrade to feature-absent ⇒ write_approval (off) ⇒ None (proceed)
            assert smt._apply_skill_write_gate("create", "s", content="x") is None
        finally:
            prov.reset_current_write_origin(tok)


# --------------------------------------------------------------------------- #
# Codex F7 — end-to-end through the real skill_manage() dispatch
# --------------------------------------------------------------------------- #
class TestSkillManageEndToEnd:
    def test_create_veto_blocks_and_does_not_write(self, clean_counters, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)
        monkeypatch.setattr(gate, "_deadline_seconds", lambda: 0.0)
        monkeypatch.setattr(gate, "_build_panel", _veto_panel)
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        tok = prov.set_current_write_origin("background_review")
        try:
            with _skill_dir(tmp_path):
                result = smt.skill_manage("create", "e2e-skill", content=_VALID_SKILL)
        finally:
            prov.reset_current_write_origin(tok)
        assert json.loads(result)["success"] is False              # blocked via real dispatch
        assert not (tmp_path / "e2e-skill" / "SKILL.md").exists()   # never written

    def test_create_gate_off_writes(self, clean_counters, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: False)
        tok = prov.set_current_write_origin("background_review")
        try:
            with _skill_dir(tmp_path):
                result = smt.skill_manage("create", "e2e-skill", content=_VALID_SKILL)
        finally:
            prov.reset_current_write_origin(tok)
        assert json.loads(result).get("success") is True
        assert (tmp_path / "e2e-skill" / "SKILL.md").exists()       # parity: normal creation
