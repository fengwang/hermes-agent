"""S4 unit tests: config defaults, rejection record, counters, and the gate's pure
calculations + orchestration. The seam-level tests live in test_skill_review_wiring.py;
the deadline/labeling/concurrency tests in test_skill_review_failclosed.py."""
from __future__ import annotations

import threading

import pytest

import tools.skill_provenance as prov
from tools.skill_review import gate, record
from tools.skill_review.panel import Panel
from tools.skill_review.reviewers.base import Reviewer
from tools.skill_review.schema import (
    SCHEMA_VERSION,
    Decision,
    DecisionRecord,
    Depth,
    Evidence,
    GraderType,
    Severity,
    Verdict,
    WriteTarget,
)


# --------------------------------------------------------------------------- #
# Task 1.1 — config defaults (config_and_killswitch)
# --------------------------------------------------------------------------- #
class TestConfigDefaults:
    def test_review_gate_defaults_present_and_off(self):
        from hermes_cli.config import DEFAULT_CONFIG, cfg_get

        assert cfg_get(DEFAULT_CONFIG, "skills", "review_gate", "enabled") is False
        assert cfg_get(DEFAULT_CONFIG, "skills", "review_gate", "deadline_seconds") == 30
        for rid in ("contract", "formal", "tool_workflow", "security", "safety"):
            assert cfg_get(DEFAULT_CONFIG, "skills", "review_gate", "reviewers", rid) is True

    def test_readers_resolve_defaults_with_no_user_config(self):
        from tools.skill_review.config import review_gate_enabled, reviewer_enabled

        assert review_gate_enabled() is False
        assert reviewer_enabled("security") is True

    def test_deadline_seconds_rejects_out_of_range_values(self, monkeypatch):
        # a bad deadline must fall back to the default, never become inf/nan/negative
        # (else join(timeout=) overflows). Guards INV-7 fail-closed robustness.
        import hermes_cli.config as cfg

        for bad in (float("inf"), float("nan"), -5, 0, 10**9):
            monkeypatch.setattr(cfg, "cfg_get", lambda *a, _v=bad, **k: _v)
            assert gate._deadline_seconds() == 30.0
        monkeypatch.setattr(cfg, "cfg_get", lambda *a, **k: 45)
        assert gate._deadline_seconds() == 45.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _target(name="s", action="create"):
    return WriteTarget(action=action, name=name, origin="background_review", file_path=None)


def _verdict(reviewer, locator, decision=Decision.VETO, detail="x"):
    return Verdict(
        reviewer=reviewer, decision=decision, severity=Severity.HIGH, confidence=1.0,
        evidence=(Evidence(locator, detail),), impacted_scope=("body",),
        rationale=f"{reviewer} {locator}", depth=Depth.FULL,
    )


def _record(*verdicts, decision=Decision.VETO, target=None):
    return DecisionRecord(SCHEMA_VERSION, target or _target(), decision, tuple(verdicts))


# --------------------------------------------------------------------------- #
# Task 1.2 — rejection key + counters (rejection_record, observability_counters)
# --------------------------------------------------------------------------- #
class TestRejectionKey:
    def test_key_is_deterministic_and_content_sensitive(self):
        from tools.skill_review import record

        t = _target(name="alpha")
        k1 = record.rejection_key(t, "content-A")
        k2 = record.rejection_key(t, "content-A")
        k3 = record.rejection_key(t, "content-B")
        k4 = record.rejection_key(_target(name="beta"), "content-A")
        assert k1 == k2
        assert k1 != k3
        assert k1 != k4
        assert len(k1) == 16


class TestCounters:
    def setup_method(self):
        from tools.skill_review import record
        record.reset()

    def test_bump_and_snapshot(self):
        from tools.skill_review import record

        record.bump("seen")
        record.bump("allowed")
        record.bump_reviewer("security")
        snap = record.snapshot()
        assert snap["seen"] == 1
        assert snap["allowed"] == 1
        assert snap["vetoed"]["security"] == 1

    def test_bump_is_thread_safe(self):
        from tools.skill_review import record

        def worker():
            for _ in range(1000):
                record.bump("seen")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert record.snapshot()["seen"] == 8000


# --------------------------------------------------------------------------- #
# Task 1.3 — write_rejection (best-effort, non-recursing, keyed)
# --------------------------------------------------------------------------- #
class TestWriteRejection:
    def test_writes_keyed_record(self, tmp_path, monkeypatch):
        from tools.skill_review import record

        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        rec = _record(_verdict("security", "shell"))
        key = record.write_rejection(rec, reason="veto", quality_signal=True, reviewable="c")

        path = tmp_path / "skill_review" / "rejections" / f"{key}.json"
        assert path.exists()
        import json
        payload = json.loads(path.read_text())
        assert payload["reason"] == "veto"
        assert payload["quality_signal"] is True
        assert payload["decision"] == "veto"
        assert "created_at" not in payload  # no wall-clock (deterministic record)

    def test_identical_rejections_dedupe_to_one_file(self, tmp_path, monkeypatch):
        from tools.skill_review import record

        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        rec = _record(_verdict("security", "shell"))
        record.write_rejection(rec, reason="veto", quality_signal=True, reviewable="c")
        record.write_rejection(rec, reason="veto", quality_signal=True, reviewable="c")
        files = list((tmp_path / "skill_review" / "rejections").glob("*.json"))
        assert len(files) == 1

    def test_disk_failure_is_swallowed(self, tmp_path, monkeypatch):
        from tools.skill_review import record

        def boom():
            raise OSError("nope")

        monkeypatch.setattr(record, "get_hermes_home", boom)
        rec = _record(_verdict("security", "shell"))
        # must not raise
        assert record.write_rejection(rec, reason="veto", quality_signal=True, reviewable="c") is None


# --------------------------------------------------------------------------- #
# Task 2.1 — classify_block precedence (fail_closed_policy)
# --------------------------------------------------------------------------- #
class TestClassifyBlock:
    def test_content_veto_is_quality(self):
        from tools.skill_review import gate

        rec = _record(_verdict("security", "shell"))
        assert gate.classify_block(rec) == ("veto", True)

    def test_reviewer_unavailable_is_nonquality(self):
        from tools.skill_review import gate

        rec = _record(_verdict("security", "reviewer-unavailable"))
        assert gate.classify_block(rec) == ("reviewer_unavailable", False)

    def test_deadline_is_nonquality_unavailable(self):
        from tools.skill_review import gate

        rec = _record(_verdict("panel", "reviewer-deadline"))
        assert gate.classify_block(rec) == ("reviewer_unavailable", False)

    def test_parse_failure_is_nonquality_error(self):
        from tools.skill_review import gate

        rec = _record(_verdict("safety", "llm-parse-failure"))
        assert gate.classify_block(rec) == ("reviewer_error", False)

    def test_genuine_veto_takes_precedence_over_unavailable(self):
        from tools.skill_review import gate

        rec = _record(
            _verdict("security", "shell"),
            _verdict("panel", "reviewer-deadline"),
        )
        assert gate.classify_block(rec) == ("veto", True)

    def test_deterministic_veto_with_deferred_note_is_still_quality(self):
        # formal/tool_workflow attach a 'deferred-dynamic-checks' note to every verdict;
        # that must NOT demote a genuine monotonicity veto to non-quality.
        from tools.skill_review import gate

        v = Verdict(
            reviewer="formal", decision=Decision.VETO, severity=Severity.HIGH, confidence=1.0,
            evidence=(Evidence("permission-monotonicity", "widens"),
                      Evidence("deferred-dynamic-checks", "static")),
            impacted_scope=("allowed-tools",), rationale="mono", depth=Depth.STATIC,
        )
        assert gate.classify_block(_record(v)) == ("veto", True)


# --------------------------------------------------------------------------- #
# Task 2.2 — reconstruct_post_image + builders (patch_post_image_review)
# --------------------------------------------------------------------------- #
class TestReconstructPostImage:
    # Semantics MUST mirror _patch_skill's engine (fuzzy_find_and_replace), NOT str.replace,
    # so the reviewed artifact is byte-identical to the written one (S4-R1).
    def test_unique_match_replaced(self):
        assert gate.reconstruct_post_image("a X b", "X", "Y", False) == "a Y b"

    def test_replace_all(self):
        assert gate.reconstruct_post_image("aXbXc", "X", "Y", True) == "aYbYc"

    def test_ambiguous_without_replace_all_is_none(self):
        # two matches + not replace_all ⇒ _patch_skill rejects (ambiguous) ⇒ fail-closed
        assert gate.reconstruct_post_image("aXbXc", "X", "Y", False) is None

    def test_no_match_returns_none(self):
        assert gate.reconstruct_post_image("abc", "X", "Y", False) is None

    def test_missing_current_or_old_returns_none(self):
        assert gate.reconstruct_post_image(None, "X", "Y", False) is None
        assert gate.reconstruct_post_image("abc", "", "Y", False) is None

    def test_fuzzy_only_match_applies_not_blocked(self):
        # exact substring FAILS (spaces vs tab) but the real engine matches fuzzily; the gate
        # must review the applied artifact, not false-block it (the High finding).
        current = "intro\n1. First do a thing\n2. Then\ttabbed line here\n"
        old = "2. Then    tabbed line here"  # spaces, not a tab
        assert old not in current  # exact would have returned None (old behavior)
        post = gate.reconstruct_post_image(current, old, "REPLACED", False)
        assert post is not None and "REPLACED" in post

    def test_reviewed_equals_written(self):
        # what the gate reviews == what _patch_skill would write (same engine)
        from tools.fuzzy_match import fuzzy_find_and_replace

        current, old, new = "line1\n  keep = 1\nline3\n", "keep = 1", "keep = 2"
        written, _c, _s, err = fuzzy_find_and_replace(current, old, new, False)
        assert not err
        assert gate.reconstruct_post_image(current, old, new, False) == written


class TestBuilders:
    def test_build_edit_write(self):
        from tools.skill_review import gate

        w = gate.build_edit_write("s", "post-content", "background_review")
        assert w.action == "edit" and w.content == "post-content" and w.name == "s"

    def test_build_patch_write_carries_delta(self):
        from tools.skill_review import gate

        w = gate.build_patch_write("s", "old", "new", None, "background_review")
        assert w.action == "patch" and w.old_string == "old" and w.new_string == "new"


# --------------------------------------------------------------------------- #
# Task 2.3 — blocked_message (review_gate_wiring)
# --------------------------------------------------------------------------- #
class TestBlockedMessage:
    def test_veto_message_carries_rationale(self):
        from tools.skill_review import gate

        rec = _record(_verdict("security", "shell", detail="d"))
        msg = gate.blocked_message(rec, "veto")
        assert "blocked by review gate" in msg.lower()
        assert "security" in msg  # from the verdict rationale

    def test_unavailable_message(self):
        from tools.skill_review import gate

        rec = _record(_verdict("panel", "reviewer-deadline"))
        assert "unavailable" in gate.blocked_message(rec, "reviewer_unavailable").lower()


# --------------------------------------------------------------------------- #
# Task 3 — review_skill_write orchestration (review_gate_wiring, patch_post_image_review)
# --------------------------------------------------------------------------- #
class _FixedReviewer(Reviewer):
    id = "fake"
    grader_type = GraderType.DETERMINISTIC

    def __init__(self, verdict):
        self._verdict = verdict

    def review(self, write):
        type(self).last_write = write
        return self._verdict


def _pass_verdict():
    return Verdict("fake", Decision.PASS, Severity.INFO, 1.0, rationale="ok")


def _panel_with(verdict):
    return Panel([_FixedReviewer(verdict)])


@pytest.fixture
def bg(monkeypatch):
    """Gate enabled + background origin + inline (no-deadline) panel + fresh counters."""
    monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)
    monkeypatch.setattr(gate, "_deadline_seconds", lambda: 0.0)  # run inline (no worker thread)
    record.reset()
    _FixedReviewer.last_write = None
    tok = prov.set_current_write_origin("background_review")
    try:
        yield
    finally:
        prov.reset_current_write_origin(tok)


class TestOrchestration:
    def test_disabled_is_noop_allow(self, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: False)

        def _boom():
            raise AssertionError("panel must not be built when disabled")

        monkeypatch.setattr(gate, "_build_panel", _boom)
        tok = prov.set_current_write_origin("background_review")
        try:
            assert gate.review_skill_write("create", "s", content="x").allow is True
        finally:
            prov.reset_current_write_origin(tok)

    def test_foreground_is_allow(self, monkeypatch):
        monkeypatch.setattr(gate, "review_gate_enabled", lambda: True)

        def _boom():
            raise AssertionError("panel must not be built for foreground writes")

        monkeypatch.setattr(gate, "_build_panel", _boom)
        # default origin is foreground
        assert gate.review_skill_write("create", "s", content="x").allow is True

    def test_delete_is_not_gated(self, bg, monkeypatch):
        def _boom():
            raise AssertionError("delete must not be reviewed")

        monkeypatch.setattr(gate, "_build_panel", _boom)
        assert gate.review_skill_write("delete", "s").allow is True

    def test_clean_create_allows_and_counts(self, bg, monkeypatch):
        monkeypatch.setattr(gate, "_build_panel", lambda: _panel_with(_pass_verdict()))
        decision = gate.review_skill_write("create", "s", content="clean")
        assert decision.allow is True
        snap = record.snapshot()
        assert snap["seen"] == 1 and snap["allowed"] == 1 and snap["blocked_veto"] == 0

    def test_veto_create_blocks_records_and_counts(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_build_panel",
                            lambda: _panel_with(_verdict("security", "shell")))
        decision = gate.review_skill_write("create", "bad", content="danger")
        assert decision.blocked is True
        assert "review gate" in decision.message.lower()
        snap = record.snapshot()
        assert snap["seen"] == 1 and snap["blocked_veto"] == 1 and snap["vetoed"]["security"] == 1
        # a keyed rejection record was written
        assert list((tmp_path / "skill_review" / "rejections").glob("*.json"))

    def test_patch_reviews_post_image(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_read_current", lambda name, fp: "before OLD after")
        monkeypatch.setattr(gate, "_build_panel",
                            lambda: _panel_with(_verdict("security", "shell")))
        decision = gate.review_skill_write("patch", "s", old_string="OLD", new_string="NEW")
        assert decision.blocked is True
        # the panel saw the reconstructed post-image (an edit-shaped write), not the delta
        assert _FixedReviewer.last_write.action == "edit"
        assert _FixedReviewer.last_write.content == "before NEW after"

    def test_patch_widening_allowed_tools_is_vetoed_by_monotonicity(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        current = "---\nname: s\nallowed-tools: [read_file]\n---\nbody"
        monkeypatch.setattr(gate, "_read_current", lambda name, fp: current)
        # the post-image panel PASSES; only the supplementary formal-delta check should veto
        monkeypatch.setattr(gate, "_build_panel", lambda: _panel_with(_pass_verdict()))
        decision = gate.review_skill_write(
            "patch", "s",
            old_string="allowed-tools: [read_file]",
            new_string="allowed-tools: [read_file, write_file]",
        )
        assert decision.blocked is True
        assert "monotonic" in decision.message.lower() or "allowed-tools" in decision.message.lower()

    def test_non_applying_patch_fails_closed(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_read_current", lambda name, fp: "no match here")

        def _boom():
            raise AssertionError("panel must not run when the post-image can't be built")

        monkeypatch.setattr(gate, "_build_panel", _boom)
        decision = gate.review_skill_write("patch", "s", old_string="ABSENT", new_string="X")
        assert decision.blocked is True
        assert "fail-closed" in decision.message.lower()
        assert record.snapshot()["blocked_unavailable"] == 1

    def test_empty_panel_fails_closed(self, bg, tmp_path, monkeypatch):
        # gate enabled but every reviewer toggled off ⇒ block, do NOT silently allow
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_build_panel", lambda: Panel([]))
        decision = gate.review_skill_write("create", "s", content="x")
        assert decision.blocked is True
        snap = record.snapshot()
        assert snap["blocked_unavailable"] == 1 and snap["blocked_veto"] == 0

    def test_edit_reviews_content(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_build_panel",
                            lambda: _panel_with(_verdict("security", "shell")))
        decision = gate.review_skill_write("edit", "s", content="danger-edit")
        assert decision.blocked is True
        assert _FixedReviewer.last_write.action == "edit"
        assert _FixedReviewer.last_write.content == "danger-edit"

    def test_write_file_reviews_file_content(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_build_panel",
                            lambda: _panel_with(_verdict("security", "shell")))
        decision = gate.review_skill_write("write_file", "s", file_path="ref.md",
                                           file_content="danger-file")
        assert decision.blocked is True
        assert _FixedReviewer.last_write.action == "write_file"
        assert _FixedReviewer.last_write.file_content == "danger-file"

    def test_patch_replace_all_reviews_full_post_image(self, bg, tmp_path, monkeypatch):
        monkeypatch.setattr(record, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gate, "_read_current", lambda name, fp: "aXbXc")
        monkeypatch.setattr(gate, "_build_panel", lambda: _panel_with(_pass_verdict()))
        decision = gate.review_skill_write("patch", "s", old_string="X", new_string="Y",
                                           replace_all=True)
        assert decision.allow is True
        assert _FixedReviewer.last_write.content == "aYbYc"  # ALL occurrences replaced


class TestReadCurrentTraversal:
    def test_rejects_path_traversal(self, tmp_path, monkeypatch):
        import tools.skill_manager_tool as smt

        monkeypatch.setattr(smt, "_find_skill", lambda name: {"path": tmp_path})
        (tmp_path / "SKILL.md").write_text("hello", encoding="utf-8")
        # a legitimate default read works
        assert gate._read_current("s", None) == "hello"
        # a traversal file_path is refused (same guard as _patch_skill), never read
        assert gate._read_current("s", "../../../../etc/passwd") is None
