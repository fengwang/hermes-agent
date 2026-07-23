"""Formal-invariants reviewer tests, driven by the labelled eval seed cases.

Covers SEED-FI1 (permission monotonicity, patch-delta), the net-new identity-consistency
check, SEED-OK2 (curator consolidation allowed), determinism (M3), read-only behaviour
(INV-4), the depth=static + deferred-checks note (exit crit. 3), and guard alignment
(INV-6 / exit crit. 4 — the EXISTING owned-skill/pinned guards still fire independently
while the reviewer defers). Rationale: docs/skill_review_static_limitations.md (fork-local).
"""
import os
from pathlib import Path

import pytest

from tools.skill_review.reviewers.base import SkillWrite
from tools.skill_review.reviewers.formal_invariants import FormalInvariantsReviewer
from tools.skill_review.schema import Decision, Depth, GraderType


def _skill_md(name="debugging-workflows",
              description="A reusable, class-level debugging workflow skill.",
              body="## Overview\n\nA reusable procedure using existing tools.\n",
              allowed_tools=None) -> str:
    lines = [f"name: {name}", f"description: {description}"]
    if allowed_tools is not None:
        lines.append(f"allowed-tools: [{', '.join(allowed_tools)}]")
    return "---\n" + "\n".join(lines) + "\n---\n\n" + body


def _review(**kw):
    base = dict(action="create", name="debugging-workflows", origin="background_review")
    base.update(kw)
    return FormalInvariantsReviewer().review(SkillWrite(**base))


class TestReviewerIdentity:
    def test_advertises_id_and_grader_type(self):
        r = FormalInvariantsReviewer()
        assert r.id == "formal"
        assert r.grader_type is GraderType.DETERMINISTIC

    def test_every_verdict_is_static_depth(self):
        assert _review(content=_skill_md()).depth is Depth.STATIC
        veto = _review(action="patch", name="x",
                       old_string="allowed-tools: [Read]",
                       new_string="allowed-tools: [Read, Bash]")
        assert veto.depth is Depth.STATIC


class TestDeferredNote:
    def test_pass_verdict_carries_deferred_checks_note(self):
        v = _review(content=_skill_md())
        assert v.decision is Decision.PASS
        assert any(e.locator == "deferred-dynamic-checks" for e in v.evidence)

    def test_veto_verdict_also_carries_deferred_checks_note(self):
        v = _review(action="patch", name="x",
                    old_string="allowed-tools: [Read]",
                    new_string="allowed-tools: [Read, Bash]")
        assert v.decision is Decision.VETO
        assert any(e.locator == "deferred-dynamic-checks" for e in v.evidence)

    def test_note_references_limitations_doc(self):
        note = next(e for e in _review(content=_skill_md()).evidence
                    if e.locator == "deferred-dynamic-checks")
        assert "skill_review_static_limitations" in note.detail


class TestPermissionMonotonicity:
    def test_fi1_patch_widens_tool_scope_vetoes(self):
        v = _review(action="patch", name="debugging-workflows",
                    old_string="allowed-tools: [Read]",
                    new_string="allowed-tools: [Read, Bash]")
        assert v.decision is Decision.VETO
        assert any(e.locator == "permission-monotonicity" for e in v.evidence)

    def test_widening_with_tiny_diff_still_vetoes(self):
        # Adversarial: widen scope while shrinking the diff — check is on the set, not size.
        v = _review(action="patch", name="x",
                    old_string="allowed-tools: [Read]",
                    new_string="allowed-tools: [Read, Bash]")
        assert v.decision is Decision.VETO

    def test_wildcard_widening_vetoes(self):
        v = _review(action="patch", name="x",
                    old_string="allowed-tools: [Read]",
                    new_string="allowed-tools: [*]")
        assert v.decision is Decision.VETO

    def test_narrowing_patch_passes(self):
        v = _review(action="patch", name="x",
                    old_string="allowed-tools: [Read, Bash]",
                    new_string="allowed-tools: [Read]")
        assert v.decision is Decision.PASS

    def test_reordering_same_set_passes(self):
        v = _review(action="patch", name="x",
                    old_string="allowed-tools: [Read, Bash]",
                    new_string="allowed-tools: [Bash, Read]")
        assert v.decision is Decision.PASS

    def test_no_manifest_in_delta_does_not_veto(self):
        # No allowed-tools in the delta ⇒ no baseline ⇒ no monotonicity veto (documented limit).
        v = _review(action="patch", name="x", old_string="teh", new_string="the")
        assert v.decision is Decision.PASS

    def test_only_new_declares_manifest_does_not_veto(self):
        # Old side has no baseline in the delta ⇒ cannot prove widening ⇒ no veto (M2-safe).
        v = _review(action="patch", name="x",
                    old_string="some prose",
                    new_string="allowed-tools: [Read, Bash]")
        assert v.decision is Decision.PASS

    def test_create_broad_scope_not_monotonicity_vetoed(self):
        # create/edit have no prior version ⇒ no monotonicity veto (I3 documented limitation).
        v = _review(action="create", name="debugging-workflows",
                    content=_skill_md(allowed_tools=["Bash", "Read", "Write"]))
        assert v.decision is Decision.PASS


class TestIdentityConsistency:
    def test_frontmatter_name_mismatch_vetoes(self):
        v = _review(action="create", name="debugging-workflows",
                    content=_skill_md(name="something-else"))
        assert v.decision is Decision.VETO
        assert any(e.locator == "identity" for e in v.evidence)

    def test_matching_name_passes(self):
        v = _review(action="create", name="debugging-workflows",
                    content=_skill_md(name="debugging-workflows"))
        assert v.decision is Decision.PASS

    def test_edit_name_mismatch_vetoes(self):
        v = _review(action="edit", name="debugging-workflows",
                    content=_skill_md(name="other-name"))
        assert v.decision is Decision.VETO

    def test_missing_frontmatter_name_is_not_an_identity_veto(self):
        # Missing name is contract's job (existence); formal must not double-penalize (E1).
        content = "---\ndescription: no name here\n---\n\n## Body\n"
        v = _review(action="create", name="debugging-workflows", content=content)
        assert not any(e.locator == "identity" for e in v.evidence)

    def test_case_only_mismatch_passes(self):
        # M2: a case-only frontmatter/write-name difference is a benign authoring slip (the write
        # name is authoritative and lowercased by VALID_NAME_RE) — not an identity mismatch.
        v = _review(action="create", name="debugging-workflows",
                    content=_skill_md(name="Debugging-Workflows"))
        assert v.decision is Decision.PASS


class TestConsolidationAllowed:
    """SEED-OK2 (R14): a legitimate curator consolidation must not be falsely vetoed."""

    def test_ok2_umbrella_create_passes(self):
        v = _review(action="create", name="debugging-umbrella",
                    content=_skill_md(name="debugging-umbrella",
                                      allowed_tools=["Bash", "Read", "skill_view"]))
        assert v.decision is Decision.PASS

    def test_ok2_absorbed_delete_passes(self):
        v = _review(action="delete", name="narrow-skill")
        assert v.decision is Decision.PASS


class TestDeterminism:
    @pytest.mark.parametrize("kw", [
        dict(content=_skill_md()),
        dict(action="patch", name="x", old_string="allowed-tools: [Read]",
             new_string="allowed-tools: [Read, Bash]"),
        dict(action="create", name="debugging-workflows", content=_skill_md(name="mismatch")),
    ])
    def test_verdict_byte_identical_across_runs(self, kw):
        assert _review(**kw).to_dict() == _review(**kw).to_dict()


class TestReadOnly:
    def test_review_writes_nothing_to_disk(self):
        skills = Path(os.environ["HERMES_HOME"]) / "skills"
        before = sorted(p.name for p in skills.iterdir())
        _review(content=_skill_md())
        assert sorted(p.name for p in skills.iterdir()) == before

    def test_review_opens_no_files_and_reads_no_config(self, monkeypatch):
        import builtins

        import hermes_cli.config as hc

        opened, loaded = [], []
        real_open = builtins.open

        def _spy_open(*args, **kwargs):
            opened.append(args[0] if args else None)
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", _spy_open)
        monkeypatch.setattr(hc, "load_config", lambda: loaded.append(1) or {})

        _review(content=_skill_md())
        _review(action="patch", name="x", old_string="allowed-tools: [Read]",
                new_string="allowed-tools: [Read, Bash]")

        assert opened == []
        assert loaded == []


class TestGuardsIntact:
    """INV-6 / exit crit. 4 (D5): the EXISTING guards still fire independently, and the formal
    reviewer DEFERS these state-dependent invariants (FI2 owned-skill, FI3 pinned) rather than
    duplicating or weakening them. The reviewer PASSes the same write; the real guard refuses."""

    @staticmethod
    def _guard(name, action="patch"):
        from tools.skill_manager_tool import _background_review_write_guard
        return _background_review_write_guard(name, Path("/nonexistent/skills") / name, action)

    @staticmethod
    def _defer_write(name):
        return SkillWrite(action="patch", name=name, old_string="Do the thing.",
                          new_string="Do the new thing.", origin="background_review")

    def test_fi3_pinned_guard_fires_and_reviewer_defers(self, monkeypatch):
        from tools.skill_provenance import (
            BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin,
        )
        import tools.skill_usage as su
        monkeypatch.setattr(su, "get_record", lambda n: {"pinned": True})
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            refusal = self._guard("pinned-skill")
        finally:
            reset_current_write_origin(token)
        assert refusal is not None and refusal["success"] is False
        assert "pinned" in refusal["error"].lower()
        # The formal reviewer defers pinned (it does not read the pinned flag): benign patch PASSes.
        assert FormalInvariantsReviewer().review(self._defer_write("pinned-skill")).decision is Decision.PASS

    def test_fi2_owned_skill_guard_fires_and_reviewer_defers(self, monkeypatch):
        from tools.skill_provenance import (
            BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin,
        )
        import agent.skill_utils as skill_utils
        import tools.skill_usage as su
        monkeypatch.setattr(su, "get_record", lambda n: {"pinned": False})
        monkeypatch.setattr(su, "is_protected_builtin", lambda n: False)
        monkeypatch.setattr(su, "is_hub_installed", lambda n: False)
        monkeypatch.setattr(su, "is_bundled", lambda n: False)
        monkeypatch.setattr(su, "load_usage", lambda: {"user-skill": {"created_by": "user"}})
        monkeypatch.setattr(skill_utils, "is_external_skill_path", lambda p: False)
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            refusal = self._guard("user-skill")
        finally:
            reset_current_write_origin(token)
        assert refusal is not None and refusal["success"] is False
        assert ("agent-created" in refusal["error"].lower()
                or "manually authored" in refusal["error"].lower())
        # The formal reviewer defers owned-skill (it does not read created_by): benign patch PASSes.
        assert FormalInvariantsReviewer().review(self._defer_write("user-skill")).decision is Decision.PASS

    def test_read_before_write_guard_fires_and_reviewer_defers(self):
        # codex-F5 / contract acceptance: the read-before-write guard is named in the contract but
        # was untested. Drive the REAL guard (background origin + an unread target) and assert the
        # reviewer defers (it does not implement read-before-write).
        from tools.skill_manager_tool import _background_review_read_before_write_guard
        from tools.skill_provenance import (
            BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin,
        )
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            refusal = _background_review_read_before_write_guard(
                "unread-skill", Path("/nonexistent/skills/unread-skill/SKILL.md"), "patch", "SKILL.md")
        finally:
            reset_current_write_origin(token)
        assert refusal is not None and refusal["success"] is False
        assert refusal.get("_read_before_write_required") is True
        assert FormalInvariantsReviewer().review(self._defer_write("unread-skill")).decision is Decision.PASS

    def test_fi4_create_collision_guard_fires_and_reviewer_defers(self):
        # codex-F5: FI4 name-collision is enforced by _create_skill (_find_skill), not the reviewer.
        from tools.skill_manager_tool import _create_skill
        content = _skill_md(name="collide-skill")
        assert _create_skill("collide-skill", content).get("success") is True
        collision = _create_skill("collide-skill", content)
        assert collision.get("success") is False
        assert "already exists" in collision["error"].lower()
        # The formal reviewer does not check collision (needs the filesystem) — it defers.
        write = SkillWrite(action="create", name="collide-skill", content=content,
                           origin="background_review")
        assert FormalInvariantsReviewer().review(write).decision is Decision.PASS


class TestBlockStyleMonotonicity:
    """codex-F2: block-style YAML allowed-tools must be parsed too, else widening evades the veto."""

    def test_block_style_widening_vetoes(self):
        old = "allowed-tools:\n  - Read\n"
        new = "allowed-tools:\n  - Read\n  - Bash\n"
        assert _review(action="patch", name="x", old_string=old, new_string=new).decision is Decision.VETO

    def test_inline_to_block_reformat_same_set_passes(self):
        # Pure reformat (no widening) must NOT veto (M2).
        old = "allowed-tools: [Read, Bash]"
        new = "allowed-tools:\n  - Read\n  - Bash\n"
        assert _review(action="patch", name="x", old_string=old, new_string=new).decision is Decision.PASS

    def test_block_to_inline_reformat_same_set_passes(self):
        old = "allowed-tools:\n  - Read\n  - Bash\n"
        new = "allowed-tools: [Bash, Read]"
        assert _review(action="patch", name="x", old_string=old, new_string=new).decision is Decision.PASS


class TestEvidenceBounds:
    """codex-F6: author-controlled strings must be capped before entering verdict evidence."""

    def test_added_tool_name_is_capped(self):
        huge = "x" * 5000
        v = _review(action="patch", name="s",
                    old_string="allowed-tools: [Read]",
                    new_string=f"allowed-tools: [Read, {huge}]")
        assert v.decision is Decision.VETO
        detail = " ".join(e.detail for e in v.evidence)
        assert huge not in detail

    def test_frontmatter_name_is_capped(self):
        huge = "y" * 5000
        v = _review(action="create", name="s", content=_skill_md(name=huge))
        assert v.decision is Decision.VETO
        detail = " ".join(e.detail for e in v.evidence)
        assert huge not in detail
