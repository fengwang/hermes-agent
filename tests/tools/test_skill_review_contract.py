"""Contract/schema reviewer tests, driven by the labelled eval seed cases.

Covers SEED-C1..C6 (must veto), SEED-OK1/OK5 (must allow), the empty-body case,
determinism (M3), and read-only behaviour (INV-4). See docs/eval_seed_cases.md and
docs/session_1/design.md §6-§7 (Requirement: Contract/schema reviewer).
"""
import os
from pathlib import Path

import pytest

from tools.skill_review.reviewers.base import SkillWrite
from tools.skill_review.reviewers.contract import (
    ContractReviewer,
    MAX_SKILL_CONTENT_CHARS,
    PATCH_MAX_NEW_CHARS,
)
from tools.skill_review.schema import Decision, GraderType, Severity


def _skill_md(name="debugging-workflows",
              description="A reusable, class-level debugging workflow skill.",
              body="## Overview\n\nA reusable procedure using existing tools.\n",
              allowed_tools=False) -> str:
    lines = [f"name: {name}", f"description: {description}"]
    if allowed_tools:
        lines.append("allowed-tools: [Bash, Read]")
    return "---\n" + "\n".join(lines) + "\n---\n\n" + body


def _review(**kw):
    base = dict(action="create", name="debugging-workflows", origin="background_review")
    base.update(kw)
    return ContractReviewer().review(SkillWrite(**base))


# --- SEED payloads (should-VETO) -------------------------------------------------

SEED_C1 = dict(action="create", name="my-skill",
               content="---\nname: my-skill\n---\n\nSome body.\n")            # no description
SEED_C2 = dict(action="create", name="big-skill",
               content=_skill_md(body="x" * (MAX_SKILL_CONTENT_CHARS + 1)))    # oversized
SEED_C3_SPACE = dict(action="create", name="my skill", content=_skill_md(name="my skill"))
SEED_C3_TRAVERSAL = dict(action="create", name="../evil", content=_skill_md(name="evil"))
SEED_C4 = dict(action="create", name="fix-pr-4821-timeout",
               content=_skill_md(name="fix-pr-4821-timeout"))                  # over-narrow
SEED_C5 = dict(action="patch", name="debugging-workflows",
               old_string="a typo", new_string="y" * (PATCH_MAX_NEW_CHARS + 1))  # rewrite-as-patch
SEED_C6 = dict(action="write_file", name="debugging-workflows",
               file_path="config/settings.md", file_content="data")           # disallowed subdir
SEED_EMPTY_BODY = dict(action="create", name="empty-skill",
                       content="---\nname: empty-skill\ndescription: has no body\n---\n")

VETO_SEEDS = {
    "C1": SEED_C1, "C2": SEED_C2, "C3_space": SEED_C3_SPACE,
    "C3_traversal": SEED_C3_TRAVERSAL, "C4": SEED_C4, "C5": SEED_C5,
    "C6": SEED_C6, "empty_body": SEED_EMPTY_BODY,
}


class TestReviewerIdentity:
    def test_advertises_id_and_grader_type(self):
        r = ContractReviewer()
        assert r.id == "contract"
        assert r.grader_type is GraderType.DETERMINISTIC


class TestVetoSeeds:
    @pytest.mark.parametrize("seed_id", list(VETO_SEEDS))
    def test_seed_vetoes(self, seed_id):
        verdict = _review(**VETO_SEEDS[seed_id])
        assert verdict.decision is Decision.VETO, seed_id
        assert verdict.reviewer == "contract"
        assert verdict.confidence == 1.0
        assert len(verdict.evidence) >= 1

    def test_c1_locates_frontmatter(self):
        v = _review(**SEED_C1)
        assert any("frontmatter" in e.locator for e in v.evidence)

    def test_c3_locates_name(self):
        v = _review(**SEED_C3_SPACE)
        assert any(e.locator == "name" for e in v.evidence)

    def test_c4_is_a_veto_not_a_warning(self):
        # Adversarial case: an over-narrow one-shot name must veto, not merely warn.
        v = _review(**SEED_C4)
        assert v.decision is Decision.VETO
        assert any(e.locator == "name" for e in v.evidence)

    def test_c5_locates_patch(self):
        v = _review(**SEED_C5)
        assert any(e.locator == "patch" for e in v.evidence)

    def test_empty_body_locates_body(self):
        v = _review(**SEED_EMPTY_BODY)
        assert v.decision is Decision.VETO
        assert any(e.locator == "body" for e in v.evidence)


class TestAllowSeeds:
    def test_ok1_well_formed_umbrella_skill_passes(self):
        v = _review(action="create", name="debugging-workflows", content=_skill_md())
        assert v.decision is Decision.PASS
        assert v.severity is Severity.INFO

    def test_ok1_missing_manifest_is_informational_only(self):
        v = _review(action="create", name="debugging-workflows", content=_skill_md(allowed_tools=False))
        assert v.decision is Decision.PASS
        assert any("allowed-tools" in e.locator for e in v.evidence)

    def test_ok1_with_manifest_has_no_notes(self):
        v = _review(action="create", name="debugging-workflows", content=_skill_md(allowed_tools=True))
        assert v.decision is Decision.PASS
        assert v.evidence == ()

    def test_ok5_small_typo_patch_passes(self):
        v = _review(action="patch", name="debugging-workflows",
                    old_string="teh", new_string="the")
        assert v.decision is Decision.PASS


class TestDeterminism:
    @pytest.mark.parametrize("seed_id", list(VETO_SEEDS))
    def test_veto_seed_is_byte_identical_across_runs(self, seed_id):
        # M3: deterministic reviewers produce identical verdicts across repeated runs.
        first = _review(**VETO_SEEDS[seed_id]).to_dict()
        second = _review(**VETO_SEEDS[seed_id]).to_dict()
        assert first == second


class TestValidatorAlignment:
    """INV-6: the reviewer must mirror the live validators faithfully, not weaken them."""

    def test_write_file_char_cap_vetoes(self):
        # Live _write_file enforces BOTH the 1 MiB byte cap AND the 100k-char cap.
        big_ascii = "a" * (MAX_SKILL_CONTENT_CHARS + 1)  # < 1 MiB bytes, > 100k chars
        v = _review(action="write_file", name="debugging-workflows",
                    file_path="references/big.md", file_content=big_ascii)
        assert v.decision is Decision.VETO
        assert any(e.locator == "file_content" for e in v.evidence)

    def test_patch_on_supporting_file_rejects_traversal(self):
        # Live _patch_skill validates file_path when patching a supporting file.
        v = _review(action="patch", name="debugging-workflows",
                    file_path="../../../etc/passwd", old_string="x", new_string="y")
        assert v.decision is Decision.VETO
        assert any(e.locator == "file_path" for e in v.evidence)

    def test_patch_on_disallowed_subdir_vetoes(self):
        v = _review(action="patch", name="debugging-workflows",
                    file_path="config/x.md", old_string="x", new_string="y")
        assert v.decision is Decision.VETO

    def test_patch_on_skill_md_is_unaffected(self):
        # A normal patch (no file_path → SKILL.md) is still allowed.
        v = _review(action="patch", name="debugging-workflows",
                    old_string="teh", new_string="the")
        assert v.decision is Decision.PASS


class TestReadOnly:
    def test_review_writes_nothing_to_disk(self):
        # INV-4: reviewers cause no side effects.
        skills = Path(os.environ["HERMES_HOME"]) / "skills"
        before = sorted(p.name for p in skills.iterdir())
        _review(action="create", name="debugging-workflows", content=_skill_md())
        after = sorted(p.name for p in skills.iterdir())
        assert before == after

    def test_review_opens_no_files_and_reads_no_config(self, monkeypatch):
        # INV-4 (stronger): the reviewer must not open files or read live config.
        import builtins

        import hermes_cli.config as hc

        opened, loaded = [], []
        real_open = builtins.open

        def _spy_open(*args, **kwargs):
            opened.append(args[0] if args else None)
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", _spy_open)
        monkeypatch.setattr(hc, "load_config", lambda: loaded.append(1) or {})

        _review(action="create", name="debugging-workflows", content=_skill_md())
        _review(action="write_file", name="debugging-workflows",
                file_path="references/x.md", file_content="data")

        assert opened == []
        assert loaded == []
