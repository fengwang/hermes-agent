"""The live skill-review gate: the panel↔``GateDecision`` adapter + fail-closed policy (S4).

This module is the *only* thing that turns the (inert) reviewer panel into an enforced
quality gate. It is consulted by ``skill_manager_tool._apply_skill_write_gate`` for
**agent-origin** writes when ``skills.review_gate.enabled`` is on; foreground/disabled
writes never reach the panel (INV-1/INV-2). It is **independent** of ``skills.write_approval``
— it reuses only the ``GateDecision`` value type and the origin helper from that module, never
its flag or pending queue (INV-3).

Functional decomposition (Grokking Simplicity ACD):
  * **Calculations** (pure, deterministic, no I/O): ``classify_block``, ``blocked_message``,
    ``reconstruct_post_image``, ``_targets_main_skill``, and the ``build_*_write`` assemblers.
  * **Actions** (I/O, isolated): ``run_panel_bounded`` (worker thread + copied context),
    ``_read_current`` (validated on-disk read), ``review_skill_write`` (the orchestrator).

The gate is a pure decision + audit: it never writes the skill and never changes lifecycle
state (INV-5). Its only side effects are the bounded panel invocation, a best-effort rejection
record, and counters.
"""
from __future__ import annotations

import contextvars
import logging
import re
import threading

from tools import write_approval as _wa  # GateDecision value type + origin helper ONLY (INV-3)
from tools.skill_review import record as _rec
from tools.skill_review.config import review_gate_enabled, reviewer_enabled
from tools.skill_review.panel import Panel, PanelMode
from tools.skill_review.reviewers.base import SkillWrite
from tools.skill_review.schema import (
    SCHEMA_VERSION,
    Decision,
    DecisionRecord,
    Depth,
    Evidence,
    Severity,
    Verdict,
    WriteTarget,
)

logger = logging.getLogger(__name__)

_GATED_ACTIONS = ("create", "edit", "patch", "write_file")

# Locators the frozen reviewers (llm.py) / this gate use to mark a block that is NOT a
# statement about skill quality (INV-8): an infra failure or a reviewer malfunction.
_INFRA_LOCATORS = frozenset({"reviewer-unavailable", "reviewer-deadline"})
_ERROR_LOCATORS = frozenset({"llm-parse-failure", "reviewer-error"})

# Internal subreason → the contract's persisted reason label (project_contract §3-§4, INV-8).
_CONTRACT_REASON = {
    "veto": "blocked-by-veto",
    "reviewer_unavailable": "blocked-by-reviewer-unavailable",
    "reviewer_error": "blocked-by-reviewer-unavailable",
}

_PREFIX = {
    "veto": "Skill blocked by review gate",
    "reviewer_unavailable": "Skill review unavailable (fail-closed)",
    "reviewer_error": "Skill review error (fail-closed)",
}

# Agent-facing message / persisted-detail bounds (R3/VC-3: the reviewed skill is untrusted, and
# reviewer rationale can echo attacker-influenced text back into the main agent / audit log).
_MSG_RATIONALE_CAP = 200
_MSG_TOTAL_CAP = 600
_WS_RE = re.compile(r"\s+")

# Cap on concurrent (possibly-abandoned) review workers so a timeout storm cannot accumulate
# unbounded threads/provider calls (F3, R6). Generous vs the fork+curator worst case.
_MAX_CONCURRENT_REVIEWS = 8
_review_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_REVIEWS)


# --------------------------------------------------------------------------- #
# Calculations (pure)
# --------------------------------------------------------------------------- #
def _locators(verdict: Verdict) -> set[str]:
    return {e.locator for e in verdict.evidence}


def classify_block(record: DecisionRecord) -> tuple[str, bool]:
    """Return ``(subreason, quality_signal)`` for a blocked record.

    Precedence (most-informative wins): a genuine content veto (a blocking verdict carrying
    ANY locator that is not an infra/error marker) ⇒ ``("veto", True)`` — so a real veto that
    also happens to carry an infra locator is still a quality signal; else a
    reviewer-unavailable / deadline block ⇒ ``("reviewer_unavailable", False)``; else
    (parse-failure / gate-error / unknown) ⇒ ``("reviewer_error", False)``. Only a genuine
    content veto is a quality signal safe to feed back (INV-8). The contract label is
    ``_CONTRACT_REASON[subreason]``.
    """
    blocking = record.blocking_verdicts()
    for v in blocking:
        if _locators(v) - (_INFRA_LOCATORS | _ERROR_LOCATORS):
            return ("veto", True)
    for v in blocking:
        if _locators(v) & _INFRA_LOCATORS:
            return ("reviewer_unavailable", False)
    return ("reviewer_error", False)


def blocked_message(record: DecisionRecord, subreason: str) -> str:
    """The agent-facing ``tool_error`` message: prefix + bounded, whitespace-collapsed,
    secret-redacted blocking rationales (so untrusted reviewer text can't flood/inject the
    main agent, F8)."""
    parts: list[str] = []
    for v in record.blocking_verdicts():
        if v.rationale:
            clean = _rec.redact_secrets(_WS_RE.sub(" ", v.rationale).strip())
            if clean:
                parts.append(clean[:_MSG_RATIONALE_CAP])
    body = "; ".join(parts)[:_MSG_TOTAL_CAP]
    prefix = _PREFIX.get(subreason, _PREFIX["reviewer_error"])
    return f"{prefix}: {body}" if body else prefix


def reconstruct_post_image(current: str | None, old: str | None, new: str | None,
                           replace_all: bool) -> str | None:
    """The post-patch content, using the SAME engine ``_patch_skill`` writes with, so the
    reviewed artifact is byte-identical to what would go active (S4-R1).

    ``_patch_skill`` applies the delta via ``fuzzy_find_and_replace`` (9-strategy fuzzy
    matching + ``new_string`` re-indent/unescape), NOT exact ``str.replace``. Mirroring only
    exact replacement would both (a) false-block patches that apply fuzzily and (b) review a
    different artifact than the one written. ``None`` when the delta would not apply
    (``match_error`` — e.g. no match, or an ambiguous match without ``replace_all``) so the
    caller fails closed. ``fuzzy_find_and_replace`` is pure (no I/O), so this stays a
    Calculation.
    """
    if current is None or not old:
        return None
    from tools.fuzzy_match import fuzzy_find_and_replace

    new_content, _count, _strategy, match_error = fuzzy_find_and_replace(
        current, old, new or "", replace_all
    )
    return None if match_error else new_content


def _targets_main_skill(file_path: str | None) -> bool:
    """True when a write targets the main ``SKILL.md`` (vs a supporting file). Mirrors
    skill_manager's rule that a path whose basename is ``SKILL.md`` is the canonical file."""
    if not file_path:
        return True
    from pathlib import PurePosixPath

    return PurePosixPath(str(file_path)).name == "SKILL.md"


def build_natural_write(action: str, name: str, content: str | None, file_path: str | None,
                        file_content: str | None, origin: str) -> SkillWrite:
    return SkillWrite(action=action, name=name, content=content, file_path=file_path,
                      file_content=file_content, origin=origin)


def build_edit_write(name: str, post: str, origin: str) -> SkillWrite:
    """A synthetic ``edit`` carrying the reconstructed main-skill artifact (frontmatter shape)."""
    return SkillWrite(action="edit", name=name, content=post, origin=origin)


def build_write_file_write(name: str, file_path: str | None, post: str, origin: str) -> SkillWrite:
    """A synthetic ``write_file`` for a supporting-file post-image (path+size review shape)."""
    return SkillWrite(action="write_file", name=name, file_path=file_path,
                      file_content=post, origin=origin)


def build_patch_write(name: str, old: str | None, new: str | None, file_path: str | None,
                      origin: str) -> SkillWrite:
    """The true ``patch``-shaped write, so the formal reviewer keeps its delta view."""
    return SkillWrite(action="patch", name=name, old_string=old, new_string=new,
                      file_path=file_path, origin=origin)


def _reviewable_of(action: str, content: str | None, file_content: str | None) -> str:
    """The text used as the rejection-record dedupe key for a non-patch write."""
    return {"create": content, "edit": content, "write_file": file_content}.get(action) or ""


# --------------------------------------------------------------------------- #
# Synthesized records (pure)
# --------------------------------------------------------------------------- #
def _deadline_record(write: SkillWrite) -> DecisionRecord:
    """A VETO record for a panel that exceeded the deadline (fail-closed, non-quality)."""
    v = Verdict(
        reviewer="panel", decision=Decision.VETO, severity=Severity.HIGH, confidence=0.0,
        evidence=(Evidence("reviewer-deadline",
                           "Panel exceeded the review deadline (fail-closed)."),),
        impacted_scope=("reviewer",),
        rationale="Review deadline exceeded; blocked fail-closed.", depth=Depth.FULL,
    )
    return DecisionRecord(
        SCHEMA_VERSION,
        WriteTarget(write.action, write.name, write.origin, write.file_path),
        Decision.VETO, (v,),
    )


def _error_record(action: str, name: str, origin: str, file_path: str | None,
                  detail: str) -> DecisionRecord:
    """A VETO record for a gate-level inability to review (e.g. un-reconstructable patch)."""
    v = Verdict(
        reviewer="gate", decision=Decision.VETO, severity=Severity.HIGH, confidence=0.0,
        evidence=(Evidence("reviewer-error", detail),), impacted_scope=("reviewer",),
        rationale=detail, depth=Depth.FULL,
    )
    return DecisionRecord(SCHEMA_VERSION, WriteTarget(action, name, origin, file_path),
                          Decision.VETO, (v,))


def _panel_error_record(write: SkillWrite) -> DecisionRecord:
    return _error_record(write.action, write.name, write.origin, write.file_path,
                         "Reviewer panel failed to produce a verdict (fail-closed).")


def _merge(base: DecisionRecord, extra: tuple[Verdict, ...]) -> DecisionRecord:
    """Fold supplementary verdict(s) (the patch monotonicity check) into the record."""
    if not extra:
        return base
    verdicts = tuple(base.verdicts) + tuple(extra)
    blocked = base.is_blocked or any(v.decision is Decision.VETO for v in extra)
    return DecisionRecord(base.schema_version, base.target,
                          Decision.VETO if blocked else Decision.PASS, verdicts)


# --------------------------------------------------------------------------- #
# Actions (I/O, isolated)
# --------------------------------------------------------------------------- #
_DEFAULT_DEADLINE_SECONDS = 30.0
_MAX_DEADLINE_SECONDS = 86400.0  # 1 day; also rejects inf/nan so join(timeout=) can't overflow


def _deadline_seconds() -> float:
    try:
        from hermes_cli.config import cfg_get, load_config

        v = float(cfg_get(load_config(), "skills", "review_gate", "deadline_seconds",
                          default=_DEFAULT_DEADLINE_SECONDS))
        # Reject non-positive AND out-of-range (inf/nan/huge) → honest default; a bad value
        # must not turn the hard bound into an unbounded/overflowing join (fail-closed cleanly).
        return v if 0 < v <= _MAX_DEADLINE_SECONDS else _DEFAULT_DEADLINE_SECONDS
    except Exception:
        return _DEFAULT_DEADLINE_SECONDS


def _max_attempts() -> int:
    """Bounded retry budget (INV-7). Default 1 (no extra retry — rely on call_llm's internal
    retry; interview Q2). Clamped to [1, 5]."""
    try:
        from hermes_cli.config import cfg_get, load_config

        v = int(cfg_get(load_config(), "skills", "review_gate", "max_attempts", default=1))
        return v if 1 <= v <= 5 else 1
    except Exception:
        return 1


def _build_panel() -> Panel:
    """Instantiate the enabled reviewers (Panel auto-orders deterministic-first)."""
    from tools.skill_review.reviewers.contract import ContractReviewer
    from tools.skill_review.reviewers.formal_invariants import FormalInvariantsReviewer
    from tools.skill_review.reviewers.safety import SafetyReviewer
    from tools.skill_review.reviewers.security import SecurityReviewer
    from tools.skill_review.reviewers.tool_workflow import ToolWorkflowReviewer

    candidates = (
        ("contract", ContractReviewer), ("formal", FormalInvariantsReviewer),
        ("tool_workflow", ToolWorkflowReviewer), ("security", SecurityReviewer),
        ("safety", SafetyReviewer),
    )
    return Panel([cls() for rid, cls in candidates if reviewer_enabled(rid)])


def _read_current(name: str, file_path: str | None) -> str | None:
    """Read the current on-disk skill text for post-image reconstruction (gate-level read).

    Imports the writer's own path helpers (``_validate_file_path`` / ``_resolve_skill_target``)
    so the gate can never read outside the skill dir (F1/traversal parity with ``_patch_skill``).
    This is a deliberate lazy import back into the caller (F9): both directions are function-
    local (no import cycle), and any failure returns ``None`` ⇒ the caller fails **closed**
    (patch blocked), so a future seam refactor cannot silently bypass reconstruction.
    """
    try:
        from tools.skill_manager_tool import (
            _find_skill,
            _resolve_skill_target,
            _validate_file_path,
        )

        found = _find_skill(name)
        if not found:
            return None
        skill_dir = found["path"]
        if file_path:
            if _validate_file_path(file_path):          # truthy ⇒ invalid (traversal, etc.)
                return None
            target, err = _resolve_skill_target(skill_dir, file_path)
            if err or target is None:
                return None
        else:
            target = skill_dir / "SKILL.md"
        return target.read_text(encoding="utf-8") if target.exists() else None
    except Exception:
        return None


def run_panel_bounded(panel: Panel, write: SkillWrite, deadline: float) -> DecisionRecord:
    """Run the panel under a hard wall-clock deadline in a daemon worker thread.

    On timeout (or an unexpected worker error), synthesize a record and **abandon** the worker
    — a daemon thread cannot block interpreter exit, and the stuck work is bounded by
    ``call_llm``'s own internal timeout (INV-7, R6). A ``BoundedSemaphore`` caps how many such
    workers can be live at once so repeated timeouts cannot accumulate unbounded work (F3); a
    saturated gate fails **closed**. The context is copied so the origin / runtime ContextVars
    the reviewers and ``call_llm`` read are visible inside the worker.
    """
    if not deadline or deadline <= 0:
        try:
            return panel.review(write, PanelMode.GATE)
        except BaseException:  # noqa: BLE001 - fail closed uniformly; reviewers shouldn't raise
            return _panel_error_record(write)

    if not _review_slots.acquire(blocking=False):
        return _panel_error_record(write)          # gate saturated → fail closed
    box: dict = {}
    ctx = contextvars.copy_context()

    def _target() -> None:
        try:
            box["r"] = ctx.run(panel.review, write, PanelMode.GATE)
        except BaseException as e:  # noqa: BLE001 - the worker must never leak; fail closed
            box["e"] = e
        finally:
            _review_slots.release()

    try:
        t = threading.Thread(target=_target, daemon=True, name="skill-review-panel")
        t.start()
    except BaseException:  # noqa: BLE001 - the worker never ran → release its slot, fail closed
        _review_slots.release()
        return _panel_error_record(write)

    t.join(timeout=deadline)
    if t.is_alive():
        return _deadline_record(write)          # true timeout (worker abandoned; daemon)
    if "e" in box or "r" not in box:
        return _panel_error_record(write)       # worker raised / produced no result
    return box["r"]


def _blocked(record: DecisionRecord, *, reviewable: str) -> "_wa.GateDecision":
    subreason, quality = classify_block(record)
    reason = _CONTRACT_REASON[subreason]
    if subreason == "veto":
        _rec.bump("blocked_veto")
        for v in record.blocking_verdicts():
            _rec.bump_reviewer(v.reviewer)
    else:
        _rec.bump("blocked_unavailable")
    if any("reviewer-deadline" in _locators(v) for v in record.blocking_verdicts()):
        _rec.bump("deadline")
    _rec.write_rejection(record, reason=reason, subreason=subreason,
                         quality_signal=quality, reviewable=reviewable)
    logger.info("skill-review: blocked name=%r reason=%s subreason=%s quality_signal=%s",
                record.target.name, reason, subreason, quality)
    return _wa.GateDecision(blocked=True, message=blocked_message(record, subreason))


def _review(action: str, name: str, content: str | None, file_path: str | None,
            file_content: str | None, old_string: str | None, new_string: str | None,
            replace_all: bool) -> "_wa.GateDecision":
    """The gate body (runs inside review_skill_write's fail-closed wrapper)."""
    origin = _wa.current_origin()
    deadline = _deadline_seconds()
    extra: tuple[Verdict, ...] = ()

    if action == "patch":
        post = reconstruct_post_image(_read_current(name, file_path), old_string,
                                      new_string, replace_all)
        if post is None:
            return _blocked(_error_record(action, name, origin, file_path,
                            "Patch could not be applied to the current skill (delta did not "
                            "match / ambiguous); blocked fail-closed."), reviewable="")
        # Review the post-image under the shape of its TARGET file (F1): the main SKILL.md
        # needs frontmatter (edit shape); a supporting file needs path+size (write_file shape).
        review_write = (build_edit_write(name, post, origin) if _targets_main_skill(file_path)
                        else build_write_file_write(name, file_path, post, origin))
        reviewable = post
        # keep the deterministic permission-monotonicity delta check (S4-R1)
        from tools.skill_review.reviewers.formal_invariants import FormalInvariantsReviewer
        extra = (FormalInvariantsReviewer().review(
                    build_patch_write(name, old_string, new_string, file_path, origin)),)
    elif action == "write_file" and _targets_main_skill(file_path):
        # write_file targeting SKILL.md IS the main skill → require frontmatter (edit shape),
        # not the path+size-only write_file shape that would let invalid SKILL.md slip (F1).
        review_write = build_edit_write(name, file_content or "", origin)
        reviewable = file_content or ""
    else:
        review_write = build_natural_write(action, name, content, file_path,
                                           file_content, origin)
        reviewable = _reviewable_of(action, content, file_content)

    panel = _build_panel()
    if not getattr(panel, "_reviewers", ()):          # enabled but every reviewer toggled off
        return _blocked(_error_record(action, name, origin, file_path,
                        "Review gate enabled but no reviewers are configured; blocked "
                        "fail-closed."), reviewable=reviewable)

    # Bounded retry (INV-7): retry only on an infra/reviewer failure (never a genuine veto),
    # within max_attempts. Default max_attempts=1 ⇒ no extra retry (relies on call_llm's own).
    max_attempts = _max_attempts()
    attempt = 0
    while True:
        attempt += 1
        result = _merge(run_panel_bounded(panel, review_write, deadline), extra)
        if not result.is_blocked:
            _rec.bump("allowed")
            return _wa.GateDecision(allow=True)
        subreason, _q = classify_block(result)
        if subreason == "veto" or attempt >= max_attempts:
            return _blocked(result, reviewable=reviewable)
        _rec.bump("retries")


def review_skill_write(action: str, name: str, *, content: str | None = None,
                       file_path: str | None = None, file_content: str | None = None,
                       old_string: str | None = None, new_string: str | None = None,
                       replace_all: bool = False, **_ignored) -> "_wa.GateDecision":
    """Decide the fate of an agent skill write: ``allow`` (proceed) or ``blocked``.

    A no-op ``allow`` when the gate is disabled, the origin is foreground, or the action is not
    content-bearing (INV-1/INV-2) — no panel is built and no side effect occurs (parity). The
    whole body is wrapped fail-closed: any unexpected error becomes a blocked record, never a
    crash and never an allow (INV-7, F5).
    """
    if not review_gate_enabled():
        return _wa.GateDecision(allow=True)
    if not _wa.is_background():            # foreground writes untouched (INV-2)
        return _wa.GateDecision(allow=True)
    if action not in _GATED_ACTIONS:       # delete/remove_file → existing owned-skill guard
        return _wa.GateDecision(allow=True)

    _rec.bump("seen")
    try:
        return _review(action, name, content, file_path, file_content,
                       old_string, new_string, replace_all)
    except BaseException:  # noqa: BLE001 - uniformly fail closed; never crash the write
        logger.warning("skill-review: gate error; blocking fail-closed", exc_info=True)
        try:
            origin = _wa.current_origin()
        except Exception:
            origin = "background_review"
        return _blocked(_error_record(action, name, origin, file_path,
                        "Skill review failed unexpectedly; blocked fail-closed."),
                        reviewable="")
