"""Render the eval metrics into a reproducible report artifact (S5).

Pure rendering (Grokking Simplicity): ``render_*`` are Calculations — a function of the
evaluation values only, with **no wall-clock** and no ordering state, so two runs over the same
corpus produce byte-identical artifacts (spec eval_report). ``write_report`` is the one Action.

To keep the module dependency graph acyclic (``harness -> report``), this module imports the
harness types only under ``TYPE_CHECKING`` and reads everything via attribute access at runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:   # annotations only; no runtime import (avoids the harness<->report cycle)
    from evals.skill_review.harness import Corpus, Evaluation
    from evals.skill_review.metrics import ClassM1, M2Result, Metrics

_NO_CALIBRATION = ("No evidence-backed calibration warranted this session: the eval surfaced no "
                   "reproducible deterministic miss/false-veto on the dev split.")


# --------------------------------------------------------------------------- #
# formatting helpers (pure)
# --------------------------------------------------------------------------- #
def _rate(numerator: int, total: int) -> str:
    if not total:
        return "n/a (0 seeds)"
    return f"{numerator / total:.3f} ({numerator}/{total})"


def _m1_line(c: "ClassM1") -> str:
    tail = f" — missed {list(c.missed_ids)}" if c.missed_ids else ""
    return f"| {c.cls} | {_rate(c.missed, c.total)} |{tail} |"


def _m2_line(m2: "M2Result") -> str:
    tail = f" — false-veto {list(m2.false_veto_ids)}" if m2.false_veto_ids else ""
    return f"{_rate(m2.false_veto, m2.total)}{tail}"


# --------------------------------------------------------------------------- #
# summary (pure) — PR-pasteable
# --------------------------------------------------------------------------- #
def render_summary(ev: "Evaluation") -> str:
    m: "Metrics" = ev.metrics
    inv = m.invariants
    all_split = m.split("all")
    lines = [
        "## Skill-review eval — summary",
        f"- **M3 determinism**: {'PASS' if ev.determinism.holds else 'FAIL'} "
        f"(runs={ev.determinism.runs}, deterministic reviewers={list(ev.determinism.reviewers)})",
        f"- **Injection attack-success (ADV1–3)**: {_rate(inv.attack_flipped, inv.attack_total)} "
        f"(target 0)",
        f"- **Failure-labeling correctness**: {_rate(inv.label_correct, inv.label_total)}",
        f"- **False-veto (M2, all split)**: {_m2_line(all_split.m2)}",
        "- **Missed-bad (M1, all split)** by class: "
        + "; ".join(f"{c.cls} {_rate(c.missed, c.total)}" for c in all_split.m1),
        f"- **Deterministic missed-bad**: {list(m.deterministic_misses) or 'none'}",
        f"- **Holdout regression guard**: {ev.baseline_diff.status}",
        f"- **Corpus/holdout integrity**: {'ok' if not ev.integrity else 'DRIFT'}",
        f"- **CI verdict**: {'PASS' if ev.verdict.ok else 'FAIL'}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# full markdown (pure)
# --------------------------------------------------------------------------- #
def render_markdown(corpus: "Corpus", ev: "Evaluation", *, calibration_log: str | None = None) -> str:
    m: "Metrics" = ev.metrics
    inv = m.invariants
    out: list[str] = ["# Skill-Review Gate — Offline Eval Report", "",
                      render_summary(ev), ""]

    # Per-class M1, per split
    out += ["## Missed-bad (M1) — per class, per split",
            "> Over *reviewer-enforced* should-veto seeds only; guard/dynamic-enforced seeds are "
            "in Deferred coverage."]
    for split in m.splits:
        out += [f"### split: {split.name}", "| class | M1 (missed/total) | detail |", "|---|---|---|"]
        out += [_m1_line(c) for c in split.m1] or ["| _(none)_ | n/a | |"]
        out.append("")

    # M2 — aggregate per split + per class (codex-F2)
    out += ["## False-veto (M2) — should-allow blocked", "| split | M2 (false/total) |", "|---|---|"]
    for split in m.splits:
        out.append(f"| {split.name} | {_m2_line(split.m2)} |")
    by_class = m.split("all").m2_by_class
    if by_class:
        out += ["", "Per-class M2 (all split):", "| class | M2 (false/total) |", "|---|---|"]
        out += [f"| {cls} | {_m2_line(m2)} |" for cls, m2 in by_class]
    attest = m.split("all").m2.attribution
    if attest:
        out += ["", "False-veto attribution (all split):"]
        out += [f"- `{sid}` blocked by `{rev}`" for sid, rev in attest]
    out.append("")

    # Determinism
    out += ["## Determinism (M3)",
            f"- runs: {ev.determinism.runs}; deterministic reviewers: "
            f"{list(ev.determinism.reviewers)}",
            f"- result: {'PASS (byte-identical across runs)' if ev.determinism.holds else 'FAIL'}"]
    if ev.determinism.unstable:
        out += [f"- unstable: {list(ev.determinism.unstable)}"]
    out.append("")

    # Per-reviewer confusion (codex-F3: partial multi-reviewer misses shown, not marked correct)
    out += ["## Per-reviewer confusion (veto seeds)",
            "| seed | class | expected | fired | missing | verdict |",
            "|---|---|---|---|---|---|"]
    for e in m.confusion:
        out.append(f"| {e.seed_id} | {e.cls} | {list(e.expected_reviewers)} | "
                   f"{list(e.firing_reviewers)} | {list(e.missing_expected)} | {e.verdict} |")
    flagged = [e.seed_id for e in m.confusion if e.verdict in ("misattributed", "partial")]
    if flagged:
        out += ["", f"⚠ mis-attributed / partial (an expected reviewer did not fire): {flagged}"]
    out.append("")

    # Injection + labeling
    out += ["## Injection resilience & failure-labeling",
            f"- **attack-success (ADV1–3)**: {_rate(inv.attack_flipped, inv.attack_total)}"
            + (f" — flipped {list(inv.attack_flipped_ids)}" if inv.attack_flipped else " — none flipped"),
            f"- **failure-labeling**: {_rate(inv.label_correct, inv.label_total)}"
            + (f" — mismatches {[list(x) for x in inv.label_mismatches]}"
               if inv.label_mismatches else " (ADV5/ADV11 → blocked-by-reviewer-unavailable)"),
            ""]

    # Deferred coverage
    out += ["## Deferred coverage (enforced by guards / the dynamic phase, NOT the panel)",
            "> Excluded from panel M1. The existing guards are proven intact by the S3/S4 anchors "
            "(`tests/tools/test_skill_review_formal.py::TestGuardsIntact` + the 246 regression "
            "anchors); hallucinated-tool (TW3) is deferred to sandbox replay (PRD §9)."]
    if m.deferred:
        out += ["", "| seed | class | enforcement | panel decision |", "|---|---|---|---|"]
        out += [f"| {sid} | {cls} | {enf} | {dec} |" for sid, cls, enf, dec in m.deferred]
    out += ["",
            "_Gate-level cases (S4 corpus ADV10 fuzzy-patch, ADV11 raising-reviewer, ADV12 empty "
            "panel, ADV13 traversal, ADV14 deadline-clamp, ADV15 deadline-exceeded, ADV16 "
            "non-applying-patch) are out of panel-eval scope and covered by the S4 gate anchors._",
            ""]

    # Holdout composition
    out += ["## Holdout composition (frozen)",
            f"- size: {len(corpus.holdout_ids)} seeds; ids: {sorted(corpus.holdout_ids)}",
            "- provenance + content hashes: `evals/skill_review/fixtures/holdout.yaml` "
            "(class-stratified floor(n/3) by sha256(seed_id); synthetic corpus — real "
            "independence arrives with trace-derived seeds).", ""]

    # Baseline diff
    out += ["## Holdout regression guard",
            f"- status: **{ev.baseline_diff.status}**"]
    out += [f"  - {d}" for d in ev.baseline_diff.details]
    out.append("")

    # Corpus & holdout integrity (silent-drop / leakage guards)
    out += ["## Corpus & holdout integrity",
            f"- deterministic missed-bad (CI-gated): {list(m.deterministic_misses) or 'none'}",
            f"- inventory: {ev.inventory.get('total')} seeds; reviewer-enforced veto/class = "
            f"{ev.inventory.get('reviewer_enforced_veto_per_class')}",
            f"- integrity check: **{'ok' if not ev.integrity else 'DRIFT'}**"]
    out += [f"  - {issue}" for issue in ev.integrity]
    out.append("")

    # Calibration log
    out += ["## Calibration log", calibration_log or _NO_CALIBRATION, ""]

    # Live delta
    if ev.live_delta:
        out += ["## Live vs canned delta (logged, not gated)",
                "| seed | canned (blocked/label) | live (blocked/label) |", "|---|---|---|"]
        out += [f"| {sid} | {c} | {live} |" for sid, c, live in ev.live_delta]
        out.append("")

    # CI verdict
    out += ["## CI verdict", f"**{'PASS' if ev.verdict.ok else 'FAIL'}**"]
    out += [f"- {r}" for r in ev.verdict.reasons]
    out.append("")
    return "\n".join(out)


def render_json(corpus: "Corpus", ev: "Evaluation") -> str:
    payload = {
        "metrics": ev.metrics.to_dict(),
        "determinism": {"runs": ev.determinism.runs, "holds": ev.determinism.holds,
                        "reviewers": list(ev.determinism.reviewers),
                        "unstable": [list(u) for u in ev.determinism.unstable]},
        "baseline_diff": {"status": ev.baseline_diff.status, "details": list(ev.baseline_diff.details)},
        "verdict": {"ok": ev.verdict.ok, "reasons": list(ev.verdict.reasons)},
        "holdout_ids": sorted(corpus.holdout_ids),
        "integrity": list(ev.integrity),
        "inventory": ev.inventory,
        "live_delta": [list(x) for x in ev.live_delta],
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# write (Action)
# --------------------------------------------------------------------------- #
def write_report(out_dir: Path, markdown: str, json_text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    (out_dir / "report.json").write_text(json_text, encoding="utf-8")
