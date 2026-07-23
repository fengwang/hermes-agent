# Skill-review quality gate (`tools/skill_review/`)

> **Status: skeleton, default-off, not wired into the live write path.**
> This package is the first slice of a quality gate for *agent-created* skill
> writes. In this state it is inert: nothing in `skill_manage` /
> `_apply_skill_write_gate` consults it. Wiring is a later change.

## What it is

A small panel of **orthogonal reviewers** inspects a proposed agent skill write
and returns a machine-actionable **verdict**. Orthogonality is by *grader type*
(a deterministic checker vs. an LLM judge vs. …), not by persona. The panel
aggregates verdicts with a **hard-veto** rule: any veto blocks the write.

Today the package ships:

- a **frozen verdict/decision-record schema** (`schema.py`) — the forward-compatible
  interface a future weighted-voting engine can consume without a schema change;
- a read-only **`Reviewer`** interface (`reviewers/base.py`);
- a **`Panel`** (`panel.py`) with deterministic-first ordering and two run modes;
- the deterministic **contract/schema reviewer** (`reviewers/contract.py`);
- default-off **config accessors** (`config.py`).

## The config flag (default off)

The gate is controlled by a new, default-off namespace. Nothing needs to be added
to `hermes_cli/config.py`; an absent key resolves to the default.

```yaml
skills:
  review_gate:
    enabled: false          # master switch (default: false)
    reviewers:
      contract: true        # per-reviewer toggle (default: true when the gate is on)
```

- `review_gate_enabled()` → reads `skills.review_gate.enabled`, default `False`.
- `reviewer_enabled(id)` → reads `skills.review_gate.reviewers.<id>`, default `True`.

This namespace is **independent** of `skills.write_approval` (the human-approval
feature): enabling one never enables the other.

## The verdict schema (frozen, `SCHEMA_VERSION = "1.0"`)

`Verdict` fields: `reviewer`, `decision` (`pass`/`veto`), `severity`
(`critical`/`high`/`medium`/`low`/`info`), `confidence` (0–1; deterministic
reviewers emit `1.0`), `evidence` (`{locator, detail}` items), `impacted_scope`
(scope tags), `rationale`, and `depth` (`full`/`static`).

`DecisionRecord` wraps `{schema_version, target, decision, verdicts[]}`, where
`decision` is the hard-veto aggregate (`veto` iff any verdict is `veto`).

**Freeze contract**

- The enums (`decision`/`severity`/`depth`) are closed and stable.
- Forward-compatibility is via **additive optional fields**: `from_dict` ignores
  unknown keys at every level, so an older reader never breaks on a newer record.
- Verdicts/records carry no wall-clock or ordering state, so deterministic
  reviewers are byte-identical across runs.

Use `to_dict()` / `from_dict()` for (de)serialization.

## Running a review (illustrative — not yet on any live path)

```python
from tools.skill_review import Panel, PanelMode, ContractReviewer, SkillWrite

panel = Panel([ContractReviewer()])
record = panel.review(
    SkillWrite(action="create", name="debugging-workflows", content=skill_md),
    mode=PanelMode.GATE,   # short-circuit on first veto; use EVAL to run all reviewers
)
if record.is_blocked:
    ...  # the aggregated rationale is on record.verdicts / blocking_verdicts()
```

- **`GATE`** mode short-circuits on the first veto (saves tokens at gate time).
- **`EVAL`** mode runs every reviewer to completion (independent scoring at eval time).

## The contract/schema reviewer

Deterministic (`depth=full`, `confidence=1.0`). It **formalizes existing structural
rules as verdicts** — it never calls, mutates, or rolls back the live validators
in `tools/skill_manager_tool.py` (it only mirrors their intent). The mirrored
bounds (name regex, size caps, allowed subdirs, description length) are copied with
a provenance comment rather than imported, to keep the reviewer decoupled.

It also adds three checks the current validators do not perform:

| Check | Effect |
|---|---|
| bounded-diff — a `patch` larger than `PATCH_MAX_NEW_CHARS` (4096) | **veto** (use `edit` for rewrites) |
| over-narrow name — one-shot / ticket-specific names (ticket ids, long numbers, date stamps) | **veto** (skills should be class-level) |
| tool-permission manifest — whether `allowed-tools` is declared | **informational only** (Hermes does not require it) |

`PATCH_MAX_NEW_CHARS` and the over-narrow-name patterns are **tunable defaults**,
to be calibrated against real traces in the eval phase.

## What it does *not* do (yet)

- It is **not** consulted by `_apply_skill_write_gate` (no wiring).
- No LLM reviewers (security/safety), no static formal-invariants / tool-workflow
  reviewers, no fail-closed retry, no rejection logging, no eval harness.
- No soft scoring / weighted voting — the MVP is pure hard-veto.
