# Skill-Review Static Reviewers — Limitations (S3)

> **Static invariants prove only what is encoded.** (risk R2/R15) The `formal` and
> `tool_workflow` reviewers are **shallow-static** hard-veto checks. Every verdict they emit
> carries `depth: static` and a machine-actionable `deferred-dynamic-checks` evidence note that
> points back here. Downstream consumers (the S4 gate, the S5 eval, a future weighted-voting
> engine) **MUST NOT** treat a `depth: static` pass as behavioral correctness — it is the absence
> of a *statically-detectable* fault, not a proof of safety.

This document is the honest depth statement required by `session_3.md` and the project
`risk_register.md` (R2, R14, R15). It lists, per reviewer, what is checked, what is deliberately
deferred to a later (dynamic / sandbox / TLA+) phase, and why.

## `formal` — formal-invariants reviewer

**Statically checked (hard veto):**
- **Permission monotonicity (patch delta).** A `patch` whose `new_string` `allowed-tools` set
  introduces a tool absent from `old_string`'s set — or widens to a wildcard `*` — is vetoed.
- **Identity consistency.** A `create`/`edit` whose frontmatter `name` differs from the write's
  `name` is vetoed (a net-new check; the contract reviewer validates name *existence* and *format*,
  never *equality*).

**Deliberately deferred / NOT proven:**
- **Cross-version monotonicity for `create`/`edit`.** The frozen `SkillWrite` carries no prior
  version, so widening is only detectable within a `patch` delta. A `create`/`edit` that declares a
  broad scope is not monotonicity-vetoed (there is no baseline to compare).
- **Block-style `allowed-tools` deltas.** Monotonicity parses the inline `allowed-tools: [a, b]`
  form. A YAML block-list (`- a` / `- b`) widening is **not** vetoed. **This check is
  defense-in-depth, not a guarantee** (S2 handoff lesson L7): a determined author can restructure
  the manifest to evade it. Deepen with a full pre/post-image diff at the wiring layer (S4) or the
  sandbox phase.
- **Owned-skill provenance (FI2), pinned protection (FI3), name collision (FI4).** These depend on
  sidecar/filesystem state (`created_by`, `pinned`, existing skill names) that a read-only reviewer
  (INV-4) cannot access, and that the **existing guards already enforce**
  (`_background_review_write_guard`, `_background_review_read_before_write_guard`, `_create_skill`'s
  collision check). The reviewer **defers** to those guards (INV-6: align, do not duplicate); it
  neither re-checks nor weakens them. `tests/tools/test_skill_review_formal.py::TestGuardsIntact`
  proves the real guards still refuse a pinned / non-agent-created write while the reviewer PASSes
  the same write. See `docs/session_3/design.md` §8/§13 for the contract clarification.

## `tool_workflow` — tool/workflow-integrity reviewer

**Statically checked (hard veto):**
- **Non-idempotent retry.** A single fenced-code block / step body that retries a non-idempotent
  network/state mutation (`POST`, `PUT`, `create`, `insert`, `charge`, `deploy`, `send`, `push`)
  with no idempotency guard. Idempotent-by-nature ops (local `write`, `delete`) are excluded so a
  benign retry is not false-vetoed; co-occurrence is scoped to a single unit so a retry in one step
  and an unrelated mutation in another do not combine into a false veto (M2).
- **Use-before-create ordering.** A numbered step list that uses a back-ticked resource, by
  **physical step position**, before the step that creates it.
- **Missing precondition.** A numbered step list with a destructive workflow step (`overwrite`,
  `force-push`, `truncate`, `wipe`, `purge`, `reset --hard`, `drop table`) and no verify/backup/
  guard token in that step or an earlier one.

**Deliberately deferred / NOT proven:**
- **Hallucinated tool (SEED-TW3) — DEFERRED, not statically vetoed.** Detecting a reference to a
  tool the agent does not have needs the **live tool set**, which a read-only reviewer cannot see
  (INV-4). A *static snapshot* of tool names was prototyped and **removed after review**: the Hermes
  tool universe is **open** — configurable toolsets, plugins, and dynamically-registered families
  (`mem0_*`, `spotify_*`, …) plus per-deployment MCP servers — and **cannot be statically
  enumerated**, so any snapshot used as a denylist false-vetoes legitimate skills (M2, unrecoverable
  under deterministic static-wins). Body-level tool-call scanning is likewise M2-dangerous (ordinary
  example code like `print(...)` is call-shaped; common English words like `read`/`process` are also
  tool names). Grounding a tool reference is therefore a **dynamic check** — PRD §9 "tool/workflow →
  sandboxed replay". `tool_workflow` passes tool declarations through untouched.
- **True idempotence / ordering / reachability.** Whether a step *is* idempotent, whether an
  ordering actually deadlocks, and whether an endpoint exists are runtime properties — deferred to
  the sandbox/replay phase. The retry/ordering/precondition regexes are conservative shapes tuned
  in S5; they intentionally under-match (miss synonyms / plain-word variants) rather than risk M2
  false-vetoes.
- **Unsafe-shell / RCE.** Command-injection and fetch-and-exec are **owned by the `security`
  reviewer** (E1 no double-penalty); `tool_workflow` does not re-flag them.

## General

- **Determinism (M3).** Both reviewers are pure Calculations over `SkillWrite`; identical input →
  byte-identical verdict. No wall-clock, RNG, set-iteration, or I/O nondeterminism.
- **All regex patterns are tunable defaults**, like the S1 contract bounds and the S2 static tables.
  Over-broad rules cause M2 false-vetoes that the deterministic static-wins path makes
  **unrecoverable**; calibrate on real Hermes traces in S5.
- **Roadmap.** Deepen `formal` → TLA+/Apalache over the write/promote transition; deepen
  `tool_workflow` → sandboxed tool-replay (`[E:D3]`, PRD §9).
