# Skill-review quality gate — configuration

The skill-review gate runs a small panel of orthogonal reviewers over **agent-created**
skill writes (the background self-improvement fork and the curator) and blocks clearly-bad
skills before they go active. It is **off by default**, independent of `skills.write_approval`,
and reversible: turning it off restores the exact pre-feature behavior.

Foreground / user-directed skill writes are **never** gated. `delete` and `remove_file` are
not gated (they are protected by the existing owned-skill guard).

## Config keys (`config.yaml`)

```yaml
skills:
  review_gate:
    enabled: false          # master switch (kill-switch). false ⇒ pre-feature behavior.
    deadline_seconds: 30     # hard wall-clock bound for the whole panel on one write.
    max_attempts: 1          # bounded retry budget; 1 = no extra retry (see below)
    reviewers:               # per-reviewer toggles (all on by default)
      contract: true
      formal: true
      tool_workflow: true
      security: true
      safety: true
```

- **`enabled`** — when `false` (default) the gate is a no-op returning "allow" before any
  panel is built; behavior is byte-identical to pre-feature. Set `true` to enforce.
- **`deadline_seconds`** — the panel runs in a bounded worker thread; if it exceeds this
  wall-clock budget the write is blocked as *reviewer-unavailable* (not a quality veto) and
  the stuck work is abandoned (a daemon thread cannot hang the agent). Because only
  off-critical-path agent writes are gated, the worst case is "skill not saved this pass".
  Tunable; calibrate on real traces (do not set so low that legitimate reviews are cut off).
- **`max_attempts`** — total panel attempts per write (bounded retry, clamped to `[1, 5]`). A
  retry fires only on an infra/reviewer failure (never a genuine veto), within the deadline
  budget; each retry increments the `retries` counter. Default `1` (no extra retry — the
  auxiliary client already retries internally). Raising it costs up to `max_attempts × deadline`
  wall-clock per write.
- **`reviewers.<id>`** — turn an individual reviewer off to remove it from the panel; the
  remaining reviewers still gate. Deterministic reviewers (`contract`, `formal`,
  `tool_workflow`) always run before the LLM reviewers (`security`, `safety`), and a
  deterministic veto short-circuits before any model is called.

## Decision semantics

| Situation | Outcome |
|---|---|
| Gate off, or a foreground write | proceed unchanged (allow) |
| Agent write, all reviewers pass | proceed to the real write |
| Agent write, any reviewer veto | blocked (`tool_error`); record `reason=blocked-by-veto` |
| Reviewer transport error / unparseable reply / deadline / gate error | blocked (fail-closed); record `reason=blocked-by-reviewer-unavailable` (**not** a quality signal) |
| Review gate enabled but failed to load / no reviewers configured | blocked (fail-closed) — an enabled gate never silently bypasses |

The gate never silently drops a write: the only non-proceed outcome is an explicit, logged
`blocked`. It also never writes the skill itself or changes lifecycle state — it is a pure
decision plus an audit record.

## Patch handling

For a `patch`, the gate reconstructs the **post-patch artifact** (current on-disk content +
the delta) and reviews *that* (as an edit), so the reviewers judge the skill that would
actually go active — not just the delta. It additionally runs the formal reviewer on the raw
delta so a patch that widens `allowed-tools` is still deterministically vetoed. A patch whose
`old_string` is not found in the current content is blocked fail-closed (the real artifact
cannot be verified).

## Rejection records

Each blocked agent write writes a best-effort JSON record to
`<HERMES_HOME>/skill_review/rejections/<key>.json`, keyed by `sha256(name + reviewed-content)`
so identical rejected writes dedupe. The write is **atomic** (temp file + `os.replace`) so a
concurrent same-key block can't leave a partial file. Each record carries the contract
`reason` (`blocked-by-veto` or `blocked-by-reviewer-unavailable`), a finer `subreason`
(`veto` / `reviewer_unavailable` / `reviewer_error`), and a `quality_signal` boolean — only a
genuine content veto is a quality signal. Persisted rationale/evidence is length-capped and
secret-redacted (the reviewed skill is untrusted). These records are written by direct file
I/O (they never re-enter the skill-write path). Consuming them to suppress regeneration is a
future enhancement.

## Observability

In-process counters track writes `seen`, `allowed`, `blocked_veto`, `blocked_unavailable`,
`deadline`, and per-reviewer veto tallies; each decision also emits a structured log line.
A rising `blocked_unavailable`/`deadline` count signals an infra problem (a learning-stall),
distinct from `blocked_veto` (the gate doing its job).
