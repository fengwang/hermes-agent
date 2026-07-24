**No evidence-backed calibration warranted this session.** The eval ran the full labelled
corpus (eval_seed_cases §2–§4 + the S1–S4 harvested seeds) through the frozen panel in EVAL mode
and surfaced **no reproducible deterministic miss (M1) or false-veto (M2)** on the dev split:
per-class M1 = 0 for every reviewer-enforced class, M2 = 0 over the should-allow set, injection
attack-success = 0, and failure-labeling = correct. The reviewers were built (S1–S4) against these
seeds with regression tests, so the offline eval confirms — rather than corrects — them.

Per the amended contract (interview Q1/Q2), calibration is **evidence-gated + holdout-validated**;
with no reproducible dev-split miss to act on, the correct, honest outcome is **zero edits** to
`tools/skill_review/llm.py` and `reviewers/**`. Any future calibration (e.g. once trace-derived
seeds reveal a real gap) must follow the same loop: RED test → minimal static-pattern edit →
frozen S1–S4 reviewer suite + 246 anchors green → holdout M1/M2 non-regression → logged here.
