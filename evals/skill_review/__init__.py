"""Offline eval for the skill-review gate (S5).

Runs the frozen reviewer panel in EVAL mode over a labelled seed corpus and reports the
initiative's acceptance metrics (M1 missed-bad, M2 false-veto, M3 determinism, per-reviewer
confusion, injection attack-success, failure-labeling correctness). Imports
``tools.skill_review`` READ-ONLY; never touches the live ``skill_manage`` write path.
"""
