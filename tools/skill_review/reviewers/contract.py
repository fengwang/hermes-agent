"""The deterministic contract/schema reviewer.

It formalizes Hermes' EXISTING structural validation rules as read-only verdicts
(it never calls, mutates, or rolls back the live validators — INV-6/INV-4) and adds
three net-new deterministic checks the current validators do not perform:

  * bounded-diff: a ``patch`` whose replacement is rewrite-scale should use ``edit``;
  * over-narrow name: a one-shot / ticket-specific name is not a reusable class skill;
  * tool-permission manifest presence: recorded as an INFORMATIONAL note (never a veto),
    because Hermes does not require ``allowed-tools`` today and real skills omit it.

The mirrored constants/regex below are copied from ``tools.skill_manager_tool``
(verified against the fork on 2026-07-23) rather than imported, to keep this reviewer
decoupled from that module's import surface. Depth is ``FULL``: the reviewer is exact
for its mechanical scope (it does not judge semantics — that is the S2 LLM reviewers).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

import yaml

from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import (
    Decision,
    Depth,
    Evidence,
    GraderType,
    Severity,
    Verdict,
)

# --- Mirrored bounds (source: tools/skill_manager_tool.py, verified 2026-07-23) ----
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000
MAX_SKILL_FILE_BYTES = 1_048_576
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}
_FRONTMATTER_CLOSE_RE = re.compile(r"\n---\s*\n")

# --- Net-new tunable defaults (docs/session_1/design.md §4 D8; calibrated in S5) ---
PATCH_MAX_NEW_CHARS = 4096
_OVER_NARROW_PATTERNS = (
    re.compile(r"(?:pr|issue|ticket|bug|gh|jira|case|cr|mr)[-_#]?\d+"),  # ticket/PR id
    re.compile(r"\d{5,}"),                                              # long id / big number
    re.compile(r"\d{4}-\d{2}-\d{2}"),                                   # ISO date stamp
    re.compile(r"\d{8}"),                                              # yyyymmdd stamp
)

_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


@dataclass(frozen=True)
class _Finding:
    severity: Severity
    locator: str
    detail: str
    scope: str
    blocking: bool


def _veto(severity: Severity, locator: str, detail: str, scope: str) -> _Finding:
    return _Finding(severity, locator, detail, scope, blocking=True)


def _note(locator: str, detail: str, scope: str) -> _Finding:
    return _Finding(Severity.INFO, locator, detail, scope, blocking=False)


def _frontmatter_mapping(content: str | None) -> dict | None:
    """Best-effort parse of the YAML frontmatter to a mapping, else ``None``."""
    text = (content or "").lstrip("\ufeff")
    if not text.startswith("---"):
        return None
    close = _FRONTMATTER_CLOSE_RE.search(text[3:])
    if not close:
        return None
    try:
        parsed = yaml.safe_load(text[3 : close.start() + 3])
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _check_frontmatter(content: str | None) -> list[_Finding]:
    """Mirror of ``_validate_frontmatter`` intent, as verdicts."""
    text = (content or "").lstrip("\ufeff")
    if not text.strip():
        return [_veto(Severity.HIGH, "content", "SKILL.md content is empty.", "content")]
    if not text.startswith("---"):
        return [_veto(Severity.HIGH, "frontmatter",
                      "SKILL.md must start with YAML frontmatter (---).", "frontmatter")]
    close = _FRONTMATTER_CLOSE_RE.search(text[3:])
    if not close:
        return [_veto(Severity.HIGH, "frontmatter",
                      "SKILL.md frontmatter is not closed with a '---' line.", "frontmatter")]
    try:
        parsed = yaml.safe_load(text[3 : close.start() + 3])
    except yaml.YAMLError as exc:
        return [_veto(Severity.HIGH, "frontmatter", f"YAML frontmatter parse error: {exc}", "frontmatter")]
    if not isinstance(parsed, dict):
        return [_veto(Severity.HIGH, "frontmatter",
                      "Frontmatter must be a YAML mapping (key: value pairs).", "frontmatter")]

    findings: list[_Finding] = []
    if "name" not in parsed:
        findings.append(_veto(Severity.HIGH, "frontmatter.name",
                              "Frontmatter must include a 'name' field.", "frontmatter"))
    if "description" not in parsed:
        findings.append(_veto(Severity.HIGH, "frontmatter.description",
                              "Frontmatter must include a 'description' field.", "frontmatter"))
    elif len(str(parsed["description"])) > MAX_DESCRIPTION_LENGTH:
        findings.append(_veto(Severity.MEDIUM, "frontmatter.description",
                              f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters.", "frontmatter"))
    body = text[close.end() + 3 :].strip()
    if not body:
        findings.append(_veto(Severity.HIGH, "body",
                              "SKILL.md has no body after the frontmatter.", "body"))
    return findings


def _check_name(name: str) -> list[_Finding]:
    """Mirror of ``_validate_name`` intent, as verdicts."""
    if not name:
        return [_veto(Severity.HIGH, "name", "Skill name is required.", "name")]
    findings: list[_Finding] = []
    if len(name) > MAX_NAME_LENGTH:
        findings.append(_veto(Severity.MEDIUM, "name",
                              f"Skill name exceeds {MAX_NAME_LENGTH} characters.", "name"))
    if not VALID_NAME_RE.match(name):
        findings.append(_veto(Severity.HIGH, "name",
                              f"Invalid skill name {name!r}: use lowercase letters, digits, "
                              "'.', '-', '_', starting with a letter or digit.", "name"))
    return findings


def _check_content_size(content: str | None) -> list[_Finding]:
    if content is not None and len(content) > MAX_SKILL_CONTENT_CHARS:
        return [_veto(Severity.MEDIUM, "content",
                      f"SKILL.md content is {len(content):,} characters "
                      f"(limit {MAX_SKILL_CONTENT_CHARS:,}).", "content")]
    return []


def _check_write_file_path(file_path: str | None) -> list[_Finding]:
    """Mirror of the allowed-subdir / path rules for supporting files."""
    fp = (file_path or "").strip()
    if not fp:
        return [_veto(Severity.HIGH, "file_path", "file_path is required.", "file_path")]
    parts = PurePosixPath(fp).parts
    if fp.startswith("/") or ".." in parts:
        return [_veto(Severity.HIGH, "file_path",
                      f"File path escapes the skill directory: {fp!r}.", "file_path")]
    if fp == "SKILL.md" or (len(parts) == 2 and parts[1] == "SKILL.md"):
        return []
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return [_veto(Severity.HIGH, "file_path",
                      f"File must be under one of: {allowed}. Got {fp!r}.", "file_path")]
    return []


def _check_file_size(file_content: str | None) -> list[_Finding]:
    """Mirror both caps the live ``_write_file`` enforces: the char cap AND the byte cap."""
    if file_content is None:
        return []
    findings: list[_Finding] = []
    if len(file_content) > MAX_SKILL_CONTENT_CHARS:
        findings.append(_veto(Severity.MEDIUM, "file_content",
                              f"Supporting file is {len(file_content):,} characters "
                              f"(limit {MAX_SKILL_CONTENT_CHARS:,}).", "file_content"))
    size = len(file_content.encode("utf-8"))
    if size > MAX_SKILL_FILE_BYTES:
        findings.append(_veto(Severity.MEDIUM, "file_content",
                              f"Supporting file is {size:,} bytes (limit {MAX_SKILL_FILE_BYTES:,}).",
                              "file_content"))
    return findings


def _check_patch_bounds(new_string: str | None) -> list[_Finding]:
    """Net-new: a ``patch`` declares a targeted change; a rewrite-scale replacement
    should use ``edit`` (which fully re-validates)."""
    if new_string is not None and len(new_string) > PATCH_MAX_NEW_CHARS:
        return [_veto(Severity.MEDIUM, "patch",
                      f"Patch replacement is {len(new_string):,} characters "
                      f"(> {PATCH_MAX_NEW_CHARS:,}); a change this large should use action 'edit'.",
                      "patch")]
    return []


def _check_over_narrow_name(name: str) -> list[_Finding]:
    """Net-new: reject one-shot / ticket-specific names (should be class-level)."""
    lowered = (name or "").lower()
    for pattern in _OVER_NARROW_PATTERNS:
        if pattern.search(lowered):
            return [_veto(Severity.MEDIUM, "name",
                          f"Name {name!r} looks like a one-shot / ticket-specific reference; "
                          "skills should be class-level and reusable.", "name")]
    return []


def _check_tool_manifest(content: str | None) -> list[_Finding]:
    """Net-new (informational): record whether an ``allowed-tools`` manifest is declared."""
    parsed = _frontmatter_mapping(content)
    if parsed is not None and "allowed-tools" not in parsed:
        return [_note("frontmatter.allowed-tools",
                      "Skill does not declare an 'allowed-tools' manifest "
                      "(informational; not required by Hermes today).", "frontmatter")]
    return []


def _contract_findings(write: SkillWrite) -> list[_Finding]:
    action = write.action
    findings: list[_Finding] = []
    if action in ("create", "edit"):
        findings += _check_frontmatter(write.content)
        findings += _check_content_size(write.content)
        findings += _check_tool_manifest(write.content)
    if action == "create":
        findings += _check_name(write.name)
        findings += _check_over_narrow_name(write.name)
    if action == "patch":
        # A patch may target a supporting file; mirror the live path-safety rule.
        if write.file_path:
            findings += _check_write_file_path(write.file_path)
        findings += _check_patch_bounds(write.new_string)
    if action in ("write_file", "remove_file"):
        findings += _check_write_file_path(write.file_path)
    if action == "write_file":
        findings += _check_file_size(write.file_content)
    return findings


def _max_severity(findings: list[_Finding]) -> Severity:
    return max((f.severity for f in findings), key=lambda s: _SEVERITY_RANK[s])


def _aggregate(findings: list[_Finding]) -> Verdict:
    blocking = [f for f in findings if f.blocking]
    decision = Decision.VETO if blocking else Decision.PASS
    severity = _max_severity(findings) if findings else Severity.INFO
    evidence = tuple(Evidence(locator=f.locator, detail=f.detail) for f in findings)
    # ordered-unique scope tags (dict preserves insertion order → deterministic)
    scope = tuple(dict.fromkeys(f.scope for f in findings))
    if blocking:
        rationale = "Contract/schema violations: " + "; ".join(f.detail for f in blocking)
    elif findings:
        rationale = "Contract/schema checks passed with notes: " + "; ".join(f.detail for f in findings)
    else:
        rationale = "Contract/schema checks passed."
    return Verdict(
        reviewer=ContractReviewer.id,
        decision=decision,
        severity=severity,
        confidence=1.0,
        evidence=evidence,
        impacted_scope=scope,
        rationale=rationale,
        depth=Depth.FULL,
    )


class ContractReviewer(Reviewer):
    """Deterministic reviewer that formalizes structural rules as machine-actionable verdicts."""

    id = "contract"
    grader_type = GraderType.DETERMINISTIC

    def review(self, write: SkillWrite) -> Verdict:
        return _aggregate(_contract_findings(write))
