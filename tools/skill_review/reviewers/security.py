"""The security/privacy reviewer: deterministic static pre-filter, then an LLM rubric pass.

Static-wins (D2): any blocking static finding vetoes *before* the model is consulted, so the
mechanical injection/exfil/secret/RCE classes are un-talk-out-able (R3). The LLM rubric judges
the subtler cases (e.g. least-privilege) the static layer defers to it.

The static checks are a *focused, in-memory* filter — deliberately complementary to (not a
reuse of) ``tools/skills_guard.py`` (which is a filesystem-`Path` install-gate). Per INV-6 we
align with its spirit without duplicating or weakening it. Patterns are tunable defaults
(calibrated on real traces in S5), like S1's contract thresholds.
"""
from __future__ import annotations

import re

from tools.skill_review.llm import (
    Finding,
    check_obfuscated_payload,
    load_rubric,
    matches_any,
    reviewable_text,
    review_with_llm,
    signal_finding,
    strip_zero_width,
    veto_finding,
)
from tools.skill_review.reviewers.base import Reviewer, SkillWrite
from tools.skill_review.schema import GraderType, Severity, Verdict

_RUBRIC = load_rubric("security")   # read once at import; review() does no file I/O (INV-4)

# --- static pattern tables (OWASP-LLM + skills_guard spirit, verified 2026-07-23) ----
_UNSAFE_SHELL = (
    re.compile(r"\b(?:bash|sh|zsh)\s+-c\s+[\"']?\$"),        # bash -c "$var"
    re.compile(r"\beval\s+[\"']?\$"),                          # eval $var / eval "$var"
    re.compile(r"\bos\.system\s*\([^)]*\+"),                  # os.system("..." + var)
    re.compile(r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True"),  # subprocess(..., shell=True)
)
_RCE = (
    re.compile(r"(?:curl|wget)\b[^\n|]*\|\s*(?:bash|sh|zsh|python\d?)\b", re.I),   # curl … | bash
    re.compile(r"\beval\s+[\"']?\$\("),                                             # eval "$(…)"
    # any decoder piped into a shell (base64 -d / --decode, xxd -r, openssl, uudecode, …)
    re.compile(r"\b(?:base64|xxd|openssl|uudecode|gunzip|gzip|xz)\b[^\n|]*\|\s*(?:bash|sh|zsh|python\d?)\b", re.I),
    re.compile(r"\b(?:bash|sh|zsh|python\d?)\s+<\(", re.I),                          # process substitution: bash <(…)
    re.compile(r"\b(?:bash|sh|source|python\d?)\s+\S*(?:references|scripts|assets|templates)/\S+", re.I),
    re.compile(r"\b(?:bash|sh)\s+\S+\.(?:sh|bash)\b", re.I),                         # bash foo.sh
)
_SECRET = (
    re.compile(r"\b[A-Z][A-Z0-9_]*(?:SECRET|TOKEN|API_?KEY|PASSWORD|CREDENTIAL)[A-Z0-9_]*\s*=\s*\S"),
    re.compile(r"\becho\s+[\"']?\$\{?[A-Za-z_][A-Za-z0-9_]*(?:SECRET|TOKEN|API_?KEY|PASSWORD)", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                             # AWS access-key id
    re.compile(r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"),                # GitHub token
)
_INJECTION = (
    re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|context)", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|the)\b[^\n]{0,40}(?:instructions?|rules?|policy|policies)", re.I),
    re.compile(r"disregard\s+(?:everything|all|any)\b", re.I),
    re.compile(r"\byou\s+are\s+(?:now\s+)?(?:a\s+|the\s+)?(?:reviewer|dan|developer\s+mode)", re.I),
    re.compile(r"(?:output|return|respond\s+with|reply\s+with|give|produce)\s+(?:a\s+)?[\"']?(?:pass|allow|approved?|passing\s+verdict)\b", re.I),
    re.compile(r"this\s+skill\s+is\s+(?:already\s+)?(?:pre-?)?(?:approved|safe|verified|trusted|cleared|reviewed)", re.I),
    re.compile(r"mark\s+(?:this|it|the\s+verdict)\s*(?:skill\s+)?(?:as\s+)?(?:safe|approved|pass)", re.I),
    re.compile(r"(?:pre-?cleared|already\s+cleared|no\s+longer\s+applies\s+to\s+you)", re.I),
)

# exfiltration = a send/upload verb AND a sensitive payload/target (precise → allows SEED-OK6)
_SEND_VERB = re.compile(
    r"\b(?:curl|wget|nc|scp|sftp|ftp|Invoke-WebRequest|requests\.(?:post|put|patch)|httpx\.(?:post|put|patch))\b",
    re.I,
)
# Require the payload to be variable-shaped ($VAR) or a known secret path — a bare prose word
# like "env"/"token" near a download must NOT static-veto (sharded-review A1 / M2 precision).
_SENSITIVE = re.compile(
    r"\$\{?[A-Za-z_]*(?:CONVERSATION|CONTEXT|HISTORY|MESSAGES|TRANSCRIPT|SECRET|TOKEN|API_?KEY|PASSWORD|CREDENTIAL)"
    r"|/etc/passwd|~/\.(?:ssh|aws)\b|\bprintenv\b|\benv\s*\|",
    re.I,
)
def _check_exfiltration(text: str) -> list[Finding]:
    if _SEND_VERB.search(text) and _SENSITIVE.search(text):
        return [veto_finding(Severity.CRITICAL, "exfiltration",
                             "Sends sensitive data (context/secrets) to an external endpoint.", "body")]
    return []


def _check_tool_scope(text: str) -> list[Finding]:
    if re.search(r"allowed-tools:\s*\[?\s*[\"']?\*", text):
        return [signal_finding("frontmatter.allowed-tools",
                               "Requests wildcard tool scope; verify least privilege.",
                               "frontmatter", Severity.MEDIUM)]
    return []


def _static_findings(write: SkillWrite) -> list[Finding]:
    text = strip_zero_width(reviewable_text(write))   # F3: zero-width can't split a keyword
    findings: list[Finding] = []
    if matches_any(text, _UNSAFE_SHELL):
        findings.append(veto_finding(Severity.CRITICAL, "shell",
                                     "Builds/executes a shell command from unsanitized input.", "body"))
    if matches_any(text, _RCE):
        findings.append(veto_finding(Severity.CRITICAL, "rce",
                                     "Fetch-and-exec / decode-and-run of remote or bundled code.", "body"))
    if matches_any(text, _SECRET):
        findings.append(veto_finding(Severity.HIGH, "secret",
                                     "Hard-codes, echoes, or leaks a secret/token/API key.", "body"))
    if matches_any(text, _INJECTION):
        findings.append(veto_finding(Severity.HIGH, "injection",
                                     "Embeds an injection/jailbreak/approval directive (reported as data).", "body"))
    findings += _check_exfiltration(text)
    findings += check_obfuscated_payload(text)
    findings += _check_tool_scope(text)
    return findings


class SecurityReviewer(Reviewer):
    """Security/privacy hard-veto reviewer (static pre-filter → LLM rubric; static-wins)."""

    id = "security"
    grader_type = GraderType.LLM

    def review(self, write: SkillWrite) -> Verdict:
        return review_with_llm(
            write,
            reviewer_id=self.id,
            rubric=_RUBRIC,
            task="skill_review_security",
            static_findings=_static_findings(write),
        )
