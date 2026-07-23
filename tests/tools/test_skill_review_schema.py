"""Schema tests for the skill-review verdict/decision-record interface (S1 freeze).

These lock the forward-compatibility contract (#412): a versioned, immutable,
round-trippable schema that ignores unknown future fields on read. See
docs/session_1/design.md §5, §7 (Requirement: Versioned machine-actionable
verdict schema).
"""
import pytest

from tools.skill_review.schema import (
    SCHEMA_VERSION,
    SchemaError,
    Decision,
    Severity,
    Depth,
    Evidence,
    Verdict,
    WriteTarget,
    DecisionRecord,
)


def _verdict() -> Verdict:
    return Verdict(
        reviewer="contract",
        decision=Decision.VETO,
        severity=Severity.HIGH,
        confidence=1.0,
        evidence=(Evidence(locator="frontmatter.description", detail="missing description"),),
        impacted_scope=("frontmatter",),
        rationale="frontmatter is missing a required field",
        depth=Depth.FULL,
    )


def _record() -> DecisionRecord:
    return DecisionRecord(
        schema_version=SCHEMA_VERSION,
        target=WriteTarget(action="create", name="foo", origin="background_review"),
        decision=Decision.VETO,
        verdicts=(_verdict(),),
    )


class TestSchemaVersion:
    def test_schema_version_is_frozen_string(self):
        assert SCHEMA_VERSION == "1.0"

    def test_record_carries_schema_version(self):
        assert _record().schema_version == SCHEMA_VERSION


class TestRoundTrip:
    def test_verdict_round_trip(self):
        v = _verdict()
        assert Verdict.from_dict(v.to_dict()) == v

    def test_record_round_trip(self):
        r = _record()
        assert DecisionRecord.from_dict(r.to_dict()) == r

    def test_evidence_round_trip(self):
        e = Evidence(locator="name", detail="illegal characters")
        assert Evidence.from_dict(e.to_dict()) == e

    def test_to_dict_uses_enum_string_values(self):
        d = _verdict().to_dict()
        assert d["decision"] == "veto"
        assert d["severity"] == "high"
        assert d["depth"] == "full"

    def test_to_dict_collections_are_lists(self):
        d = _verdict().to_dict()
        assert isinstance(d["evidence"], list)
        assert isinstance(d["impacted_scope"], list)


class TestForwardCompat:
    def test_unknown_verdict_field_ignored(self):
        v = _verdict()
        raw = v.to_dict()
        raw["weight"] = 0.7  # a field a future voting engine might add
        raw["evidence"][0]["confidence"] = 0.9  # unknown nested field
        assert Verdict.from_dict(raw) == v

    def test_unknown_record_field_ignored(self):
        r = _record()
        raw = r.to_dict()
        raw["created_at"] = "2026-07-23T00:00:00Z"  # a future envelope field
        raw["aggregate_score"] = 0.42
        assert DecisionRecord.from_dict(raw) == r

    def test_voting_engine_can_read_per_verdict_fields(self):
        # #412 forward-compat: {severity, confidence, decision} readable per verdict.
        d = _record().to_dict()
        v0 = d["verdicts"][0]
        assert {"severity", "confidence", "decision"} <= set(v0)


class TestValidation:
    def test_confidence_below_range_rejected(self):
        with pytest.raises(SchemaError):
            Verdict(
                reviewer="contract", decision=Decision.PASS, severity=Severity.INFO,
                confidence=-0.1, evidence=(), impacted_scope=(), rationale="", depth=Depth.FULL,
            )

    def test_confidence_above_range_rejected(self):
        with pytest.raises(SchemaError):
            Verdict(
                reviewer="contract", decision=Decision.PASS, severity=Severity.INFO,
                confidence=1.5, evidence=(), impacted_scope=(), rationale="", depth=Depth.FULL,
            )

    def test_empty_reviewer_rejected(self):
        with pytest.raises(SchemaError):
            Verdict(
                reviewer="", decision=Decision.PASS, severity=Severity.INFO,
                confidence=1.0, evidence=(), impacted_scope=(), rationale="", depth=Depth.FULL,
            )

    def test_unknown_decision_value_rejected(self):
        raw = _verdict().to_dict()
        raw["decision"] = "maybe"
        with pytest.raises(SchemaError):
            Verdict.from_dict(raw)

    def test_unknown_severity_value_rejected(self):
        raw = _verdict().to_dict()
        raw["severity"] = "catastrophic"
        with pytest.raises(SchemaError):
            Verdict.from_dict(raw)

    def test_string_enums_coerced_on_construction(self):
        # Ergonomic: from_dict passes raw strings; __post_init__ coerces + validates.
        v = Verdict(
            reviewer="contract", decision="pass", severity="info",
            confidence=1.0, evidence=(), impacted_scope=(), rationale="", depth="full",
        )
        assert v.decision is Decision.PASS and v.severity is Severity.INFO


class TestSequenceValidation:
    """List-typed fields must reject a bare string / non-sequence (frozen-surface safety)."""

    def test_impacted_scope_bare_string_rejected(self):
        # A bare string must NOT be silently char-split into per-character scopes.
        with pytest.raises(SchemaError):
            Verdict(
                reviewer="contract", decision=Decision.PASS, severity=Severity.INFO,
                confidence=1.0, evidence=(), impacted_scope="frontmatter",  # type: ignore[arg-type]
                rationale="", depth=Depth.FULL,
            )

    def test_impacted_scope_list_still_accepted(self):
        v = Verdict(
            reviewer="contract", decision=Decision.PASS, severity=Severity.INFO,
            confidence=1.0, evidence=(), impacted_scope=["frontmatter", "name"],
            rationale="", depth=Depth.FULL,
        )
        assert v.impacted_scope == ("frontmatter", "name")

    def test_non_iterable_evidence_raises_schema_error(self):
        raw = _verdict().to_dict()
        raw["evidence"] = 5
        with pytest.raises(SchemaError):
            Verdict.from_dict(raw)

    def test_non_iterable_verdicts_raises_schema_error(self):
        raw = _record().to_dict()
        raw["verdicts"] = 5
        with pytest.raises(SchemaError):
            DecisionRecord.from_dict(raw)


class TestImmutability:
    def test_verdict_is_frozen(self):
        v = _verdict()
        with pytest.raises(Exception):
            v.reviewer = "other"  # type: ignore[misc]


class TestPublicApi:
    """The package root re-exports the stable public surface."""

    def test_public_symbols_importable_from_package_root(self):
        import tools.skill_review as sr

        for symbol in (
            "SCHEMA_VERSION", "SchemaError", "Decision", "Severity", "Depth",
            "GraderType", "Evidence", "Verdict", "WriteTarget", "DecisionRecord",
            "Reviewer", "SkillWrite", "ContractReviewer", "Panel", "PanelMode",
            "review_gate_enabled", "reviewer_enabled",
        ):
            assert hasattr(sr, symbol), symbol


class TestConfigAccessor:
    """The default-off config accessor (skeleton; not consulted by skill_manage in S1).

    See docs/session_1/design.md §7 (Requirement: Default-off config accessor,
    independent of write_approval). Config reads are stubbed via load_config so the
    scenarios are deterministic and require no edit to hermes_cli/config.py.
    """

    def _patch_config(self, monkeypatch, value):
        import hermes_cli.config as hc

        if isinstance(value, Exception):
            def _raise():
                raise value
            monkeypatch.setattr(hc, "load_config", _raise)
        else:
            monkeypatch.setattr(hc, "load_config", lambda: value)

    def test_default_off_when_unset(self, monkeypatch):
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, {})
        assert review_gate_enabled() is False

    def test_off_when_skills_present_but_no_gate_key(self, monkeypatch):
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, {"skills": {"external_dirs": []}})
        assert review_gate_enabled() is False

    def test_independent_of_write_approval(self, monkeypatch):
        # INV-3: enabling the human-approval feature must not enable the review gate.
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, {"skills": {"write_approval": True}})
        assert review_gate_enabled() is False

    def test_enabled_when_set_true(self, monkeypatch):
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, {"skills": {"review_gate": {"enabled": True}}})
        assert review_gate_enabled() is True

    def test_truthy_string_normalized(self, monkeypatch):
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, {"skills": {"review_gate": {"enabled": "true"}}})
        assert review_gate_enabled() is True

    def test_falsy_string_normalized(self, monkeypatch):
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, {"skills": {"review_gate": {"enabled": "false"}}})
        assert review_gate_enabled() is False

    def test_fail_safe_on_config_error(self, monkeypatch):
        from tools.skill_review.config import review_gate_enabled
        self._patch_config(monkeypatch, RuntimeError("config blew up"))
        assert review_gate_enabled() is False

    def test_reviewer_enabled_defaults_on(self, monkeypatch):
        from tools.skill_review.config import reviewer_enabled
        self._patch_config(monkeypatch, {"skills": {"review_gate": {"enabled": True}}})
        assert reviewer_enabled("contract") is True

    def test_reviewer_can_be_disabled(self, monkeypatch):
        from tools.skill_review.config import reviewer_enabled
        self._patch_config(
            monkeypatch,
            {"skills": {"review_gate": {"reviewers": {"contract": False}}}},
        )
        assert reviewer_enabled("contract") is False
