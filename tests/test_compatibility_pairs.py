"""Tests for ``bruteforce_canvas.compatibility_pairs``."""

from __future__ import annotations

import pytest

from bruteforce_canvas.compatibility_pairs import (
    PAIR_FAMILY_RULES,
    CompatibilitySeverity,
)
from bruteforce_canvas.router import CompatibilitySeverity as RouterCompatibilitySeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITIES = {s.value for s in CompatibilitySeverity}
ALL_SEVERITIES = {
    "hard_reject",
    "strong_downrank",
    "soft_downrank",
    "boost",
}


def _as_rule(entry: tuple) -> tuple[str, str, CompatibilitySeverity, str]:
    """Accept either a tuple[..., CompatibilitySeverity, ...] or a plain tuple."""
    assert len(entry) == 4, f"rule must be a 4-tuple, got {entry!r}"
    a, b, severity, reason = entry
    assert isinstance(severity, CompatibilitySeverity), (
        f"severity must be CompatibilitySeverity, got {type(severity)}"
    )
    assert isinstance(a, str), f"axis1_value must be str, got {type(a)}"
    assert isinstance(b, str), f"axis2_value must be str, got {type(b)}"
    assert isinstance(reason, str), f"reason must be str, got {type(reason)}"
    return a, b, severity, reason


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuleCount:
    """The module must expose at least 40 rules."""

    def test_rule_count_meets_minimum(self) -> None:
        assert len(PAIR_FAMILY_RULES) >= 40, (
            f"Expected >= 40 rules, got {len(PAIR_FAMILY_RULES)}"
        )

    def test_rule_count_is_int(self) -> None:
        assert isinstance(PAIR_FAMILY_RULES, list)


class TestRuleShape:
    """Every rule must be a 4-tuple with the correct types."""

    def test_all_rules_are_four_tuples(self) -> None:
        bad = [r for r in PAIR_FAMILY_RULES if not isinstance(r, tuple) or len(r) != 4]
        assert not bad, f"Non-4-tuple rules: {bad!r}"

    def test_all_axis_values_are_strings(self) -> None:
        bad = [
            (i, r)
            for i, r in enumerate(PAIR_FAMILY_RULES)
            if not isinstance(r[0], str) or not isinstance(r[1], str)
        ]
        assert not bad, f"Non-string axis values at indices: {bad!r}"

    def test_all_reasons_are_strings(self) -> None:
        bad = [
            (i, r)
            for i, r in enumerate(PAIR_FAMILY_RULES)
            if not isinstance(r[3], str)
        ]
        assert not bad, f"Non-string reasons at indices: {bad!r}"

    def test_all_severities_are_compatibility_severity(self) -> None:
        bad = [
            (i, r)
            for i, r in enumerate(PAIR_FAMILY_RULES)
            if not isinstance(r[2], CompatibilitySeverity)
        ]
        assert not bad, f"Non-CompatibilitySeverity entries at indices: {bad!r}"


class TestSeverityDistribution:
    """All four severity levels must appear in the rule set."""

    def test_all_four_severities_present(self) -> None:
        observed = {r[2].value for r in PAIR_FAMILY_RULES}
        missing = ALL_SEVERITIES - observed
        assert not missing, f"Missing severities: {missing!r}"

    def test_hard_reject_present(self) -> None:
        assert any(r[2] == CompatibilitySeverity.HARD_REJECT for r in PAIR_FAMILY_RULES)

    def test_boost_present(self) -> None:
        assert any(r[2] == CompatibilitySeverity.BOOST for r in PAIR_FAMILY_RULES)

    def test_severity_counts_nonzero(self) -> None:
        for sev in CompatibilitySeverity:
            count = sum(1 for r in PAIR_FAMILY_RULES if r[2] == sev)
            assert count > 0, f"No rules with severity {sev!r}"


class TestSampleRules:
    """Spot-check specific rules against spec §11.11 intent."""

    def test_macro_wide_shot_is_hard_reject(self) -> None:
        """Macro lens cannot serve a wide_shot framing."""
        matches = [
            r
            for r in PAIR_FAMILY_RULES
            if {r[0], r[1]} == {"macro", "wide_shot"}
            and r[2] == CompatibilitySeverity.HARD_REJECT
        ]
        assert matches, "macro × wide_shot should be a HARD_REJECT"

    def test_overhead_flat_lay_birds_eye_boost(self) -> None:
        """overhead_flat_lay and birds_eye are the same perspective — natural boost."""
        matches = [
            r
            for r in PAIR_FAMILY_RULES
            if {r[0], r[1]} == {"overhead_flat_lay", "birds_eye"}
            and r[2] == CompatibilitySeverity.BOOST
        ]
        assert matches, "overhead_flat_lay × birds_eye should be a BOOST"

    def test_studio_softbox_color_accuracy_boost(self) -> None:
        """studio_softbox provides even illumination ideal for color accuracy."""
        matches = [
            r
            for r in PAIR_FAMILY_RULES
            if "studio_softbox" in {r[0], r[1]}
            and "high" in {r[0], r[1]}
            and r[2] == CompatibilitySeverity.BOOST
        ]
        assert matches, (
            "studio_softbox paired with high requires_color_accuracy should be BOOST"
        )

    def test_macro_group_shot_hard_reject(self) -> None:
        """Macro has impossibly shallow DOF for group shots."""
        matches = [
            r
            for r in PAIR_FAMILY_RULES
            if {r[0], r[1]} == {"macro", "group"}
            and r[2] == CompatibilitySeverity.HARD_REJECT
        ]
        assert matches, "macro × group should be a HARD_REJECT"

    def test_monochrome_explicit_color_hard_reject(self) -> None:
        """Monochrome removes all chromatic info; explicit color fields must be rejected."""
        matches = [
            r
            for r in PAIR_FAMILY_RULES
            if "monochrome" in {r[0], r[1]}
            and "yes" in {r[0], r[1]}
            and r[2] == CompatibilitySeverity.HARD_REJECT
        ]
        assert matches, "monochrome × yes (explicit color) should be HARD_REJECT"

    def test_low_poly_photorealism_hard_reject(self) -> None:
        """low_poly is intentionally stylized; incompatible with photorealism."""
        matches = [
            r
            for r in PAIR_FAMILY_RULES
            if {r[0], r[1]} == {"low_poly", "high"}
            and r[2] == CompatibilitySeverity.HARD_REJECT
        ]
        assert matches, "low_poly × high (photorealism) should be HARD_REJECT"

    def test_severity_alias_matches_router_enum(self) -> None:
        """CompatibilitySeverity values from this module must match router.py exactly."""
        router_values = {s.value for s in RouterCompatibilitySeverity}
        module_values = {s.value for s in CompatibilitySeverity}
        assert module_values == router_values, (
            f"Severity value mismatch: module={module_values}, router={router_values}"
        )

    def test_no_duplicate_rules(self) -> None:
        """No two rules should be identical 4-tuples."""
        seen: set[tuple[str, str, str, str]] = set()
        duplicates: list[tuple] = []
        for r in PAIR_FAMILY_RULES:
            key = (r[0], r[1], r[2].value, r[3])
            if key in seen:
                duplicates.append(r)
            seen.add(key)
        assert not duplicates, f"Duplicate 4-tuple rules found: {duplicates!r}"
