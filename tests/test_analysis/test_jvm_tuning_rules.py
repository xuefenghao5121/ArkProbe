"""Tests for JVM tuning rules in optimization analyzer."""

import pytest

from arkprobe.analysis.optimization_analyzer import TUNING_RULES, TuningRule
from arkprobe.model.enums import TuningLayer, ScenarioType


class TestJVMTuningRulesExist:
    def test_jvm_rules_present(self):
        jvm_rules = [r for r in TUNING_RULES if r.layer == TuningLayer.JVM]
        assert len(jvm_rules) >= 8

    def test_rule_names(self):
        jvm_rules = {r.parameter_name: r for r in TUNING_RULES if r.layer == TuningLayer.JVM}
        expected = {
            "jvm.heap_max_size",
            "jvm.gc_algorithm",
            "jvm.gc_thread_count",
            "jvm.young_gen_size",
            "jvm.metaspace_size",
            "jvm.jit_compiler_threads",
            "jvm.thread_stack_size",
            "jvm.large_pages",
        }
        assert expected.issubset(set(jvm_rules.keys()))


class TestHeapMaxSizeRule:
    @pytest.fixture
    def rule(self) -> TuningRule:
        return next(r for r in TUNING_RULES if r.parameter_name == "jvm.heap_max_size")

    def test_gc_heavy_high_impact(self, rule: TuningRule):
        assert rule.base_impact.get("jvm_gc_heavy", 0) >= 0.7

    def test_jit_intensive_lower_impact(self, rule: TuningRule):
        assert rule.base_impact.get("jvm_jit_intensive", 0) < rule.base_impact.get("jvm_gc_heavy", 0)

    def test_has_gc_pause_condition(self, rule: TuningRule):
        gc_conditions = [c for c in rule.impact_conditions
                         if "gc_pause_ratio" in c.get("metric", "")]
        assert len(gc_conditions) > 0

    def test_has_jvm_scenario_values(self, rule: TuningRule):
        assert "jvm_general" in rule.recommended_values
        assert "jvm_gc_heavy" in rule.recommended_values


class TestGCAlgorithmRule:
    @pytest.fixture
    def rule(self) -> TuningRule:
        return next(r for r in TUNING_RULES if r.parameter_name == "jvm.gc_algorithm")

    def test_gc_heavy_recommends_zgc(self, rule: TuningRule):
        assert rule.recommended_values.get("jvm_gc_heavy") == "ZGC"

    def test_database_oltp_recommends_zgc(self, rule: TuningRule):
        assert rule.recommended_values.get("database_oltp") == "ZGC"

    def test_bigdata_batch_recommends_parallel(self, rule: TuningRule):
        assert rule.recommended_values.get("bigdata_batch") == "Parallel"

    def test_default_is_g1(self, rule: TuningRule):
        assert rule.recommended_values.get("_default") == "G1"

    def test_gc_heavy_highest_impact(self, rule: TuningRule):
        assert rule.base_impact.get("jvm_gc_heavy", 0) >= 0.8


class TestJVMRulesStructure:
    def test_all_jvm_rules_have_apply_template(self):
        jvm_rules = [r for r in TUNING_RULES if r.layer == TuningLayer.JVM]
        for rule in jvm_rules:
            assert rule.apply_template, f"{rule.parameter_name} missing apply_template"

    def test_all_jvm_rules_have_config_path(self):
        jvm_rules = [r for r in TUNING_RULES if r.layer == TuningLayer.JVM]
        for rule in jvm_rules:
            assert rule.config_path.startswith("jvm."), f"{rule.parameter_name} config_path should start with jvm."

    def test_all_jvm_rules_have_default_impact(self):
        jvm_rules = [r for r in TUNING_RULES if r.layer == TuningLayer.JVM]
        for rule in jvm_rules:
            assert "_default" in rule.base_impact, f"{rule.parameter_name} missing _default base_impact"

    def test_difficulty_and_risk_valid(self):
        jvm_rules = [r for r in TUNING_RULES if r.layer == TuningLayer.JVM]
        for rule in jvm_rules:
            assert rule.difficulty is not None
            assert rule.risk is not None
