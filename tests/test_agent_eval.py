import json
import math

import pytest

from scripts import run_agent_eval
from src.agent_eval import (
    CASES,
    EvalCase,
    ExpectedCall,
    call_matches,
    expected_quotes,
    resolve,
    run_evaluation,
)
from src.agent_models import ModelTurn, ToolCall
from src.inventory_whatif import load_reports


@pytest.fixture(scope="module")
def reports():
    return load_reports()


class OracleModel:
    """Answers every case the way a correct agent would, from real tool results."""

    provider = "oracle"
    model = "test"

    def __init__(self):
        self.cases = {case.question: case for case in CASES}

    def complete(self, system, conversation, tools):
        case = self.cases[conversation[0]["text"]]
        results = [
            result
            for item in conversation
            if item["role"] == "tool"
            for result in item["results"]
        ]
        if case.calls and not results:
            calls = []
            for index, call in enumerate(case.calls):
                arguments = dict(call.arguments)
                if call.tool == "stock_for_fill_rate":
                    arguments.setdefault("target_fill_rate_percent", 95)
                calls.append(ToolCall(f"call-{index}", call.tool, arguments))
            return ModelTurn(text="", tool_calls=tuple(calls))
        if case.must_decline:
            return ModelTurn(
                text="I can only answer questions about the published 2016 simulation."
            )
        parts = [
            f"{path} is {resolve(results[index]['content'], path)}"
            for index, path in case.quotes
        ]
        parts += [alternatives[0] for alternatives in case.wording]
        return ModelTurn(text="; ".join(parts) + ".")


class GuessingModel:
    """Answers without tools and invents a number."""

    provider = "guess"
    model = "test"

    def complete(self, system, conversation, tools):
        return ModelTurn(text="LightGBM needs about 41,000 units.")


def test_case_ids_are_unique_and_expected_values_exist(reports):
    assert len({case.id for case in CASES}) == len(CASES)
    for case in CASES:
        for _, value in expected_quotes(reports, case):
            assert math.isfinite(value)


def test_a_correct_agent_passes_every_case(reports):
    summary = run_evaluation(reports, OracleModel())

    failures = {
        result["id"]: result["failures"]
        for result in summary["results"]
        if not result["passed"]
    }
    assert failures == {}
    assert summary["passed"] == summary["cases"] == len(CASES)
    assert summary["pass_rate"] == 1.0
    json.dumps(summary, allow_nan=False)


def test_a_guessing_agent_fails_with_reasons(reports):
    case = next(case for case in CASES if case.id == "stock-90-7d")

    summary = run_evaluation(reports, GuessingModel(), cases=(case,))

    failures = summary["results"][0]["failures"]
    assert any(
        failure.startswith("missing call stock_for_fill_rate") for failure in failures
    )
    assert any(failure.startswith("does not quote") for failure in failures)
    assert any(failure.startswith("unverified numbers: 41,000") for failure in failures)


def test_a_crashing_model_counts_as_a_failure(reports):
    class Broken:
        provider = "broken"
        model = "test"

        def complete(self, system, conversation, tools):
            raise RuntimeError("provider down")

    summary = run_evaluation(reports, Broken(), cases=CASES[:1])

    assert summary["passed"] == 0
    assert summary["results"][0]["failures"] == ["error: provider down"]


def test_call_matching_treats_a_missing_variant_as_primary():
    expected = ExpectedCall(
        "stock_for_fill_rate", {"lead_time_days": 7, "target_fill_rate_percent": 90}
    )

    assert call_matches(
        {"lead_time_days": 7, "target_fill_rate_percent": 90.0}, expected
    )
    assert call_matches(
        {"lead_time_days": 7, "target_fill_rate_percent": 90, "variant": "primary"},
        expected,
    )
    assert not call_matches(
        {"lead_time_days": 7, "target_fill_rate_percent": 90, "variant": "half_up"},
        expected,
    )
    assert not call_matches(
        {"lead_time_days": 3, "target_fill_rate_percent": 90}, expected
    )


def test_resolve_reads_nested_paths():
    assert resolve({"a": {"b": [10, 20]}}, "a.b.1") == 20


def test_quoting_an_empty_value_is_a_case_error(reports):
    case = EvalCase(
        id="broken",
        question="?",
        calls=(
            ExpectedCall(
                "stock_for_fill_rate",
                {"lead_time_days": 7, "target_fill_rate_percent": 95},
            ),
        ),
        quotes=((0, "lightgbm.average_units_on_hand"),),
    )

    with pytest.raises(ValueError, match="cannot be quoted"):
        expected_quotes(reports, case)


def test_the_live_script_needs_a_key(monkeypatch, capsys, tmp_path):
    for variable in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "SUPPLYPILOT_AGENT_PROVIDER",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(run_agent_eval, "ROOT", tmp_path)

    assert run_agent_eval.main([]) == 2
    assert "No API key found" in capsys.readouterr().out


def test_reports_are_written_with_a_safe_name(tmp_path):
    summary = {
        "provider": "anthropic",
        "model": "claude/odd name",
        "run_at_utc": "2026-10-25T17:30:00+00:00",
        "cases": 0,
        "passed": 0,
        "results": [],
    }

    path = run_agent_eval.write_report(summary, tmp_path)

    assert path.name == "eval_anthropic_claude-odd-name_20261025T173000.json"
    assert json.loads(path.read_text())["provider"] == "anthropic"
