import json

import pytest

from src.agent_models import ModelTurn, ToolCall
from src.inventory_whatif import load_reports
from src.planning_agent import (
    STEP_LIMIT_ANSWER,
    ToolCallRecord,
    run_agent,
    suggested_question,
)


class ScriptedModel:
    """Replays prepared turns and keeps every conversation it was shown."""

    provider = "scripted"
    model = "test"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete(self, system, conversation, tools):
        self.seen.append([dict(item) for item in conversation])
        turn = self.turns.pop(0) if self.turns else self.turns_after_end
        return turn

    turns_after_end = ModelTurn(
        text="", tool_calls=(ToolCall("loop", "list_scenarios", {}),)
    )


@pytest.fixture(scope="module")
def reports():
    return load_reports()


STOCK_CALL = ToolCall(
    "call-1",
    "stock_for_fill_rate",
    {"lead_time_days": 7, "target_fill_rate_percent": 90},
)


def test_an_answer_quoting_the_tools_is_returned_as_grounded(reports):
    model = ScriptedModel(
        [
            ModelTurn(text="", tool_calls=(STOCK_CALL,)),
            ModelTurn(
                text="For 90% fill at a 7-day lead time, LightGBM needs 39,157 units "
                "and MA28 33,044: 18.5% more stock for LightGBM."
            ),
        ]
    )

    reply = run_agent(
        "How much stock for 90% fill at 7 days?", client=model, reports=reports
    )

    assert reply.grounded and not reply.revised
    assert reply.stop_reason == "answered"
    assert [record.name for record in reply.tool_calls] == ["stock_for_fill_rate"]
    assert reply.tool_calls[0].result["ma28"]["average_units_on_hand"] == 33044
    tool_message = model.seen[1][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["results"][0]["is_error"] is False
    payload = reply.to_dict()
    assert payload["suggested_question"] == {"lead_time": 7, "target_fill_rate": 0.9}
    json.dumps(payload, allow_nan=False)


def test_an_invented_number_gets_one_revision(reports):
    model = ScriptedModel(
        [
            ModelTurn(text="", tool_calls=(STOCK_CALL,)),
            ModelTurn(text="LightGBM needs about 41,000 units."),
            ModelTurn(text="LightGBM needs 39,157 units."),
        ]
    )

    reply = run_agent("Stock for 90% at 7 days?", client=model, reports=reports)

    assert reply.grounded and reply.revised
    assert reply.answer == "LightGBM needs 39,157 units."
    revision_request = model.seen[2][-1]
    assert revision_request["role"] == "user"
    assert "41,000" in revision_request["text"]


def test_an_answer_still_ungrounded_after_revision_is_flagged(reports):
    model = ScriptedModel(
        [
            ModelTurn(text="", tool_calls=(STOCK_CALL,)),
            ModelTurn(text="LightGBM saves 30% of stock."),
            ModelTurn(text="LightGBM really saves 30% of stock."),
        ]
    )

    reply = run_agent("Does LightGBM save stock?", client=model, reports=reports)

    assert not reply.grounded
    assert reply.unverified_numbers == ["30"]


def test_tool_errors_go_back_to_the_model(reports):
    model = ScriptedModel(
        [
            ModelTurn(
                text="", tool_calls=(ToolCall("bad", "forecast_sales", {"year": 2027}),)
            ),
            ModelTurn(
                text="I can only answer questions about the published 2016 simulation."
            ),
        ]
    )

    reply = run_agent("Forecast my sales", client=model, reports=reports)

    assert reply.tool_calls[0].error.startswith("Unknown tool")
    assert reply.tool_calls[0].result is None
    tool_message = model.seen[1][-1]
    assert tool_message["results"][0]["is_error"] is True
    assert reply.grounded


def test_the_loop_stops_at_the_step_limit(reports):
    model = ScriptedModel([])

    reply = run_agent("Keep going", client=model, reports=reports, max_steps=3)

    assert reply.stop_reason == "step_limit"
    assert reply.answer == STEP_LIMIT_ANSWER
    assert len(reply.tool_calls) == 3


@pytest.mark.parametrize(
    ("question", "message"), [("   ", "Ask a question"), ("x" * 501, "under 500")]
)
def test_empty_and_overlong_questions_are_refused(reports, question, message):
    with pytest.raises(ValueError, match=message):
        run_agent(question, client=ScriptedModel([]), reports=reports)


def test_suggested_question_merges_successful_calls():
    records = [
        ToolCallRecord(
            "stock_for_fill_rate",
            {"lead_time_days": 3, "target_fill_rate_percent": 95},
            {},
            None,
        ),
        ToolCallRecord(
            "compare_costs",
            {"lead_time_days": 1, "lost_sale_penalty_times_price": 1.0},
            {},
            None,
        ),
        ToolCallRecord(
            "fill_rate_for_stock",
            {"lead_time_days": 5, "average_units_on_hand": 1},
            None,
            "bad",
        ),
        ToolCallRecord(
            "compare_costs",
            {"lead_time_days": 1, "holding_rate_percent_per_year": 30},
            {},
            None,
        ),
    ]

    assert suggested_question(records) == {
        "lead_time": 1,
        "target_fill_rate": 0.95,
        "penalty_fraction": 1.0,
        "holding_rate": 0.3,
    }
