import pytest

from src.agent_grounding import check_grounding, extract_numbers, matches, mentions


def test_numbers_are_read_with_their_precision():
    numbers = extract_numbers(
        "LightGBM needs 39,157 units (39.2K), 18.5% more, at $44,877 and z = 3.0"
    )

    assert [number.value for number in numbers] == [39157, 39200, 18.5, 44877, 3.0]
    assert [number.precision for number in numbers] == [1, 100, 0.1, 1, 0.1]


def test_names_with_digits_are_not_numbers():
    assert extract_numbers("MA28, the M5 data and Phase3") == []


@pytest.mark.parametrize(
    ("written", "value", "expected"),
    [
        ("92.2%", 92.2, True),
        ("92%", 92.2, True),
        ("93%", 92.2, False),
        ("39.2K", 39157, True),
        ("39,000", 39157, False),
        ("18.5%", 0.185, False),
    ],
)
def test_matching_respects_the_written_precision(written, value, expected):
    (number,) = extract_numbers(written)
    assert matches(number, value) is expected


def test_an_answer_quoting_tool_results_is_grounded():
    result = {"lightgbm": {"average_units_on_hand": 39157}, "difference_percent": 18.5}
    answer = "LightGBM needs 39,157 units, 18.5% more than MA28, for a 90% fill rate at 7 days."

    report = check_grounding(
        answer, [result, {"lead_time_days": 7, "target_fill_rate_percent": 90}]
    )

    assert report.grounded
    assert report.checked == ["39,157", "18.5", "90", "7"]


def test_an_invented_number_is_reported():
    report = check_grounding(
        "LightGBM saves 30% of stock.", [{"difference_percent": 18.5}]
    )

    assert not report.grounded
    assert report.unverified == ["30"]


def test_fractions_match_percentages_and_question_numbers_count():
    report = check_grounding(
        "At 95% fill and 40,000 units, MA28 reaches it.",
        [{"target_fill_rate": 0.95}, "with 40000 units"],
    )
    assert report.grounded


def test_list_markers_small_counts_and_dates():
    answer = "1. Both methods\n2. 2 forecasters over Feb 1 to Apr 24, 2016"
    report = check_grounding(
        answer, [{"evaluation_start": "2016-02-01", "evaluation_end": "2016-04-24"}]
    )

    assert report.grounded
    assert "1" not in report.checked


def test_mentions_finds_a_value_at_any_written_precision():
    assert mentions("about 18.5% more", 18.5)
    assert mentions("needs 39.2K units", 39157)
    assert not mentions("needs 41,000 units", 39157)
