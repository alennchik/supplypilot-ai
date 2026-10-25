"""HTTP tests for the planning API (need FastAPI and httpx, installed in CI)."""

import pytest

from src.agent_models import ModelError, ModelTurn, ToolCall
from src.inventory_whatif import load_reports, scenarios, what_if

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(scope="module")
def reports():
    return load_reports()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    no_web_app = tmp_path_factory.mktemp("no_web")
    return TestClient(create_app(web_dir=no_web_app, agent=None))


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scenarios_come_from_the_engine(client, reports):
    response = client.get("/api/scenarios")

    assert response.status_code == 200
    assert response.json() == scenarios(reports)


def test_an_empty_question_gets_the_primary_answer(client, reports):
    response = client.post("/api/what-if", json={})

    assert response.status_code == 200
    assert response.json() == what_if(reports)


def test_a_custom_question_is_answered_and_labelled(client, reports):
    question = {
        "variant": "empty_start",
        "lead_time": 3,
        "target_fill_rate": 0.9,
        "holding_rate": 0.3,
        "penalty_fraction": 0.75,
        "stock_budget": 30000,
    }

    response = client.post("/api/what-if", json=question)

    assert response.status_code == 200
    body = response.json()
    assert body == what_if(reports, **question)
    assert body["settings"]["costs_preregistered"] is False


@pytest.mark.parametrize(
    ("question", "fragment"),
    [
        ({"lead_time": 5}, "Lead time must be one of"),
        ({"variant": "late_start"}, "Unknown variant"),
        ({"lead_time": "7"}, "lead_time"),
        ({"target_fill_rate": 1.5}, "target_fill_rate"),
        ({"penalty_fraction": -1}, "penalty_fraction"),
        ({"stock_budget": 0}, "stock_budget"),
        ({"unexpected": 1}, "unexpected"),
    ],
)
def test_invalid_questions_get_422(client, question, fragment):
    response = client.post("/api/what-if", json=question)

    assert response.status_code == 422
    assert fragment in response.text


def test_the_built_web_app_is_served_beside_the_api(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>SupplyPilot</title>")
    client = TestClient(create_app(web_dir=tmp_path))

    page = client.get("/")
    health = client.get("/api/health")

    assert page.status_code == 200
    assert "SupplyPilot" in page.text
    assert health.json() == {"status": "ok"}


def test_without_a_build_only_the_api_is_served(client):
    assert client.get("/").status_code == 404


class FakeAgentModel:
    provider = "fake"
    model = "test"

    def __init__(self, turns=()):
        self.turns = list(turns)

    def complete(self, system, conversation, tools):
        if not self.turns:
            raise AssertionError("the model should not have been called")
        return self.turns.pop(0)


def test_the_agent_answers_with_its_tool_calls(tmp_path):
    model = FakeAgentModel(
        [
            ModelTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        "call-1",
                        "stock_for_fill_rate",
                        {"lead_time_days": 7, "target_fill_rate_percent": 90},
                    ),
                ),
            ),
            ModelTurn(text="LightGBM needs 39,157 units and MA28 needs 33,044."),
        ]
    )
    client = TestClient(create_app(web_dir=tmp_path, agent=model))

    status = client.get("/api/agent/status").json()
    response = client.post("/api/agent", json={"question": "Stock for 90% at 7 days?"})

    assert status == {
        "configured": True,
        "provider": "fake",
        "model": "test",
        "problem": None,
    }
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["tool_calls"][0]["name"] == "stock_for_fill_rate"
    assert body["tool_calls"][0]["result"]["ma28"]["average_units_on_hand"] == 33044
    assert body["suggested_question"] == {"lead_time": 7, "target_fill_rate": 0.9}


def test_the_agent_is_off_without_a_key(client):
    status = client.get("/api/agent/status").json()
    response = client.post("/api/agent", json={"question": "Hi"})

    assert status["configured"] is False
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_provider_errors_become_502(tmp_path):
    class Failing(FakeAgentModel):
        def complete(self, system, conversation, tools):
            raise ModelError("anthropic answered 529: overloaded")

    client = TestClient(create_app(web_dir=tmp_path, agent=Failing()))

    response = client.post("/api/agent", json={"question": "Stock for 90%?"})

    assert response.status_code == 502
    assert "overloaded" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "   "}, {"question": "x" * 501}, {"q": "hi"}],
)
def test_bad_agent_questions_get_422(tmp_path, body):
    client = TestClient(create_app(web_dir=tmp_path, agent=FakeAgentModel()))

    assert client.post("/api/agent", json=body).status_code == 422


def test_the_agent_model_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("SUPPLYPILOT_AGENT_PROVIDER", "anthropic")
    status = TestClient(create_app(web_dir=tmp_path)).get("/api/agent/status").json()
    assert (status["configured"], status["provider"]) == (True, "anthropic")

    monkeypatch.setenv("SUPPLYPILOT_AGENT_PROVIDER", "gemini")
    status = TestClient(create_app(web_dir=tmp_path)).get("/api/agent/status").json()
    assert status["configured"] is False
    assert "must be one of" in status["problem"]
