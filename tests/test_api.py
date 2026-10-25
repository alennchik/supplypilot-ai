"""HTTP tests for the planning API (need FastAPI and httpx, installed in CI)."""

import pytest

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
    return TestClient(create_app(web_dir=no_web_app))


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
