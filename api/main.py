"""HTTP API for the SupplyPilot planning app.

    .venv/bin/python -m uvicorn api.main:app --reload

GET  /api/health        liveness check
GET  /api/scenarios     what the published Phase 3 run covers, with provenance
POST /api/what-if       one what-if question, answered for both forecasters
GET  /api/agent/status  whether the planning agent has a model configured
POST /api/agent         one plain-English question for the planning agent

Answers come from the published Phase 3 reports (checked against their
checksums when the app starts); nothing is simulated again. The planning
agent is on only when an API key is set in the environment or in a .env file
at the project root (see src/agent_models.py). When the web app has been
built (web/dist), it is served at /.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.agent_models import ModelError, model_from_environment, read_env_file
from src.inventory_whatif import (
    DEFAULT_PENALTY,
    HOLDING_RATE_RANGE,
    PENALTY_RANGE,
    REPORT_DIR,
    STOCK_BUDGET_MAX,
    load_reports,
    scenarios,
    what_if,
)
from src.planning_agent import MAX_QUESTION_CHARS, run_agent

ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = ROOT / "web" / "dist"
ENV_FILE = ROOT / ".env"
AGENT_OFF = (
    "The planning agent is off: put ANTHROPIC_API_KEY (or OPENAI_API_KEY) in a .env "
    "file at the project root and restart the server."
)

# create_app() reads the agent's model from the environment unless one is passed.
FROM_ENVIRONMENT = object()


class WhatIfRequest(BaseModel):
    """A what-if question. Omitted fields take the protocol's primary settings."""

    model_config = ConfigDict(extra="forbid")

    variant: str = "primary"
    lead_time: int = Field(7, strict=True)
    target_fill_rate: float = Field(0.95, gt=0, le=1)
    holding_rate: float = Field(
        0.25, ge=HOLDING_RATE_RANGE[0], le=HOLDING_RATE_RANGE[1]
    )
    penalty_fraction: float = Field(
        DEFAULT_PENALTY, ge=PENALTY_RANGE[0], le=PENALTY_RANGE[1]
    )
    stock_budget: float | None = Field(None, gt=0, le=STOCK_BUDGET_MAX)


class AgentRequest(BaseModel):
    """A plain-English question for the planning agent."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def _agent_from_environment():
    """The configured model and, if configuration failed, why."""
    environ = {**read_env_file(ENV_FILE), **os.environ}
    try:
        return model_from_environment(environ), None
    except ValueError as exc:
        return None, str(exc)


def create_app(
    report_dir=REPORT_DIR, web_dir=WEB_DIST, agent=FROM_ENVIRONMENT
) -> FastAPI:
    """Build the app around one set of verified reports and an optional agent model."""
    reports = load_reports(report_dir)
    problem = None
    if agent is FROM_ENVIRONMENT:
        agent, problem = _agent_from_environment()

    app = FastAPI(
        title="SupplyPilot API",
        version="0.5.0",
        summary="Inventory what-if over the preregistered Phase 3 simulation.",
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/scenarios")
    def get_scenarios() -> dict:
        return scenarios(reports)

    @app.post("/api/what-if")
    def post_what_if(question: WhatIfRequest) -> dict:
        try:
            return what_if(reports, **question.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/status")
    def agent_status() -> dict:
        return {
            "configured": agent is not None,
            "provider": getattr(agent, "provider", None),
            "model": getattr(agent, "model", None),
            "problem": problem,
        }

    @app.post("/api/agent")
    def ask_agent(request: AgentRequest) -> dict:
        if agent is None:
            raise HTTPException(status_code=503, detail=problem or AGENT_OFF)
        try:
            reply = run_agent(request.question, client=agent, reports=reports)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return reply.to_dict()

    web = Path(web_dir)
    if (web / "index.html").is_file():
        # Registered after the API routes, so /api/... is never shadowed.
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app


app = create_app()
