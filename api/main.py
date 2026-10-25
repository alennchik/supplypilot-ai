"""HTTP API for the SupplyPilot planning app.

    .venv/bin/python -m uvicorn api.main:app --reload

GET  /api/health     liveness check
GET  /api/scenarios  what the published Phase 3 run covers, with provenance
POST /api/what-if    one what-if question, answered for both forecasters

Answers come from the published Phase 3 reports (checked against their
checksums when the app starts); nothing is simulated again. When the web app
has been built (web/dist), it is served at /.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

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

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


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


def create_app(report_dir=REPORT_DIR, web_dir=WEB_DIST) -> FastAPI:
    """Build the app around one set of verified reports."""
    reports = load_reports(report_dir)
    app = FastAPI(
        title="SupplyPilot API",
        version="0.4.0",
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

    web = Path(web_dir)
    if (web / "index.html").is_file():
        # Registered after the API routes, so /api/... is never shadowed.
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app


app = create_app()
