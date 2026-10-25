"""Run the planning agent's evaluation questions against a live model.

    .venv/bin/python -m scripts.run_agent_eval
    .venv/bin/python -m scripts.run_agent_eval --cases stock-90-7d,injection

Reads the API key from the environment or a .env file at the project root
(see src/agent_models.py), asks every question, scores the answers, prints a
table and saves the full report under reports/phase4_agent/. Each case makes
a few model calls, which the provider bills to that key. CI never runs this:
it checks the same harness with a scripted model instead.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.agent_eval import CASES, run_evaluation
from src.agent_models import model_from_environment, read_env_file
from src.inventory_whatif import load_reports

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "phase4_agent"


def source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_report(summary: dict, output_dir: Path) -> Path:
    stamp = summary["run_at_utc"].replace(":", "").replace("-", "")[:15]
    model = re.sub(r"[^A-Za-z0-9.-]+", "-", summary["model"]) or "model"
    path = Path(output_dir) / f"eval_{summary['provider']}_{model}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="Comma-separated case ids (default: all)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    cases = CASES
    if args.cases:
        wanted = {
            case_id.strip() for case_id in args.cases.split(",") if case_id.strip()
        }
        cases = tuple(case for case in CASES if case.id in wanted)
        unknown = wanted - {case.id for case in cases}
        if unknown:
            parser.error(f"unknown case ids: {', '.join(sorted(unknown))}")

    environ = {**read_env_file(ROOT / ".env"), **os.environ}
    client = model_from_environment(environ)
    if client is None:
        print(
            "No API key found. Put ANTHROPIC_API_KEY (or OPENAI_API_KEY) in a .env "
            "file at the project root, or export it, then run this again."
        )
        return 2

    print(f"Asking {len(cases)} questions to {client.provider} / {client.model} ...")
    summary = run_evaluation(load_reports(), client, cases)
    summary["run_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary["source_commit"] = source_commit()

    for result in summary["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}  {result['id']}")
        for failure in result["failures"]:
            print(f"      {failure}")
    print(f"\n{summary['passed']} of {summary['cases']} cases passed.")
    print(f"Report: {write_report(summary, args.output_dir)}")
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
