"""
StudyLens AI - Study Coach entry point for the Orchestrator (Member 1)

    from study_coach_agent import run_study_coach_agent
    out = run_study_coach_agent(quiz_result, exam_date="2026-10-20", hours_per_day=2)

    out["performance"]  -> Performance Agent output (weak/strong topics, insight)
    out["study_plan"]   -> Study Planner Agent output (day-by-day plan)

Return value:
    {"status": "success" | "error", "error": None | "message",
     "performance": {...} | None, "study_plan": {...} | None}

If only the planner fails (e.g. a bad exam date) status is "error" but
"performance" is still filled, so the UI can still show the analysis.

Adaptive use: after the student retakes the quiz, call it again with the new
quiz result and pass the old analysis as previous_performance:

    out2 = run_study_coach_agent(new_quiz_result, exam_date, hours,
                             previous_performance=out["performance"])
"""

from __future__ import annotations

from typing import Optional

from performance_agent import analyze_performance
from study_planner_agent import generate_study_plan


def run_study_coach_agent(quiz_result: dict, exam_date, hours_per_day: float = 2.0,
                      start_date=None, previous_performance: Optional[dict] = None,
                      llm=None) -> dict:
    """One-call entry point: Performance Agent -> Study Planner Agent. Never raises."""
    try:
        perf = analyze_performance(quiz_result, previous_performance, llm)
        if perf["status"] == "error":
            return {"status": "error", "error": perf["error"],
                    "performance": None, "study_plan": None}
        plan = generate_study_plan(perf, exam_date, hours_per_day, start_date, llm)
        return {
            "status": plan["status"], "error": plan["error"],
            "performance": perf,
            "study_plan": plan if plan["status"] == "success" else None,
        }
    except Exception as exc:
        return {"status": "error", "error": f"run_study_coach_agent failed: {exc}",
                "performance": None, "study_plan": None}
