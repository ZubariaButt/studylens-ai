"""
Level 1 tests for the Study Coach (Performance Agent + Study Planner Agent).

Run:  python test_study_coach.py
The integration test at the bottom runs only if Member 4's quiz_agent.py is
importable (copy it next to this file, or set PYTHONPATH to Member 4's repo).
"""

import copy
import json
import os
from datetime import date

from performance_agent import analyze_performance, classify
from study_planner_agent import generate_study_plan
from study_coach_agent import run_study_coach_agent

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "sample_quiz_result.json"), encoding="utf-8") as f:
    QUIZ = json.load(f)

START = date(2026, 10, 3)


def minutes_for(plan, topic):
    return next(a["total_minutes"] for a in plan["topic_allocation"] if a["topic"] == topic)


# ------------------------------------------------------------ performance
def test_classify_thresholds():
    assert classify(0) == "weak"
    assert classify(49.9) == "weak"
    assert classify(50) == "needs_practice"
    assert classify(74.9) == "needs_practice"
    assert classify(75) == "strong"
    assert classify(100) == "strong"


def test_analyze_sample():
    perf = analyze_performance(QUIZ)
    assert perf["status"] == "success" and perf["error"] is None
    assert perf["overall"]["percentage"] == 50.0
    assert perf["weak_topics"] == ["Ecology", "Genetics"]          # weakest first
    assert perf["needs_practice_topics"] == ["Evolution"]
    assert perf["strong_topics"] == ["Cell Biology"]
    assert [t["priority_rank"] for t in perf["topics"]] == [1, 2, 3, 4]
    eco = perf["topics"][0]
    assert eco["topic"] == "Ecology" and eco["unanswered"] == 1
    assert len(eco["wrong_questions"]) == 2
    assert perf["insight_source"] == "rules" and "Ecology" in perf["insight"]


def test_performance_errors_never_raise():
    for bad in (None, "x", 5, [], {}, {"status": "error", "error": "boom"},
                {"status": "success", "topic_performance": {}},
                {"topic_performance": {"A": "not a dict"}},
                {"topic_performance": {"A": {}}}):
        out = analyze_performance(bad)
        assert out["status"] == "error" and out["error"], bad


def test_trend_against_previous():
    first = analyze_performance(QUIZ)
    better = copy.deepcopy(QUIZ)
    better["topic_performance"]["Genetics"].update(correct=3, marks=3.0, percentage=100.0)
    second = analyze_performance(better, previous_performance=first)
    gen = next(t for t in second["topics"] if t["topic"] == "Genetics")
    assert gen["trend"]["direction"] == "improved" and gen["trend"]["change"] == 66.7
    # a raw earlier quiz result is accepted too
    third = analyze_performance(better, previous_performance=QUIZ)
    assert next(t for t in third["topics"] if t["topic"] == "Genetics")["trend"]["direction"] == "improved"


def test_llm_insight_and_fallback():
    class GoodLLM:
        def generate_json(self, system, prompt, max_tokens=300):
            return {"insight": "Nice effort - focus on Ecology."}

    class BadLLM:
        def generate_json(self, *a, **k):
            raise RuntimeError("api down")

    assert analyze_performance(QUIZ, llm=GoodLLM())["insight_source"] == "llm"
    out = analyze_performance(QUIZ, llm=BadLLM())
    assert out["status"] == "success" and out["insight_source"] == "rules"


def test_fallback_when_percentage_missing():
    q = {"topic_performance": {"A": {"correct": 1, "total": 4}, "B": {"marks": 3, "max_marks": 4}}}
    perf = analyze_performance(q)
    assert perf["status"] == "success"
    assert {t["topic"]: t["percentage"] for t in perf["topics"]} == {"A": 25.0, "B": 75.0}


# ---------------------------------------------------------------- planner
def test_plan_structure_and_weighting():
    perf = analyze_performance(QUIZ)
    plan = generate_study_plan(perf, "2026-10-10", hours_per_day=2, start_date=START)
    assert plan["status"] == "success"
    days = plan["daily_plan"]
    assert len(days) == 7 and plan["days_until_exam"] == 7
    assert [d["date"] for d in days] == [f"2026-10-{n:02d}" for n in range(3, 10)]
    assert all(d["total_minutes"] == 120 for d in days)
    assert days[-1]["phase"] == "final_revision"
    assert all(d["phase"] == "study" for d in days[:-1])
    assert {b["topic"] for b in days[-1]["blocks"]} == {"Ecology", "Genetics", "Evolution", "Cell Biology"}
    # weaker topic -> more time
    assert minutes_for(plan, "Ecology") > minutes_for(plan, "Genetics") \
        > minutes_for(plan, "Evolution") > minutes_for(plan, "Cell Biology") > 0
    # first appearance of a weak topic lists the questions to redo
    first_eco = next(b for d in days for b in d["blocks"] if b["topic"] == "Ecology")
    assert first_eco["focus_questions"]


def test_checkpoint_on_long_plan():
    plan = generate_study_plan(analyze_performance(QUIZ), "2026-10-24", 2, START)
    notes = [d["note"] for d in plan["daily_plan"] if d["note"] and "Checkpoint" in d["note"]]
    assert len(notes) == 1


def test_exam_tomorrow_single_day():
    plan = generate_study_plan(analyze_performance(QUIZ), "2026-10-04", 1, START)
    assert plan["status"] == "success" and len(plan["daily_plan"]) == 1
    assert plan["daily_plan"][0]["phase"] == "final_revision"


def test_planner_errors_never_raise():
    perf = analyze_performance(QUIZ)
    cases = [
        (None, "2026-10-10", 2, START),
        ({"status": "error", "error": "x"}, "2026-10-10", 2, START),
        ({"status": "success", "topics": []}, "2026-10-10", 2, START),
        (perf, "2026-10-03", 2, START),          # exam today
        (perf, "2026-09-01", 2, START),          # exam in the past
        (perf, "10/10/2026", 2, START),          # wrong format
        (perf, None, 2, START),
        (perf, "2026-10-10", "abc", START),
        (perf, "2026-10-10", 0, START),
        (perf, "2026-10-10", True, START),
    ]
    for p, exam, hours, start in cases:
        out = generate_study_plan(p, exam, hours, start)
        assert out["status"] == "error" and out["error"], (exam, hours)


def test_hours_are_clamped_with_warning():
    plan = generate_study_plan(analyze_performance(QUIZ), "2026-10-10", 0.1, START)
    assert plan["status"] == "success" and plan["hours_per_day"] == 0.5 and plan["warnings"]
    assert all(d["total_minutes"] == 30 for d in plan["daily_plan"])


def test_plan_adapts_when_performance_improves():
    perf1 = analyze_performance(QUIZ)
    plan1 = generate_study_plan(perf1, "2026-10-17", 2, START)
    better = copy.deepcopy(QUIZ)
    better["topic_performance"]["Genetics"].update(correct=3, marks=3.0, percentage=100.0)
    better["results"] = [r for r in better["results"] if r["topic"] != "Genetics"]
    perf2 = analyze_performance(better, previous_performance=perf1)
    plan2 = generate_study_plan(perf2, "2026-10-17", 2, START)
    assert minutes_for(plan2, "Genetics") < minutes_for(plan1, "Genetics")


def test_all_strong_still_gets_a_plan():
    q = {"topic_performance": {"A": {"percentage": 100.0, "total": 3, "correct": 3},
                               "B": {"percentage": 90.0, "total": 3, "correct": 3}}}
    plan = generate_study_plan(analyze_performance(q), "2026-10-10", 1, START)
    assert plan["status"] == "success" and "steady revision" in plan["summary"]


def test_many_topics_tiny_budget():
    q = {"topic_performance": {f"T{i}": {"percentage": float(i * 10), "total": 2, "correct": 1}
                               for i in range(1, 9)}}
    plan = generate_study_plan(analyze_performance(q), "2026-10-05", 0.5, START)
    assert plan["status"] == "success"
    assert all(d["total_minutes"] == 30 for d in plan["daily_plan"])


def test_coach_message_llm():
    class LLM:
        def generate_json(self, system, prompt, max_tokens=200):
            return {"message": "You've got this!"}
    plan = generate_study_plan(analyze_performance(QUIZ), "2026-10-10", 2, START, llm=LLM())
    assert plan["coach_message"] == "You've got this!"


def test_hours_alongside_minutes():
    from study_planner_agent import format_duration, minutes_with_hours
    assert format_duration(45) == "45 min"
    assert format_duration(120) == "2h"
    assert format_duration(105) == "1h 45m"
    assert minutes_with_hours(45) == "45 min"
    assert minutes_with_hours(2100) == "2100 min (35h)"

    plan = generate_study_plan(analyze_performance(QUIZ), "2026-10-10", 2, START)
    assert plan["total_minutes"] == 840 and plan["total_hours"] == 14.0
    assert "840 min (14h) in total" in plan["summary"]
    assert "Ecology: 345 min (5h 45m), " in plan["summary"]
    assert sum(a["total_hours"] for a in plan["topic_allocation"]) == plan["total_hours"]
    assert all(d["total_hours"] == round(d["total_minutes"] / 60, 2) for d in plan["daily_plan"])


# ------------------------------------------------------------ orchestrator
def test_run_study_coach_agent():
    out = run_study_coach_agent(QUIZ, "2026-10-10", 2, start_date=START)
    assert out["status"] == "success"
    assert out["performance"]["weak_topics"] and out["study_plan"]["daily_plan"]
    json.dumps(out)                                   # must be JSON-serialisable

    bad_date = run_study_coach_agent(QUIZ, "not-a-date", 2, start_date=START)
    assert bad_date["status"] == "error" and bad_date["performance"] is None \
        or bad_date["performance"]["status"] == "success"
    assert run_study_coach_agent(None, "2026-10-10")["status"] == "error"


# ------------------------------------------------- integration (Member 4)
def test_integration_with_member4():
    try:
        from quiz_agent import run_quiz_agent
    except ImportError:
        print("  (skipped - quiz_agent.py not importable)")
        return
    questions = {
        "mcqs": [{"question": "Q1?", "options": ["a", "b"], "correct_answer": "a", "topic": "X"},
                 {"question": "Q2?", "options": ["a", "b"], "correct_answer": "a", "topic": "Y"}],
        "true_false": [{"statement": "S?", "answer": True, "topic": "Y"}],
    }
    quiz = run_quiz_agent(questions, {"mcq_1": "a", "mcq_2": "b", "tf_1": "False"})
    out = run_study_coach_agent(quiz, "2026-10-10", 1, start_date=START)
    assert out["status"] == "success"
    assert out["performance"]["strong_topics"] == ["X"]
    assert out["performance"]["weak_topics"] == ["Y"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
