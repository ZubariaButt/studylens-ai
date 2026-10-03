"""
StudyLens AI - Study Planner Agent

Flow:  Performance Agent -> generate_study_plan() -> UI (Member 6)

The schedule is computed in plain Python (so it is deterministic and always
valid). An optional LLM only adds a short friendly "coach_message" on top.

INTERFACE CONTRACT (share this with the team)
---------------------------------------------
Input  performance   : output of performance_agent.analyze_performance()
       exam_date     : "YYYY-MM-DD" string (or a datetime.date)
       hours_per_day : study hours available per day (0.5 - 16, default 2)
       start_date    : optional "YYYY-MM-DD"; defaults to today (tests use it)
       llm           : optional, object with generate_json(system, prompt, max_tokens)

Output of generate_study_plan():

    {
      "status": "success" | "error", "error": None | "message",
      "exam_date": "2026-10-20", "start_date": "2026-10-03",
      "days_until_exam": 17, "hours_per_day": 2.0, "warnings": [],
      "total_minutes": 2040, "total_hours": 34.0,
      "topic_allocation": [{"topic", "status", "percentage", "total_minutes",
                            "total_hours", "share_percent"}],
      "daily_plan": [
        {"day": 1, "date": "2026-10-03", "weekday": "Sat",
         "phase": "study" | "final_revision", "total_minutes": 120, "total_hours": 2.0,
         "blocks": [{"topic", "minutes", "status", "activity",
                     "focus_questions": [...]}],
         "note": "..." | None}
      ],
      "summary": "...", "coach_message": None | "..."
    }

How the plan works
  * Day 1 .. N-1 are study days, day N (the day before the exam) is final revision.
  * Each topic gets time proportional to how weak it is: weight = max(15, 100 - score%).
    Weak topics get the most time, strong topics a light recap, and every topic
    appears at least once.
  * Time is split in 15-minute slots and spread across days so weak topics come
    back again and again (spaced practice) instead of one long block.
  * ADAPTIVE: call it again with the new quiz's performance and today's date -
    the plan is rebuilt for the remaining days with the updated weak topics.
    On a plan of 5+ study days a "checkpoint" day reminds the student to retake the quiz.

On failure: {"status": "error", "error": "message", ...} - the agent never raises.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

SLOT_MINUTES = 15
MIN_HOURS, MAX_HOURS = 0.5, 16.0
WEIGHT_FLOOR = 15.0            # even a 100% topic keeps a small share of time
CHECKPOINT_MIN_STUDY_DAYS = 5
MAX_FOCUS_QUESTIONS = 3

STUDY_ACTIVITY = {
    "weak": "Re-learn the core concepts from your notes, then redo the questions you missed",
    "needs_practice": "Practise questions on this topic and review the explanations for your mistakes",
    "strong": "Quick recap and a few practice questions to stay sharp",
}
FINAL_ACTIVITY = {
    "weak": "Final pass over your key weak points and previously missed questions",
    "needs_practice": "Skim your summary notes and retry a few missed questions",
    "strong": "Quick skim of key points only",
}
FINAL_NOTE = ("Retake the StudyLens quiz today as a mock check, then rest well "
              "before your exam.")
CHECKPOINT_NOTE = ("Checkpoint: retake the StudyLens quiz today so your plan can be "
                   "updated with your new scores.")


# ----------------------------------------------------------------- helpers
def format_duration(minutes) -> str:
    """Human-friendly length: 45 -> '45 min', 120 -> '2h', 105 -> '1h 45m'."""
    m = int(round(minutes))
    if m < 60:
        return f"{m} min"
    hours, rest = divmod(m, 60)
    return f"{hours}h" if rest == 0 else f"{hours}h {rest}m"


def minutes_with_hours(minutes) -> str:
    """Minutes with hours in brackets: 2100 -> '2100 min (35h)'; under an hour -> '45 min'."""
    m = int(round(minutes))
    return f"{m} min" if m < 60 else f"{m} min ({format_duration(m)})"


def _error(message: str) -> dict:
    return {
        "status": "error", "error": message, "exam_date": None, "start_date": None,
        "days_until_exam": 0, "hours_per_day": 0, "warnings": [],
        "total_minutes": 0, "total_hours": 0.0,
        "topic_allocation": [], "daily_plan": [], "summary": "", "coach_message": None,
    }


def _parse_date(value, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            pass
    raise ValueError(f"{label} must be a date in YYYY-MM-DD format (got {value!r}).")


def _allocate(total_slots: int, topics: list[dict]) -> dict:
    """Split total_slots between topics by weight (topics are weakest-first).
    Every topic gets at least one slot when there are enough slots; if not,
    the weakest topics get them first."""
    alloc = {t["topic"]: 0 for t in topics}
    if total_slots <= 0 or not topics:
        return alloc
    if total_slots < len(topics):
        for t in topics[:total_slots]:
            alloc[t["topic"]] = 1
        return alloc

    for t in topics:
        alloc[t["topic"]] = 1
    spare = total_slots - len(topics)
    total_weight = sum(t["weight"] for t in topics)
    exact = {t["topic"]: spare * t["weight"] / total_weight for t in topics}
    for name, value in exact.items():
        alloc[name] += int(value)
    leftover = spare - sum(int(v) for v in exact.values())
    by_remainder = sorted(
        enumerate(topics),
        key=lambda it: (-(exact[it[1]["topic"]] - int(exact[it[1]["topic"]])), it[0]),
    )
    for _, t in by_remainder[:leftover]:
        alloc[t["topic"]] += 1
    return alloc


def _distribute(num_days: int, slots_per_day: int, alloc: dict, order: list[str]) -> list[dict]:
    """Spread each topic's slots across the days in proportion to what is left,
    so weak topics reappear every day. Returns one {topic: slots} dict per day."""
    remaining = dict(alloc)
    rank = {name: i for i, name in enumerate(order)}
    days = []
    for d in range(num_days):
        days_left = num_days - d
        quota = {t: remaining[t] / days_left for t in order}
        counts: dict = {}
        for _ in range(slots_per_day):
            candidates = [t for t in order if remaining[t] > 0]
            if not candidates:
                break
            pick = max(candidates, key=lambda t: (quota[t] - counts.get(t, 0), -rank[t]))
            counts[pick] = counts.get(pick, 0) + 1
            remaining[pick] -= 1
        days.append(counts)
    return days


def _shorten(text, limit: int = 120) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _blocks(counts: dict, order: list[str], by_name: dict, final: bool, seen: set) -> list[dict]:
    blocks = []
    for name in order:                         # weakest first - tackle hard stuff while fresh
        slots = counts.get(name, 0)
        if not slots:
            continue
        info = by_name[name]
        status = info["status"]
        focus: list[str] = []
        if name not in seen and status != "strong":
            focus = [_shorten(q.get("question")) for q in info["wrong_questions"][:MAX_FOCUS_QUESTIONS]
                     if q.get("question")]
        seen.add(name)
        blocks.append({
            "topic": name,
            "minutes": slots * SLOT_MINUTES,
            "status": status,
            "activity": (FINAL_ACTIVITY if final else STUDY_ACTIVITY)[status],
            "focus_questions": focus,
        })
    return blocks


def _llm_coach_message(llm, performance: dict, summary: str) -> Optional[str]:
    if llm is None:
        return None
    system = (
        "You are a supportive study coach. In 2 short sentences, motivate the student "
        "about their study plan. Use ONLY the facts provided; do not invent numbers or "
        'topics. Return ONLY JSON: {"message": "..."}'
    )
    prompt = f"{performance.get('insight', '')}\n{summary}"
    try:
        out = llm.generate_json(system, prompt, max_tokens=200)
        text = str(out.get("message", "")).strip() if isinstance(out, dict) else ""
        return text or None
    except Exception:
        return None


# -------------------------------------------------------------- main entry
def generate_study_plan(performance: dict, exam_date, hours_per_day: float = 2.0,
                        start_date=None, llm=None) -> dict:
    """Build a personalised, weakness-weighted study plan. Never raises."""
    try:
        if not isinstance(performance, dict):
            return _error("performance must be a dict (the Performance Agent output).")
        if performance.get("status") == "error":
            return _error(f"Performance Agent reported an error: {performance.get('error')}")
        raw_topics = performance.get("topics")
        if not isinstance(raw_topics, list) or not raw_topics:
            return _error("performance has no topics. Run analyze_performance() first.")

        try:
            exam = _parse_date(exam_date, "exam_date")
            start = _parse_date(start_date, "start_date") if start_date is not None else date.today()
        except ValueError as exc:
            return _error(str(exc))

        if isinstance(hours_per_day, bool):
            return _error("hours_per_day must be a number.")
        try:
            hours = float(hours_per_day)
        except (TypeError, ValueError):
            return _error("hours_per_day must be a number.")
        if not hours > 0:
            return _error("hours_per_day must be greater than 0.")

        warnings: list[str] = []
        if hours < MIN_HOURS or hours > MAX_HOURS:
            clamped = min(max(hours, MIN_HOURS), MAX_HOURS)
            warnings.append(f"hours_per_day adjusted from {hours:g} to {clamped:g} "
                            f"(allowed range {MIN_HOURS:g}-{MAX_HOURS:g}).")
            hours = clamped

        days_available = (exam - start).days
        if days_available < 1:
            return _error(f"The exam date ({exam.isoformat()}) must be after the start date "
                          f"({start.isoformat()}).")

        # weakest first, each with a weight
        topics = []
        for t in raw_topics:
            if not isinstance(t, dict) or "topic" not in t or "percentage" not in t:
                return _error("Each performance topic needs 'topic' and 'percentage'.")
            topics.append({
                "topic": t["topic"],
                "percentage": float(t["percentage"]),
                "status": t.get("status") if t.get("status") in STUDY_ACTIVITY else "needs_practice",
                "weight": max(WEIGHT_FLOOR, 100.0 - float(t["percentage"])),
                "wrong_questions": t.get("wrong_questions") or [],
            })
        topics.sort(key=lambda t: (t["percentage"], str(t["topic"])))
        order = [t["topic"] for t in topics]
        by_name = {t["topic"]: t for t in topics}

        slots_per_day = max(1, int(round(hours * 60 / SLOT_MINUTES)))
        n_study = days_available - 1       # last day before the exam is final revision

        study_counts = _distribute(
            n_study, slots_per_day, _allocate(n_study * slots_per_day, topics), order
        ) if n_study else []
        final_counts = _allocate(slots_per_day, topics)

        checkpoint_day = n_study // 2 if n_study >= CHECKPOINT_MIN_STUDY_DAYS else None
        seen: set = set()
        daily_plan = []
        for i in range(days_available):
            current = start + timedelta(days=i)
            is_final = i == days_available - 1
            blocks = _blocks(final_counts if is_final else study_counts[i], order, by_name,
                             is_final, seen)
            note = None
            if is_final:
                note = FINAL_NOTE
            elif checkpoint_day is not None and i + 1 == checkpoint_day:
                note = CHECKPOINT_NOTE
            elif i == 0:
                note = "Start with your weakest topic while your mind is fresh."
            daily_plan.append({
                "day": i + 1,
                "date": current.isoformat(),
                "weekday": current.strftime("%a"),
                "phase": "final_revision" if is_final else "study",
                "total_minutes": sum(b["minutes"] for b in blocks),
                "total_hours": round(sum(b["minutes"] for b in blocks) / 60, 2),
                "blocks": blocks,
                "note": note,
            })

        minutes = {name: 0 for name in order}
        for day in daily_plan:
            for b in day["blocks"]:
                minutes[b["topic"]] += b["minutes"]
        grand_total = sum(minutes.values()) or 1
        topic_allocation = [
            {"topic": name, "status": by_name[name]["status"],
             "percentage": by_name[name]["percentage"],
             "total_minutes": minutes[name],
             "total_hours": round(minutes[name] / 60, 2),
             "share_percent": round(100 * minutes[name] / grand_total, 1)}
            for name in order
        ]

        top = max(topic_allocation, key=lambda a: (a["total_minutes"], -order.index(a["topic"])))
        summary = (
            f"{days_available} day{'s' if days_available != 1 else ''} until your exam on "
            f"{exam.isoformat()}, studying {hours:g}h/day, {minutes_with_hours(grand_total)} in total. "
        )
        if top["status"] == "strong":
            summary += "Your scores are strong everywhere, so the plan focuses on steady revision. "
        else:
            summary += (f"Most time goes to {top['topic']}: "
                        f"{minutes_with_hours(top['total_minutes'])}, "
                        f"{top['share_percent']:g}% of the plan. ")
        summary += "Retake the quiz to refresh the plan as you improve."

        return {
            "status": "success", "error": None,
            "exam_date": exam.isoformat(), "start_date": start.isoformat(),
            "days_until_exam": days_available, "hours_per_day": hours, "warnings": warnings,
            "total_minutes": grand_total, "total_hours": round(grand_total / 60, 2),
            "topic_allocation": topic_allocation, "daily_plan": daily_plan,
            "summary": summary,
            "coach_message": _llm_coach_message(llm, performance, summary),
        }
    except Exception as exc:  # never crash the orchestrator
        return _error(f"generate_study_plan failed: {exc}")
