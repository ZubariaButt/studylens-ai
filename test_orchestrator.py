"""
Level 2 integration tests for the Orchestrator (Member 1).

Run:  python test_orchestrator.py

Needs these files in the same folder: orchestrator.py, quiz_agent.py (M4),
performance_agent.py + study_planner_agent.py (M5). document_agent.py (M2) is used
for the real-PDF test. Member 3's agents and the Groq LLM are replaced by fakes,
so NO API key and NO internet are needed.
"""

import json
from datetime import date, timedelta

from orchestrator import StudyLensOrchestrator, clean_question_set, simulate_answers

EXAM = (date.today() + timedelta(days=10)).isoformat()


# ----------------------------------------------------------------- fakes
class FakeDoc:
    def __init__(self, text="Cells contain organelles.\n\n" * 40, status="success"):
        self.text, self.status = text, status

    def process_document(self, path):
        if self.status != "success":
            return {"status": "error", "message": "File not found."}
        return {"status": "success", "file_name": "fake.pdf", "total_pages": 2,
                "total_characters": len(self.text), "total_chunks": 3, "full_text": self.text}


class FakeTutor:
    def process(self, text):
        return {"status": "success", "summary": "A summary.", "key_points": ["a"],
                "explanations": [], "difficult_terms": []}


class BrokenTutor:
    def process(self, text):
        raise RuntimeError("groq exploded")


MESSY = {
    "status": "success",
    "mcqs": [
        {"question": "Powerhouse of the cell?", "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi"],
         "correct_answer": "B", "explanation": "ATP.", "topic": "cell biology"},
        {"question": "Which base pairs with A?", "options": ["G", "T", "C", "U"],
         "correct_answer": "t", "topic": "Genetics"},
        {"question": "Broken question?", "options": ["a", "b"], "correct_answer": "zzz", "topic": "Genetics"},
        {"question": "Protein synthesis site?", "options": ["Ribosome", "Lysosome"],
         "correct_answer": "Ribosome", "topic": "Cell Biology "},
    ],
    "true_false": [
        {"statement": "DNA is single stranded.", "answer": "false", "topic": "genetics"},
        {"statement": "Ribosomes make proteins.", "answer": True, "topic": "Cell Biology"},
        {"statement": "No answer given.", "topic": "X"},
    ],
    "short_questions": [{"question": "What is DNA?", "answer": "A molecule that carries genetic information",
                         "topic": "Genetics"}],
    "long_questions": [],
}


class FakeQuestions:
    def __init__(self, data=MESSY):
        self.data = data

    def generate(self, text):
        return self.data


class FakeLLM:
    def generate_json(self, system, prompt, max_tokens=300):
        if "grade" in system.lower():
            return {"score": 1.0, "feedback": "Good answer."}
        return {"insight": "Nice effort!", "message": "You can do it!"}


class DeadLLM:
    def generate_json(self, *a, **k):
        raise RuntimeError("api down")


def make(**kw):
    defaults = dict(llm=FakeLLM(), document_agent=FakeDoc(), tutor_agent=FakeTutor(),
                    question_agent=FakeQuestions())
    defaults.update(kw)
    return StudyLensOrchestrator(**defaults)


def ready_quiz(**kw):
    orch = make(**kw)
    assert orch.process_document("x.pdf")["status"] == "success"
    assert orch.run_tutor()["status"] == "success"
    assert orch.generate_questions()["status"] == "success"
    assert orch.start_quiz()["status"] == "success"
    return orch


def tiny_pdf(lines):
    """Build a minimal one-page PDF by hand (no extra libraries needed)."""
    stream = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({l}) '" for l in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    pdf += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    pdf += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return pdf.encode("latin-1")


# ----------------------------------------------------------------- tests
def test_clean_question_set():
    clean, warnings = clean_question_set(MESSY)
    assert [q["correct_answer"] for q in clean["mcqs"]] == ["Mitochondria", "T", "Ribosome"]
    assert [q["answer"] for q in clean["true_false"]] == [False, True]
    topics = {q["topic"] for q in clean["mcqs"] + clean["true_false"] + clean["short_questions"]}
    assert topics == {"Cell Biology", "Genetics"}
    assert len(warnings) == 2                       # the broken MCQ and the T/F without answer


def test_full_pipeline_with_fakes():
    orch = make()
    res = orch.run_pipeline("x.pdf", lambda qs: simulate_answers(orch.state["question_set"]),
                            exam_date=EXAM, hours_per_day=2)
    assert res["status"] == "success", res
    s = res["state"]
    assert s["quiz_result"]["total"] == 5 and s["quiz_result"]["score"] == 4
    assert set(s["performance"]["weak_topics"] + s["performance"]["needs_practice_topics"]
               + s["performance"]["strong_topics"]) == {"Cell Biology", "Genetics"}
    assert s["performance"]["insight_source"] == "llm"
    assert s["study_plan"]["status"] == "success" and s["study_plan"]["daily_plan"]
    assert [e["stage"] for e in res["log"]] == ["document", "tutor", "questions", "quiz_prepare",
                                                "quiz_evaluate", "planner"]
    json.dumps(res)                                  # whole result must be JSON-serialisable
    assert "document_text" not in s


def test_awaiting_answers_has_no_answer_keys():
    orch = make()
    res = orch.run_pipeline("x.pdf")
    assert res["status"] == "awaiting_answers" and res["failed_stage"] is None
    qs = orch.state["quiz_questions"]
    assert qs and all("correct_answer" not in q and "explanation" not in q for q in qs)


def test_stage_order_is_enforced():
    orch = make()
    assert orch.run_tutor()["status"] == "error"
    assert orch.generate_questions()["status"] == "error"
    assert orch.start_quiz()["status"] == "error"
    assert orch.submit_quiz({})["status"] == "error"
    assert orch.build_study_plan(EXAM)["status"] == "error"


def test_document_failure_stops_pipeline():
    orch = make(document_agent=FakeDoc(status="error"))
    res = orch.run_pipeline("missing.pdf")
    assert res["status"] == "error" and res["failed_stage"] == "document"
    assert "File not found" in res["error"] and orch.state["document"] is None


def test_scanned_pdf_gives_clear_error():
    res = make(document_agent=FakeDoc(text="   \n ")).process_document("scan.pdf")
    assert res["status"] == "error" and "scanned" in res["error"]


def test_agent_crash_is_caught():
    orch = make(tutor_agent=BrokenTutor())
    orch.process_document("x.pdf")
    res = orch.run_tutor()
    assert res["status"] == "error" and "groq exploded" in res["error"]


def test_long_text_is_truncated_with_warning():
    orch = make(document_agent=FakeDoc(text=("Paragraph about cells.\n\n" * 200)), max_text_chars=500)
    res = orch.process_document("big.pdf")
    assert res["status"] == "success" and res["document"]["truncated"]
    assert len(orch.state["document_text"]) <= 500 and orch.warnings


def test_no_usable_questions_is_an_error():
    orch = make(question_agent=FakeQuestions({"status": "success", "mcqs": [], "true_false": []}))
    orch.process_document("x.pdf")
    assert orch.generate_questions()["status"] == "error"


def test_retake_shows_improvement_and_plan_adapts():
    orch = ready_quiz()
    first = orch.submit_quiz(simulate_answers(orch.state["question_set"]))
    assert first["status"] == "success" and orch.state["previous_performance"] is None
    plan1 = orch.build_study_plan(EXAM, 2)["study_plan"]

    assert orch.start_quiz()["status"] == "success"                      # retake
    from quiz_agent import from_member3
    perfect = {q["id"]: q["correct_answer"] for q in from_member3(orch.state["question_set"])}
    second = orch.submit_quiz(perfect)
    assert second["quiz_result"]["percentage"] == 100.0
    assert orch.state["previous_performance"] is not None and orch.state["study_plan"] is None
    trends = [t["trend"]["direction"] for t in second["performance"]["topics"] if t["trend"]]
    assert "improved" in trends

    plan2 = orch.build_study_plan(EXAM, 2)["study_plan"]
    assert plan2["status"] == "success" and plan2 != plan1


def test_double_submit_is_cached():
    orch = ready_quiz()
    answers = simulate_answers(orch.state["question_set"])
    assert orch.submit_quiz(answers)["cached"] is False
    assert orch.submit_quiz(answers)["cached"] is True
    assert orch.state["previous_performance"] is None


def test_dead_llm_does_not_break_anything():
    orch = ready_quiz(llm=DeadLLM())
    res = orch.submit_quiz(simulate_answers(orch.state["question_set"]))
    assert res["status"] == "success" and res["performance"]["insight_source"] == "rules"
    assert orch.build_study_plan(EXAM)["status"] == "success"


def test_bad_exam_date_keeps_performance():
    orch = ready_quiz()
    orch.submit_quiz(simulate_answers(orch.state["question_set"]))
    res = orch.build_study_plan("20/10/2026")
    assert res["status"] == "error" and res["stage"] == "planner" and orch.state["performance"]


def test_written_answers_graded_by_llm():
    orch = ready_quiz()
    orch.start_quiz(include_written=True)
    qs = orch.state["quiz_questions"]
    assert any(q["id"] == "short_1" for q in qs)
    answers = simulate_answers(orch.state["question_set"], include_written=True)
    answers["short_1"] = "something about heredity"            # keyword matching would fail this
    res = orch.submit_quiz(answers)
    short = next(r for r in res["quiz_result"]["results"] if r["id"] == "short_1")
    assert short["is_correct"] and short["marks"] == 1.0


def test_new_document_resets_session():
    orch = ready_quiz()
    orch.submit_quiz(simulate_answers(orch.state["question_set"]))
    orch.process_document("another.pdf")
    assert orch.state["quiz_result"] is None and orch.state["question_set"] is None


def test_real_document_agent_with_real_pdf():
    try:
        import pypdf  # noqa: F401
        import document_agent  # noqa: F401
    except ImportError:
        print("  (skipped - pypdf / document_agent.py not available)")
        return
    pdf = tiny_pdf(["Mitochondria produce ATP for the cell.", "Ribosomes build proteins."])
    orch = StudyLensOrchestrator(llm=FakeLLM(), tutor_agent=FakeTutor(), question_agent=FakeQuestions())
    res = orch.process_pdf_bytes(pdf, "lecture.pdf")
    assert res["status"] == "success", res
    assert res["document"]["file_name"] == "lecture.pdf" and res["document"]["total_pages"] == 1
    assert "Mitochondria" in orch.state["document_text"]
    bad = orch.process_pdf_bytes(b"this is not a pdf", "broken.pdf")
    assert bad["status"] == "error" and orch.state["document"]["file_name"] == "lecture.pdf"


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
