"""
Headless UI test: runs app.py with Streamlit's AppTest, using Member 3's REAL
Tutor/Question agents but a fake LLM (no API key, no internet).

Run:  python test_app.py
Note: the PDF upload button cannot be clicked in AppTest, so the test starts from
an orchestrator that already processed a (fake) document; upload is covered by
test_orchestrator.py::test_real_document_agent_with_real_pdf.
"""

from streamlit.testing.v1 import AppTest

from orchestrator import StudyLensOrchestrator


class FakeDoc:
    def process_document(self, path):
        text = "Cells contain organelles.\n\n" * 30
        return {"status": "success", "file_name": "demo.pdf", "total_pages": 2,
                "total_characters": len(text), "total_chunks": 3, "full_text": text}


class FakeLLM:
    def generate_json(self, system_prompt=None, user_prompt=None, max_tokens=0, **kw):
        system = system_prompt or kw.get("system") or ""
        if "Tutor Agent" in system:
            return {"summary": "Cells have organelles.", "key_points": ["Mitochondria make ATP"],
                    "explanations": [{"concept": "ATP", "explanation": "Cell energy."}],
                    "difficult_terms": [{"term": "Organelle", "meaning": "A cell part."}]}
        if "Question Agent" in system:
            return {
                "mcqs": [
                    {"question": "Powerhouse of the cell?", "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi"],
                     "correct_answer": "Mitochondria", "explanation": "ATP.", "topic": "Cell Biology"},
                    {"question": "Which base pairs with A?", "options": ["G", "T", "C", "U"],
                     "correct_answer": "T", "explanation": "A-T.", "topic": "Genetics"},
                ],
                "true_false": [
                    {"statement": "DNA is single stranded.", "answer": False, "explanation": "Double helix.",
                     "topic": "Genetics"},
                ],
                "short_questions": [], "long_questions": [],
            }
        return {"insight": "Good effort - revise Genetics.", "message": "You can do it!"}


def button(at, text):
    return next(b for b in at.button if text in b.label)


def test_full_click_through():
    orch = StudyLensOrchestrator(llm=FakeLLM(), document_agent=FakeDoc())
    for step in (lambda: orch.process_document("demo.pdf"), orch.run_tutor,
                 orch.generate_questions, orch.start_quiz):
        assert step()["status"] == "success"

    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["orch"] = orch
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5 and len(at.radio) == 3            # the 3 quiz questions

    # answer: Q1 right, Q2 wrong, Q3 right
    for radio, choice in zip(at.radio, ["Mitochondria", "G", "False"]):
        radio.set_value(choice)
    button(at, "Submit quiz").click().run()
    assert not at.exception, at.exception
    assert orch.state["quiz_result"]["score"] == 2.0
    assert orch.state["performance"]["weak_topics"] == ["Genetics"] or \
        "Genetics" in orch.state["performance"]["needs_practice_topics"]

    # study plan
    button(at, "Generate my study plan").click().run()
    assert not at.exception, at.exception
    assert orch.state["study_plan"] and orch.state["study_plan"]["daily_plan"]

    # retake keeps the earlier performance as 'previous' and clears the old plan
    button(at, "Retake quiz").click().run()
    assert not at.exception, at.exception
    for radio, choice in zip(at.radio, ["Mitochondria", "T", "False"]):
        radio.set_value(choice)
    button(at, "Submit quiz").click().run()
    assert not at.exception, at.exception
    assert orch.state["quiz_result"]["percentage"] == 100.0
    assert orch.state["previous_performance"] is not None and orch.state["study_plan"] is None


def test_empty_app_renders():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS ", name)
            except Exception as exc:
                failed += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
