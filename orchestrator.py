"""
StudyLens AI - Orchestrator (Member 1)

Connects every agent into one workflow and is the ONLY thing the UI should call:

    Upload PDF
      -> process_document()    Document Agent  (Member 2)
      -> run_tutor()           Tutor Agent     (Member 3)
      -> generate_questions()  Question Agent  (Member 3)   + cleaning / topic fixes
      -> start_quiz()          Quiz Agent      (Member 4)   student-safe questions
         ... student answers ...
      -> submit_quiz(answers)  Quiz Agent + Performance Agent (Members 4, 5)
      -> build_study_plan()    Study Planner Agent (Member 5)

Every method returns the same envelope and NEVER raises:

    {"status": "success" | "error", "stage": "...", "error": None | "message", ...payload}

All results are also kept in orchestrator.state, so the UI can re-draw any tab
after a Streamlit rerun. Keep ONE orchestrator per user session:

    if "orch" not in st.session_state:
        st.session_state.orch = StudyLensOrchestrator(api_key=st.secrets["GROQ_API_KEY"])
    orch = st.session_state.orch

    res = orch.process_pdf_bytes(uploaded_file.getvalue(), uploaded_file.name)
    if res["status"] == "error": st.error(res["error"])

    orch.run_tutor();  orch.generate_questions();  orch.start_quiz()
    quiz = orch.state["quiz_questions"]            # [{id, type, topic, question, options}]
    orch.submit_quiz({"mcq_1": "Mitochondria", "tf_1": "True"})
    orch.build_study_plan(exam_date, hours_per_day=2)

    orch.state["tutor"] / ["quiz_result"] / ["performance"] / ["study_plan"]

Retake: call start_quiz() again, then submit_quiz() - the previous performance is
used automatically so the Performance Agent can show improvement / decline, and
build_study_plan() should be called again to adapt the plan.

Group-testing without a UI: run_pipeline(pdf_path, answer_fn, exam_date).
Every agent can be injected (llm=, document_agent=, tutor_agent=, question_agent=)
so the whole workflow can be tested without an API key.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import date, timedelta
from typing import Callable, Optional

# Characters of study material sent to the LLM per call. Long PDFs are cut at a
# paragraph boundary (with a warning) to avoid context / rate-limit errors.
# Raise it if your Groq plan allows more.
MAX_TEXT_CHARS = 20000


# ------------------------------------------------------------------ envelopes
def _ok(stage: str, **payload) -> dict:
    return {"status": "success", "stage": stage, "error": None, **payload}


def _fail(stage: str, message, **extra) -> dict:
    return {"status": "error", "stage": stage, "error": str(message), **extra}


def _agent_error(result, default: str) -> str:
    if isinstance(result, dict):
        return str(result.get("message") or result.get("error") or default)
    return default


# ------------------------------------------------------- question cleaning
def _text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def _as_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "t", "yes"):
            return True
        if v in ("false", "f", "no"):
            return False
    return None


def _match_option(options: list[str], answer) -> Optional[str]:
    """Return the option text that `answer` refers to (text, letter, or 'A) text')."""
    if answer is None:
        return None
    a = _text(answer)
    if not a:
        return None
    for opt in options:
        if opt.casefold() == a.casefold():
            return opt
    m = re.fullmatch(r"\(?([A-Za-z])[\).:]?", a)
    if m:
        idx = ord(m.group(1).upper()) - 65
        if 0 <= idx < len(options):
            return options[idx]
    for opt in options:                      # options written like "A) Mitochondria"
        if re.sub(r"^\(?[A-Da-d][\).:]\s+", "", opt).casefold() == a.casefold():
            return opt
    return None


class _TopicNames:
    """Makes 'genetics', 'Genetics ' and 'GENETICS' the same topic for Member 5."""

    def __init__(self):
        self.seen: dict[str, str] = {}

    def canon(self, topic) -> str:
        name = _text(topic).rstrip(".:") or "General"
        if name == name.lower():
            name = name.title()
        return self.seen.setdefault(name.casefold(), name)


def clean_question_set(raw: dict) -> tuple[dict, list[str]]:
    """Validate the Question Agent's output before the Quiz Agent sees it.

    * MCQ correct_answer is mapped to the exact option text (letters are accepted);
      MCQs whose answer is not among the options are dropped, because the
      student could never get them right.
    * True/False answers become real booleans.
    * Topic names are unified (case / spacing), missing topics become 'General'.
    Returns (clean_question_set, warnings).
    """
    warnings: list[str] = []
    topics = _TopicNames()
    clean: dict = {"mcqs": [], "true_false": [], "short_questions": [], "long_questions": []}

    for i, q in enumerate(raw.get("mcqs") or [], 1):
        if not isinstance(q, dict):
            warnings.append(f"Dropped MCQ #{i}: not a valid object.")
            continue
        question = _text(q.get("question"))
        opts = q.get("options") if isinstance(q.get("options"), list) else []
        options = [_text(o) for o in opts if _text(o)]
        if not question or len(options) < 2:
            warnings.append(f"Dropped MCQ #{i}: missing question or options.")
            continue
        answer = _match_option(options, q.get("correct_answer"))
        if answer is None:
            warnings.append(f"Dropped MCQ #{i}: the correct answer is not one of its options.")
            continue
        item = dict(q)
        item.update(question=question, options=options, correct_answer=answer,
                    topic=topics.canon(q.get("topic")))
        clean["mcqs"].append(item)

    for i, q in enumerate(raw.get("true_false") or [], 1):
        statement = _text(q.get("statement")) if isinstance(q, dict) else ""
        answer = _as_bool(q.get("answer")) if isinstance(q, dict) else None
        if not statement or answer is None:
            warnings.append(f"Dropped True/False #{i}: missing statement or answer.")
            continue
        item = dict(q)
        item.update(statement=statement, answer=answer, topic=topics.canon(q.get("topic")))
        clean["true_false"].append(item)

    for key in ("short_questions", "long_questions"):
        for i, q in enumerate(raw.get(key) or [], 1):
            question = _text(q.get("question")) if isinstance(q, dict) else ""
            answer = _text(q.get("answer")) if isinstance(q, dict) else ""
            if not question or not answer:
                warnings.append(f"Dropped {key.replace('_', ' ')} #{i}: missing question or answer.")
                continue
            item = dict(q)
            item.update(question=question, answer=answer, topic=topics.canon(q.get("topic")))
            clean[key].append(item)

    return clean, warnings


def simulate_answers(question_set: dict, include_written: bool = False) -> dict:
    """TEST HELPER: answers every question correctly except each third one.
    Returns {question_id: answer} using the Quiz Agent's generated ids."""
    from quiz_agent import from_member3

    answers: dict = {}
    for i, q in enumerate(from_member3(question_set, include_written)):
        correct = q["correct_answer"]
        if q["type"] in ("short", "long") or i % 3 != 2:
            answers[q["id"]] = correct
        elif q["type"] == "true_false":
            answers[q["id"]] = "False" if correct == "True" else "True"
        else:
            wrong = [o for o in q["options"] if o != correct]
            answers[q["id"]] = wrong[0] if wrong else None
    return answers


def _truncate(text: str, limit: int) -> str:
    cut = text[:limit]
    idx = cut.rfind("\n\n")
    return cut[:idx] if idx > limit * 0.7 else cut


# --------------------------------------------------------------- orchestrator
class StudyLensOrchestrator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.2, max_text_chars: int = MAX_TEXT_CHARS,
                 chunk_size: int = 1000, chunk_overlap: int = 150,
                 use_llm_coach: bool = True, llm=None, document_agent=None,
                 tutor_agent=None, question_agent=None):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_text_chars = max_text_chars
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_llm_coach = use_llm_coach
        # injectable for tests; otherwise created lazily on first use
        self._llm = llm
        self._document_agent = document_agent
        self._tutor_agent = tutor_agent
        self._question_agent = question_agent
        self._quiz_agent = None
        self._last_answers: Optional[dict] = None
        self.log: list[dict] = []
        self.warnings: list[str] = []
        self.state: dict = self._empty_state()

    # ------------------------------------------------------------- plumbing
    @staticmethod
    def _empty_state() -> dict:
        return {
            "document": None, "document_text": None, "tutor": None,
            "question_set": None, "quiz_questions": None, "quiz_result": None,
            "performance": None, "previous_performance": None, "study_plan": None,
        }

    def reset(self) -> None:
        """Forget everything (new document / new student). Agents and config are kept."""
        self.state = self._empty_state()
        self._quiz_agent = None
        self._last_answers = None
        self.log, self.warnings = [], []

    def _reset_quiz_state(self) -> None:
        for key in ("quiz_questions", "quiz_result", "performance",
                    "previous_performance", "study_plan"):
            self.state[key] = None
        self._quiz_agent = None
        self._last_answers = None

    def _guard(self, stage: str, work: Callable[[], dict]) -> dict:
        """Run one stage: time it, log it, collect warnings, turn crashes into errors."""
        start = time.perf_counter()
        try:
            out = work()
        except Exception as exc:                                   # never crash the UI
            out = _fail(stage, f"{type(exc).__name__}: {exc}")
        self.log.append({"stage": out.get("stage", stage), "status": out["status"],
                         "seconds": round(time.perf_counter() - start, 2),
                         "error": out["error"]})
        for w in out.get("warnings") or []:
            if w not in self.warnings:
                self.warnings.append(w)
        return out

    def _get_llm(self):
        if self._llm is None:
            from llm_client import GroqLLM          # Member 3's shared Groq client
            self._llm = GroqLLM(api_key=self.api_key, model=self.model,
                                temperature=self.temperature)
        return self._llm

    def _coach_llm(self):
        """Optional LLM used only to phrase insight / coach text. Never required."""
        if not self.use_llm_coach:
            return None
        try:
            return self._get_llm()
        except Exception:
            return None

    # ------------------------------------------- stage 1: Document Agent (M2)
    def process_document(self, pdf_path: str) -> dict:
        def work():
            if self._document_agent is None:
                from document_agent import DocumentProcessingAgent
                self._document_agent = DocumentProcessingAgent(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            result = self._document_agent.process_document(pdf_path)
            if not isinstance(result, dict) or result.get("status") != "success":
                return _fail("document", _agent_error(result, "Document Agent failed."))
            text = (result.get("full_text") or "").strip()
            if not text:
                return _fail("document", "No readable text was found in this PDF "
                                         "(it may be scanned images).")
            warnings, truncated = [], False
            if len(text) > self.max_text_chars:
                text = _truncate(text, self.max_text_chars)
                truncated = True
                warnings.append(
                    f"The PDF is long, so only the first {len(text):,} characters are used "
                    f"for summaries and questions.")
            self.reset()                      # new document -> start a clean session
            doc = {
                "file_name": result.get("file_name") or os.path.basename(str(pdf_path)),
                "total_pages": result.get("total_pages", 0),
                "total_characters": result.get("total_characters", len(text)),
                "total_chunks": result.get("total_chunks", 0),
                "characters_used": len(text), "truncated": truncated,
            }
            self.state["document"], self.state["document_text"] = doc, text
            return _ok("document", document=doc, warnings=warnings)
        return self._guard("document", work)

    def process_pdf_bytes(self, data: bytes, filename: str = "upload.pdf") -> dict:
        """For Streamlit: pass uploaded_file.getvalue() and uploaded_file.name."""
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                path = tmp.name
            result = self.process_document(path)
            if result["status"] == "success":
                self.state["document"]["file_name"] = filename
                result["document"]["file_name"] = filename
            return result
        except Exception as exc:
            return _fail("document", f"{type(exc).__name__}: {exc}")
        finally:
            if path and os.path.exists(path):
                os.remove(path)

    # ------------------------------------------ stage 2: Tutor Agent (M3)
    def run_tutor(self) -> dict:
        def work():
            text = self.state["document_text"]
            if not text:
                return _fail("tutor", "No document yet. Call process_document() first.")
            if self._tutor_agent is None:
                from tutor_agent import TutorAgent
                self._tutor_agent = TutorAgent(self._get_llm())
            result = self._tutor_agent.process(text)
            if not isinstance(result, dict) or result.get("status") != "success":
                return _fail("tutor", _agent_error(result, "Tutor Agent failed."))
            self.state["tutor"] = result
            return _ok("tutor", tutor=result)
        return self._guard("tutor", work)

    # ---------------------------------------- stage 3: Question Agent (M3)
    def generate_questions(self) -> dict:
        def work():
            text = self.state["document_text"]
            if not text:
                return _fail("questions", "No document yet. Call process_document() first.")
            if self._question_agent is None:
                from question_agent import QuestionAgent
                self._question_agent = QuestionAgent(self._get_llm())
            raw = self._question_agent.generate(text)
            if not isinstance(raw, dict) or raw.get("status") != "success":
                return _fail("questions", _agent_error(raw, "Question Agent failed."))
            clean, warnings = clean_question_set(raw)
            if not clean["mcqs"] and not clean["true_false"]:
                return _fail("questions", "The Question Agent produced no usable MCQ or "
                                          "True/False questions. Please try again.",
                             warnings=warnings)
            clean["status"] = "success"
            self._reset_quiz_state()          # new questions invalidate old quiz results
            self.state["question_set"] = clean
            counts = {k: len(clean[k]) for k in
                      ("mcqs", "true_false", "short_questions", "long_questions")}
            return _ok("questions", questions=clean, counts=counts, warnings=warnings)
        return self._guard("questions", work)

    # ------------------------------------------- stage 4: Quiz Agent (M4)
    def start_quiz(self, include_written: bool = False) -> dict:
        """Prepare (or restart) the quiz. Returns student-safe questions (no answers).
        Calling it again after a submission starts a retake."""
        def work():
            qs = self.state["question_set"]
            if not qs:
                return _fail("quiz_prepare", "No questions yet. Call generate_questions() first.")
            from quiz_agent import QuizAgent
            warnings, grader = [], None
            if include_written and (qs["short_questions"] or qs["long_questions"]):
                try:
                    from quiz_agent import make_groq_grader
                    grader = make_groq_grader(self._get_llm())
                except Exception as exc:
                    warnings.append(f"AI grading is unavailable ({exc}); written answers "
                                    f"will use keyword matching.")
            agent = QuizAgent(grader)
            prep = agent.prepare_quiz(qs, include_written)
            if prep["status"] != "success":
                return _fail("quiz_prepare", prep["error"], warnings=warnings)
            self._quiz_agent = agent
            self._last_answers = None
            self.state["quiz_questions"] = prep["questions"]
            return _ok("quiz_prepare", quiz_questions=prep["questions"],
                       total_questions=prep["total_questions"], warnings=warnings)
        return self._guard("quiz_prepare", work)

    def submit_quiz(self, student_answers: dict) -> dict:
        """Score the quiz (Member 4) and analyse it (Member 5). Keys of
        student_answers are the ids in state['quiz_questions'] (mcq_1, tf_1, ...)."""
        def work():
            if self._quiz_agent is None:
                return _fail("quiz_evaluate", "No active quiz. Call start_quiz() first.")
            if not isinstance(student_answers, dict):
                return _fail("quiz_evaluate", "student_answers must be a dict {question_id: answer}.")
            if (self._last_answers == student_answers and self.state["quiz_result"]
                    and self.state["performance"]):          # double-click / rerun
                return _ok("quiz_evaluate", quiz_result=self.state["quiz_result"],
                           performance=self.state["performance"], cached=True)

            result = self._quiz_agent.evaluate(student_answers)
            if result.get("status") != "success":
                return _fail("quiz_evaluate", _agent_error(result, "Quiz Agent failed."))

            from performance_agent import analyze_performance
            previous = self.state["performance"]            # earlier attempt, if any
            perf = analyze_performance(result, previous, self._coach_llm())
            if perf.get("status") != "success":
                return _fail("performance", _agent_error(perf, "Performance Agent failed."),
                             quiz_result=result)

            self.state.update(quiz_result=result, performance=perf,
                              previous_performance=previous, study_plan=None)
            self._last_answers = dict(student_answers)
            return _ok("quiz_evaluate", quiz_result=result, performance=perf, cached=False)
        return self._guard("quiz_evaluate", work)

    # ------------------------------------- stage 5: Study Planner Agent (M5)
    def build_study_plan(self, exam_date, hours_per_day: float = 2.0, start_date=None) -> dict:
        """exam_date: 'YYYY-MM-DD' or datetime.date. Call again after a retake to adapt."""
        def work():
            perf = self.state["performance"]
            if not perf:
                return _fail("planner", "Submit the quiz first so the planner knows your performance.")
            from study_planner_agent import generate_study_plan
            plan = generate_study_plan(perf, exam_date, hours_per_day, start_date, self._coach_llm())
            if plan.get("status") != "success":
                return _fail("planner", _agent_error(plan, "Study Planner Agent failed."),
                             performance=perf)
            self.state["study_plan"] = plan
            return _ok("planner", study_plan=plan, warnings=plan.get("warnings") or [])
        return self._guard("planner", work)

    # --------------------------------------------------------- full pipeline
    def run_pipeline(self, pdf_path: str, answer_fn: Optional[Callable[[list], dict]] = None,
                     exam_date=None, hours_per_day: float = 2.0,
                     include_written: bool = False) -> dict:
        """Run the whole student journey (for Level 3 testing / demos without a UI).

        answer_fn(quiz_questions) -> {question_id: answer} plays the student.
        Without answer_fn the pipeline stops after the quiz is ready and returns
        status 'awaiting_answers'. Stops at the first failing stage.
        """
        steps = [
            lambda: self.process_document(pdf_path),
            self.run_tutor,
            self.generate_questions,
            lambda: self.start_quiz(include_written),
        ]
        for step in steps:
            res = step()
            if res["status"] != "success":
                return self._pipeline_result("error", res)
        if answer_fn is None:
            return self._pipeline_result("awaiting_answers")
        try:
            answers = answer_fn(self.state["quiz_questions"])
        except Exception as exc:
            return self._pipeline_result("error", _fail("answers", f"answer_fn failed: {exc}"))
        res = self.submit_quiz(answers)
        if res["status"] != "success":
            return self._pipeline_result("error", res)
        res = self.build_study_plan(exam_date, hours_per_day)
        if res["status"] != "success":
            return self._pipeline_result("error", res)
        return self._pipeline_result("success")

    def _pipeline_result(self, status: str, failed: Optional[dict] = None) -> dict:
        return {
            "status": status,
            "failed_stage": failed["stage"] if failed else None,
            "error": failed["error"] if failed else None,
            "state": self.export_state(), "log": self.log, "warnings": self.warnings,
        }

    def export_state(self) -> dict:
        """JSON-serialisable snapshot (without the raw document text)."""
        out = {k: v for k, v in self.state.items() if k != "document_text"}
        out["warnings"], out["log"] = list(self.warnings), list(self.log)
        return out


# ------------------------------------------------------------------- CLI
def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the StudyLens pipeline on a PDF.")
    ap.add_argument("pdf")
    ap.add_argument("--exam", default=(date.today() + timedelta(days=14)).isoformat(),
                    help="exam date YYYY-MM-DD (default: 14 days from today)")
    ap.add_argument("--hours", type=float, default=2.0, help="study hours per day")
    ap.add_argument("--auto", action="store_true",
                    help="simulate a student (every third answer wrong) and build the plan")
    ap.add_argument("--out", default="orchestrator_output.json")
    args = ap.parse_args()

    orch = StudyLensOrchestrator()
    answer_fn = (lambda _qs: simulate_answers(orch.state["question_set"])) if args.auto else None
    result = orch.run_pipeline(args.pdf, answer_fn, args.exam, args.hours)

    for entry in result["log"]:
        mark = "OK  " if entry["status"] == "success" else "FAIL"
        print(f"{mark} {entry['stage']:<14} {entry['seconds']:>6}s  {entry['error'] or ''}")
    for w in result["warnings"]:
        print("WARN", w)
    print("Pipeline:", result["status"], result["error"] or "")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Saved", args.out)


if __name__ == "__main__":
    _main()
