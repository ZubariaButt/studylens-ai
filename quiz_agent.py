"""
StudyLens AI - Quiz Agent (Member 4)

Flow:  Question Agent -> QuizAgent.prepare_quiz() -> student answers
       -> QuizAgent.evaluate() -> Performance Agent (Member 5) + Gradio UI (Member 6)

INTERFACE CONTRACT (share this with the team)
---------------------------------------------
Input from Question Agent: a list of question dicts

    {
      "id": "q1",                       # unique string
      "type": "mcq" | "true_false" | "short" | "long",
      "topic": "Cell Biology",          # needed for topic-wise performance
      "question": "....",
      "options": ["A", "B", "C", "D"],  # mcq only (true_false is auto True/False)
      "correct_answer": "B",            # mcq: option text or letter (A-D)
                                        # true_false: "True"/"False"
                                        # short/long: model answer text
      "keywords": ["mitochondria"],     # optional, helps grade short/long
      "explanation": "...."             # optional, used for feedback
    }

Input from student: {"q1": "B", "q2": "True", ...}   (question id -> answer)

Output of evaluate():

    {
      "status": "success" | "error",
      "error": None | "message",
      "score": 7.5, "total": 10, "percentage": 75.0,
      "results": [ {id, topic, type, question, student_answer, correct_answer,
                    is_correct, marks, feedback}, ... ],
      "topic_performance": {
          "Cell Biology": {"correct": 2, "total": 3, "marks": 2.0,
                           "max_marks": 3, "percentage": 66.7}
      }
    }

Optional: pass llm_grader(question_dict, student_answer) -> (marks 0..1, feedback)
to grade short/long answers with an LLM. Without it, keyword matching is used.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

VALID_TYPES = {"mcq", "true_false", "short", "long"}
PASS_MARKS = 0.5  # fraction of marks at/above which a short/long answer counts correct


def _norm(text) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def _error(message: str) -> dict:
    return {
        "status": "error", "error": message, "score": 0, "total": 0,
        "percentage": 0.0, "results": [], "topic_performance": {},
    }


class QuizAgent:
    def __init__(self, llm_grader: Optional[Callable] = None):
        self.llm_grader = llm_grader
        self.questions: list[dict] = []

    # ------------------------------------------------------------------ prepare
    def prepare_quiz(self, questions, include_written: bool = False) -> dict:
        """Validate questions from the Question Agent and store them.
        Accepts either Member 3's raw output dict (mcqs / true_false / ...)
        or an already-flat list. Returns a student-safe version (no correct
        answers) for the UI."""
        try:
            if isinstance(questions, dict):
                questions = from_member3(questions, include_written)
            if not isinstance(questions, list) or not questions:
                return _error("No questions received from Question Agent.")

            clean, seen = [], set()
            for i, q in enumerate(questions, start=1):
                if not isinstance(q, dict):
                    return _error(f"Question #{i} is not a dictionary.")
                qtype = str(q.get("type", "")).lower().replace("/", "_").replace(" ", "_")
                if qtype in ("tf", "truefalse"):
                    qtype = "true_false"
                if qtype not in VALID_TYPES:
                    return _error(f"Question #{i}: invalid type '{q.get('type')}'.")
                if not q.get("question") or q.get("correct_answer") in (None, ""):
                    return _error(f"Question #{i}: 'question' and 'correct_answer' are required.")
                qid = str(q.get("id", f"q{i}"))
                if qid in seen:
                    return _error(f"Duplicate question id '{qid}'.")
                seen.add(qid)

                item = dict(q)
                item.update({
                    "id": qid,
                    "type": qtype,
                    "topic": q.get("topic") or "General",
                    "options": list(q.get("options") or []),
                })
                if qtype == "true_false":
                    item["options"] = ["True", "False"]
                if qtype == "mcq" and len(item["options"]) < 2:
                    return _error(f"Question #{i}: MCQ needs at least 2 options.")
                clean.append(item)

            self.questions = clean
            student_view = [
                {k: q[k] for k in ("id", "type", "topic", "question", "options")}
                for q in clean
            ]
            return {"status": "success", "error": None,
                    "total_questions": len(clean), "questions": student_view}
        except Exception as exc:  # never crash the orchestrator
            return _error(f"prepare_quiz failed: {exc}")

    # ----------------------------------------------------------------- evaluate
    def evaluate(self, student_answers: dict) -> dict:
        try:
            if not self.questions:
                return _error("Quiz not prepared. Call prepare_quiz() first.")
            if not isinstance(student_answers, dict):
                return _error("student_answers must be a dict of {question_id: answer}.")

            results, topics = [], {}
            score = 0.0

            for q in self.questions:
                answer = student_answers.get(q["id"])
                marks, feedback = self._grade(q, answer)
                is_correct = marks >= (PASS_MARKS if q["type"] in ("short", "long") else 1.0)
                score += marks

                results.append({
                    "id": q["id"], "topic": q["topic"], "type": q["type"],
                    "question": q["question"],
                    "student_answer": answer if answer not in (None, "") else "(not answered)",
                    "correct_answer": q["correct_answer"],
                    "is_correct": is_correct, "marks": round(marks, 2),
                    "feedback": feedback,
                })

                t = topics.setdefault(q["topic"], {"correct": 0, "total": 0, "marks": 0.0, "max_marks": 0})
                t["total"] += 1
                t["max_marks"] += 1
                t["marks"] += marks
                t["correct"] += int(is_correct)

            for t in topics.values():
                t["marks"] = round(t["marks"], 2)
                t["percentage"] = round(100 * t["marks"] / t["max_marks"], 1)

            total = len(self.questions)
            return {
                "status": "success", "error": None,
                "score": round(score, 2), "total": total,
                "percentage": round(100 * score / total, 1),
                "results": results, "topic_performance": topics,
            }
        except Exception as exc:
            return _error(f"evaluate failed: {exc}")

    # ------------------------------------------------------------------ grading
    def _grade(self, q: dict, answer) -> tuple[float, str]:
        if answer is None or str(answer).strip() == "":
            return 0.0, f"Not answered. {self._expl(q)}"

        if q["type"] == "mcq":
            return self._grade_mcq(q, answer)
        if q["type"] == "true_false":
            ok = _norm(answer) == _norm(q["correct_answer"]) or (
                _norm(answer) in ("t", "f") and _norm(answer)[0] == _norm(q["correct_answer"])[0]
            )
            return (1.0, "Correct!") if ok else (0.0, f"Incorrect. The answer is {q['correct_answer']}. {self._expl(q)}")
        return self._grade_text(q, answer)

    def _resolve_mcq(self, q: dict, value) -> str:
        """Turn 'B' / 'b)' / full option text into normalized option text."""
        v = str(value).strip()
        m = re.fullmatch(r"\(?([A-Za-z])[\).]?", v)
        if m:
            idx = ord(m.group(1).upper()) - 65
            if 0 <= idx < len(q["options"]):
                return _norm(q["options"][idx])
        return _norm(v)

    def _grade_mcq(self, q: dict, answer) -> tuple[float, str]:
        if self._resolve_mcq(q, answer) == self._resolve_mcq(q, q["correct_answer"]):
            return 1.0, "Correct!"
        return 0.0, f"Incorrect. Correct answer: {q['correct_answer']}. {self._expl(q)}"

    def _grade_text(self, q: dict, answer) -> tuple[float, str]:
        if self.llm_grader:
            try:
                marks, fb = self.llm_grader(q, str(answer))
                return max(0.0, min(1.0, float(marks))), fb
            except Exception:
                pass  # fall back to keyword grading

        keywords = [k for k in q.get("keywords", []) if k] or self._auto_keywords(q["correct_answer"])
        if not keywords:
            return 0.0, "Could not grade automatically."
        ans = _norm(answer)
        hit = [k for k in keywords if _norm(k) in ans]
        marks = len(hit) / len(keywords)
        if marks >= 1.0:
            return 1.0, "Correct!"
        missing = [k for k in keywords if k not in hit]
        verdict = "Partially correct." if marks > 0 else "Incorrect."
        return marks, f"{verdict} Missing key points: {', '.join(missing)}. {self._expl(q)}"

    @staticmethod
    def _auto_keywords(model_answer: str, limit: int = 6) -> list[str]:
        stop = {"the", "a", "an", "is", "are", "of", "and", "to", "in", "it", "that", "for", "with", "as", "on", "by", "this", "which"}
        words = [w for w in _norm(model_answer).split() if len(w) > 3 and w not in stop]
        return list(dict.fromkeys(words))[:limit]

    @staticmethod
    def _expl(q: dict) -> str:
        return q.get("explanation", "") or ""


# ------------------------------------------------- Member 3 format adapter
def from_member3(result: dict, include_written: bool = False) -> list[dict]:
    """Convert Question Agent output into the flat list QuizAgent uses.

    Generated ids:  mcq_1.. , tf_1.. , short_1.. , long_1..
    (the UI must use these ids as keys in student_answers)
    """
    flat: list[dict] = []
    for i, q in enumerate(result.get("mcqs", []), 1):
        flat.append({
            "id": f"mcq_{i}", "type": "mcq", "topic": q.get("topic"),
            "question": q.get("question"), "options": q.get("options", []),
            "correct_answer": q.get("correct_answer"),
            "explanation": q.get("explanation", ""),
        })
    for i, q in enumerate(result.get("true_false", []), 1):
        ans = q.get("answer")
        if isinstance(ans, str):
            ans = ans.strip().lower() == "true"
        flat.append({
            "id": f"tf_{i}", "type": "true_false", "topic": q.get("topic"),
            "question": q.get("statement"),
            "correct_answer": "True" if ans else "False",
            "explanation": q.get("explanation", ""),
        })
    if include_written:
        for kind in ("short", "long"):
            for i, q in enumerate(result.get(f"{kind}_questions", []), 1):
                flat.append({
                    "id": f"{kind}_{i}", "type": kind, "topic": q.get("topic"),
                    "question": q.get("question"),
                    "correct_answer": q.get("answer"),
                })
    return flat


def make_groq_grader(llm) -> Callable:
    """Build an llm_grader from Member 3's GroqLLM (llm_client.py) to grade
    short/long answers by meaning instead of keywords."""
    system = (
        "You grade a student's answer against a model answer. Return ONLY JSON: "
        '{"score": number between 0 and 1, "feedback": "one or two sentences"}'
    )

    def grader(q: dict, student_answer: str):
        prompt = (
            f"Question: {q['question']}\nModel answer: {q['correct_answer']}\n"
            f"Student answer: {student_answer}"
        )
        out = llm.generate_json(system, prompt, max_tokens=300)
        return float(out.get("score", 0)), out.get("feedback", "")

    return grader


# ---------------------------------------------------------------- convenience
def run_quiz_agent(questions, student_answers: dict, llm_grader=None,
                   include_written: bool = False) -> dict:
    """One-call entry point for the Orchestrator (Member 1)."""
    agent = QuizAgent(llm_grader)
    prep = agent.prepare_quiz(questions, include_written)
    if prep["status"] == "error":
        return prep
    return agent.evaluate(student_answers)


if __name__ == "__main__":
    import json

    sample = [
        {"id": "q1", "type": "mcq", "topic": "Cell Biology", "question": "Powerhouse of the cell?",
         "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi"], "correct_answer": "Mitochondria",
         "explanation": "Mitochondria produce ATP."},
        {"id": "q2", "type": "true_false", "topic": "Cell Biology", "question": "Ribosomes make proteins.",
         "correct_answer": "True"},
        {"id": "q3", "type": "short", "topic": "Genetics", "question": "What is DNA?",
         "correct_answer": "Molecule carrying genetic information", "keywords": ["molecule", "genetic"]},
        {"id": "q4", "type": "mcq", "topic": "Genetics", "question": "DNA base pair with A?",
         "options": ["G", "T", "C", "U"], "correct_answer": "B"},
    ]
    answers = {"q1": "B", "q2": "False", "q3": "A molecule with genetic code", "q4": "T"}
    print(json.dumps(run_quiz_agent(sample, answers), indent=2))
