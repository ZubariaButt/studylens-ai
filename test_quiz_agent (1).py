"""Level 1 test for Quiz Agent using Member 3's real output format."""
from quiz_agent import QuizAgent, run_quiz_agent

member3_output = {
    "status": "success",
    "mcqs": [
        {"question": "Powerhouse of the cell?", "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi"],
         "correct_answer": "Mitochondria", "explanation": "Produces ATP.", "topic": "Cell Biology"},
        {"question": "Which base pairs with A?", "options": ["G", "T", "C", "U"],
         "correct_answer": "T", "explanation": "A-T pairing.", "topic": "Genetics"},
    ],
    "true_false": [
        {"statement": "Ribosomes make proteins.", "answer": True, "explanation": "Yes.", "topic": "Cell Biology"},
        {"statement": "DNA is single stranded.", "answer": False, "explanation": "Double helix.", "topic": "Genetics"},
    ],
    "short_questions": [{"question": "What is DNA?", "answer": "Molecule carrying genetic information", "topic": "Genetics"}],
    "long_questions": [],
}

# 1. UI-safe view has no answers
agent = QuizAgent()
prep = agent.prepare_quiz(member3_output)
assert prep["status"] == "success" and prep["total_questions"] == 4
assert all("correct_answer" not in q for q in prep["questions"])

# 2. Scoring + topic performance
res = agent.evaluate({"mcq_1": "Mitochondria", "mcq_2": "C", "tf_1": "True", "tf_2": "False"})
assert res["score"] == 3 and res["percentage"] == 75.0
assert res["topic_performance"]["Genetics"]["percentage"] == 50.0
assert res["topic_performance"]["Cell Biology"]["percentage"] == 100.0

# 3. Unanswered question counts as wrong
res = run_quiz_agent(member3_output, {"mcq_1": "Mitochondria"})
assert res["score"] == 1 and res["results"][1]["student_answer"] == "(not answered)"

# 4. Written questions optional
res = run_quiz_agent(member3_output, {"short_1": "a molecule with genetic information"}, include_written=True)
assert res["total"] == 5

# 5. Errors never crash
assert run_quiz_agent({}, {})["status"] == "error"
assert QuizAgent().evaluate({})["status"] == "error"
print("All Quiz Agent tests passed")
