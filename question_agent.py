from typing import Any, Dict

from llm_client import GroqLLM


class QuestionAgent:
    """
    Member 3 — Question Agent.

    Generates questions strictly from the supplied study material.
    """

    def __init__(self, llm: GroqLLM):
        self.llm = llm

    def generate(self, processed_content: str) -> Dict[str, Any]:
        if not processed_content or not processed_content.strip():
            return {
                "status": "error",
                "message": "No processed study content was provided."
            }

        system_prompt = """
You are the Question Agent of StudyLens AI.

Generate educational questions ONLY from the supplied study material.

Rules:
1. Do not create random/general-knowledge questions.
2. Every question must be answerable from the supplied material.
3. Avoid duplicate questions.
4. Make questions clear and suitable for students.
5. For MCQs, provide exactly 4 options.
6. MCQ correct_answer must be one of the four options.
7. Include a short explanation for each MCQ answer.
8. Include answers for short, long, and true/false questions.
9. Return ONLY valid JSON.

Required JSON structure:
{
  "mcqs": [
    {
      "question": "string",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "string",
      "explanation": "string",
      "topic": "string"
    }
  ],
  "short_questions": [
    {
      "question": "string",
      "answer": "string",
      "topic": "string"
    }
  ],
  "long_questions": [
    {
      "question": "string",
      "answer": "string",
      "topic": "string"
    }
  ],
  "true_false": [
    {
      "statement": "string",
      "answer": true,
      "explanation": "string",
      "topic": "string"
    }
  ]
}
"""

        user_prompt = f"""
Generate a balanced question set from this study material.

STUDY MATERIAL:
{processed_content}

Generate approximately:
- 5 MCQs
- 3 short questions
- 2 long questions
- 5 True/False questions

Make sure all questions are directly supported by the material.
Return the required JSON only.
"""

        try:
            result = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=7000,
            )
            return {
                "status": "success",
                **result
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Question Agent failed: {str(exc)}"
            }
