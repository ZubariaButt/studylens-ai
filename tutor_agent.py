from typing import Any, Dict

from llm_client import GroqLLM


class TutorAgent:
    """
    Member 3 — Tutor/Understanding Agent.

    Input:
        Processed study material from Member 2.

    Output:
        Summary, key points, easy explanations, and difficult terms.
    """

    def __init__(self, llm: GroqLLM):
        self.llm = llm

    def process(self, processed_content: str) -> Dict[str, Any]:
        if not processed_content or not processed_content.strip():
            return {
                "status": "error",
                "message": "No processed study content was provided."
            }

        system_prompt = """
You are the Tutor Agent of StudyLens AI.

Your job is to help a student understand ONLY the supplied study material.

Rules:
1. Do not introduce unrelated facts.
2. Do not invent information that is not supported by the material.
3. Keep explanations beginner-friendly and clear.
4. Preserve important technical terms.
5. Return ONLY valid JSON.
6. The JSON must contain exactly these top-level keys:
   summary, key_points, explanations, difficult_terms.

Required structure:
{
  "summary": "string",
  "key_points": ["string", "string"],
  "explanations": [
    {
      "concept": "string",
      "explanation": "string"
    }
  ],
  "difficult_terms": [
    {
      "term": "string",
      "meaning": "string"
    }
  ]
}
"""

        user_prompt = f"""
Analyze the following processed study material.

STUDY MATERIAL:
{processed_content}

Create:
- a concise but useful summary
- the most important key points
- easy explanations of difficult concepts
- meanings of difficult terms

Return the required JSON only.
"""

        try:
            result = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=5000,
            )
            return {
                "status": "success",
                **result
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Tutor Agent failed: {str(exc)}"
            }
