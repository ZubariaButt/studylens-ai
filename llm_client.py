import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqLLM:
    """
    Shared Groq client used by the Tutor Agent and Question Agent.

    One Groq API key can be shared by multiple specialized agents.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to Streamlit secrets "
                "or your .env file."
            )

        self.client = Groq(api_key=self.api_key)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 5000,
    ) -> Dict[str, Any]:
        """Call Groq and return validated JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Groq returned invalid JSON: {content[:500]}"
            ) from exc
