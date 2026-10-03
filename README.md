# StudyLens AI — Multi-Agent Study Assistant

Upload a PDF → lesson → quiz → performance analysis → personalised study plan.
The **Orchestrator** (`orchestrator.py`) runs the agents; `app.py` (Streamlit) only talks to it.

| File | Owner | Role |
|---|---|---|
| `orchestrator.py` | Member 1 | Runs the whole workflow, validates data between agents |
| `app.py` | Member 1 / 6 | Streamlit UI (5 tabs) |
| `document_agent.py` | Member 2 | PDF → clean text + chunks |
| `llm_client.py`, `tutor_agent.py`, `question_agent.py` | Member 3 | Lesson + questions (Groq) |
| `quiz_agent.py` | Member 4 | Quiz scoring + topic-wise results |
| `performance_agent.py`, `study_planner_agent.py`, `study_coach_agent.py` | Member 5 | Weak/strong topics + adaptive plan |

## Flow
```
PDF → Document Agent → Tutor Agent
                     → Question Agent → (cleaning) → Quiz Agent → Performance Agent → Study Planner
```

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # then put your real GROQ_API_KEY in .env
streamlit run app.py
```

## Tests (no API key needed)
```bash
python test_quiz_agent.py
python test_study_coach.py
python test_orchestrator.py
python test_app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to GitHub (never commit `.env`).
2. share.streamlit.io → New app → main file `app.py`.
3. App settings → Secrets:
```toml
GROQ_API_KEY = "your_real_groq_key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## Notes
- Long PDFs: only the first 20,000 characters go to the LLM (`MAX_TEXT_CHARS` in `orchestrator.py`).
- Scanned (image-only) PDFs have no extractable text and show a clear error.
