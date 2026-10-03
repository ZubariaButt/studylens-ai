"""
StudyLens AI - final Streamlit app (Member 1: integration)

Every button talks to the Orchestrator only. The orchestrator runs the agents:
Document (M2) -> Tutor + Question (M3) -> Quiz (M4) -> Performance + Planner (M5).

Run:  streamlit run app.py
"""

import json
import os
from datetime import date, timedelta

import streamlit as st

from orchestrator import StudyLensOrchestrator
from study_planner_agent import minutes_with_hours

ICON = {"weak": "🔴", "needs_practice": "🟡", "strong": "🟢"}
LABEL = {"weak": "Weak", "needs_practice": "Needs practice", "strong": "Strong"}

st.set_page_config(page_title="StudyLens AI", page_icon="🎓", layout="wide")


# ------------------------------------------------------------------ helpers
def read_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:                     # no secrets file on a local machine
        key = None
    return key or os.getenv("GROQ_API_KEY")


API_KEY = read_api_key()


def get_orch() -> StudyLensOrchestrator:
    if "orch" not in st.session_state:
        st.session_state.orch = StudyLensOrchestrator(api_key=API_KEY)
    return st.session_state.orch


def clear_answer_keys():
    for key in [k for k in st.session_state if k.startswith("ans_")]:
        del st.session_state[key]


def restart_quiz():
    """Start a fresh attempt (also used when the 'written questions' box is toggled)."""
    clear_answer_keys()
    orch = get_orch()
    if orch.state["question_set"]:
        res = orch.start_quiz(include_written=st.session_state.get("include_written", False))
        st.session_state.quiz_error = res["error"] if res["status"] == "error" else None


def show_error(res: dict):
    st.error(f"**{res['stage']}** failed: {res['error']}")


orch = get_orch()
state = orch.state

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 StudyLens AI")
    st.caption("Multi-agent learning assistant")
    st.divider()
    st.markdown("### Progress")
    for name, done in [
        ("Document processed", state["document"]),
        ("Lesson ready", state["tutor"]),
        ("Questions generated", state["question_set"]),
        ("Quiz taken", state["quiz_result"]),
        ("Study plan created", state["study_plan"]),
    ]:
        st.write(("✅ " if done else "⬜ ") + name)
    st.divider()
    st.markdown("### Document settings")
    chunk_size = st.slider("Chunk size", 300, 2500, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 50, 500, 150, 25)
    if st.button("🗑️ Start over"):
        orch.reset()
        clear_answer_keys()
        st.rerun()

st.title("🎓 StudyLens AI")
st.write("Upload your study material to get a lesson, a quiz, performance insights "
         "and a personalised study plan.")
if not API_KEY:
    st.warning("GROQ_API_KEY is not set. Add it to `.env` (local) or Streamlit Secrets "
               "(cloud) before processing a document.")

tab_upload, tab_lesson, tab_quiz, tab_perf, tab_plan = st.tabs([
    "📁 1. Upload", "👨‍🏫 2. Lesson", "📝 3. Quiz", "📊 4. Performance", "📅 5. Study plan",
])

# ------------------------------------------------------------------ 1. upload
with tab_upload:
    uploaded = st.file_uploader("Upload a PDF (lecture, notes or textbook)", type=["pdf"])
    if uploaded is not None:
        st.info(f"**{uploaded.name}** - {uploaded.size / 1024:.1f} KB")
        if st.button("🚀 Process document", type="primary"):
            if not API_KEY:
                st.error("GROQ_API_KEY is missing.")
            elif chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than the chunk size.")
            else:
                clear_answer_keys()
                orch = StudyLensOrchestrator(api_key=API_KEY, chunk_size=chunk_size,
                                             chunk_overlap=chunk_overlap)
                st.session_state.orch = orch
                steps = [
                    ("📄 Document Agent is reading the PDF...",
                     lambda: orch.process_pdf_bytes(uploaded.getvalue(), uploaded.name)),
                    ("👨‍🏫 Tutor Agent is preparing your lesson...", orch.run_tutor),
                    ("❓ Question Agent is writing questions...", orch.generate_questions),
                    ("📝 Quiz Agent is preparing the quiz...",
                     lambda: orch.start_quiz(st.session_state.get("include_written", False))),
                ]
                with st.status("Agents are working...", expanded=True) as status:
                    failed = None
                    for label, call in steps:
                        st.write(label)
                        res = call()
                        if res["status"] != "success":
                            failed = res
                            break
                    if failed:
                        status.update(label=f"Stopped at: {failed['stage']}", state="error")
                    else:
                        status.update(label="All agents finished", state="complete", expanded=False)
                if failed:
                    show_error(failed)
                else:
                    st.success("Done! Open the Lesson and Quiz tabs.")

    doc = get_orch().state["document"]
    if doc:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", doc["total_pages"])
        c2.metric("Characters", f"{doc['total_characters']:,}")
        c3.metric("Chunks", doc["total_chunks"])
        for w in get_orch().warnings:
            st.warning(w)

# ------------------------------------------------------------------ 2. lesson
with tab_lesson:
    tutor = get_orch().state["tutor"]
    if not tutor:
        st.info("Process a document in the Upload tab to see your lesson.")
    else:
        st.subheader("📌 Summary")
        st.write(tutor.get("summary", ""))
        st.subheader("🔑 Key points")
        for point in tutor.get("key_points", []):
            st.markdown(f"- {point}")
        st.subheader("💡 Easy explanations")
        for item in tutor.get("explanations", []):
            with st.expander(item.get("concept", "Concept")):
                st.write(item.get("explanation", ""))
        st.subheader("📖 Difficult terms")
        for item in tutor.get("difficult_terms", []):
            st.markdown(f"**{item.get('term', '')}** - {item.get('meaning', '')}")

# -------------------------------------------------------------------- 3. quiz
with tab_quiz:
    o = get_orch()
    quiz = o.state["quiz_questions"]
    if not quiz:
        st.info("Process a document first, then your quiz will appear here.")
        if st.session_state.get("quiz_error"):
            st.error(st.session_state.quiz_error)
    else:
        st.checkbox("Also include short / long written questions",
                    key="include_written", on_change=restart_quiz,
                    help="Written answers are graded by AI (or by keywords if AI is unavailable).")
        if st.session_state.get("quiz_error"):
            st.error(st.session_state.quiz_error)

        with st.form("quiz_form"):
            for i, q in enumerate(quiz, 1):
                st.markdown(f"**Q{i}. {q['question']}**  \n:gray[Topic: {q['topic']}]")
                if q["type"] in ("short", "long"):
                    st.text_area("Your answer", key=f"ans_{q['id']}", label_visibility="collapsed")
                else:
                    st.radio("Your answer", q["options"], index=None, key=f"ans_{q['id']}",
                             label_visibility="collapsed")
            submitted = st.form_submit_button("✅ Submit quiz", type="primary")

        if submitted:
            answers = {q["id"]: (st.session_state.get(f"ans_{q['id']}") or None) for q in quiz}
            res = o.submit_quiz(answers)
            if res["status"] != "success":
                show_error(res)
            else:
                st.success("Quiz scored! See the Performance and Study plan tabs.")

        result = o.state["quiz_result"]
        if result:
            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Score", f"{result['score']:g} / {result['total']}")
            m2.metric("Percentage", f"{result['percentage']:g}%")
            for i, r in enumerate(result["results"], 1):
                box = st.success if r["is_correct"] else st.error
                box(f"**Q{i}. {r['question']}**\n\nYour answer: {r['student_answer']}\n\n{r['feedback']}")
            st.button("🔄 Retake quiz", on_click=restart_quiz)

# ------------------------------------------------------------- 4. performance
with tab_perf:
    perf = get_orch().state["performance"]
    if not perf:
        st.info("Submit the quiz to see your performance.")
    else:
        overall = perf["overall"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{overall['score']:g} / {overall['total']}")
        c2.metric("Overall", f"{overall['percentage']:g}%",
                  delta=(f"{overall['trend']['change']:+g} pts" if overall.get("trend") else None))
        c3.metric("Level", LABEL[overall["level"]])
        st.info(perf["insight"])

        left, right = st.columns([3, 2])
        with left:
            st.bar_chart({t["topic"]: t["percentage"] for t in perf["topics"]}, y_label="Score %")
        with right:
            for t in perf["topics"]:
                trend = (f" ({t['trend']['direction']}, {t['trend']['change']:+g})"
                         if t.get("trend") else "")
                st.write(f"{ICON[t['status']]} **{t['topic']}** - {t['percentage']:g}%{trend}")

        with st.expander("Questions to review"):
            any_wrong = False
            for t in perf["topics"]:
                for q in t["wrong_questions"]:
                    any_wrong = True
                    st.write(f"**{t['topic']}:** {q['question']}  \nCorrect answer: {q['correct_answer']}")
            if not any_wrong:
                st.write("Nothing to review - all answers were correct. 🎉")

# ------------------------------------------------------------- 5. study plan
with tab_plan:
    o = get_orch()
    if not o.state["performance"]:
        st.info("Submit the quiz first so the planner knows your weak and strong topics.")
    else:
        c1, c2 = st.columns(2)
        exam = c1.date_input("Exam date", value=date.today() + timedelta(days=14),
                             min_value=date.today() + timedelta(days=1))
        hours = c2.slider("Study hours per day", 0.5, 8.0, 2.0, 0.5)
        if st.button("📅 Generate my study plan", type="primary"):
            res = o.build_study_plan(exam, hours)
            if res["status"] != "success":
                show_error(res)

        plan = o.state["study_plan"]
        if plan:
            st.write(plan["summary"])
            if plan.get("coach_message"):
                st.success(plan["coach_message"])
            for w in plan["warnings"]:
                st.warning(w)

            st.subheader("Time per topic")
            alloc = plan["topic_allocation"]
            st.bar_chart({a["topic"]: a["total_minutes"] for a in alloc}, y_label="Minutes")
            st.caption("  |  ".join(f"{a['topic']}: {minutes_with_hours(a['total_minutes'])}"
                                    for a in alloc))

            st.subheader("Day by day")
            for d in plan["daily_plan"]:
                title = (f"Day {d['day']} - {d['weekday']} {d['date']} - "
                         f"{minutes_with_hours(d['total_minutes'])}")
                if d["phase"] == "final_revision":
                    title += " - Final revision"
                with st.expander(title, expanded=d["day"] == 1):
                    if d["note"]:
                        st.caption(d["note"])
                    for b in d["blocks"]:
                        st.markdown(f"{ICON[b['status']]} **{b['topic']}** - "
                                    f"{minutes_with_hours(b['minutes'])}  \n{b['activity']}")
                        for q in b["focus_questions"]:
                            st.write(f"  - Revisit: {q}")
        elif o.state["previous_performance"]:
            st.info("You retook the quiz - generate the plan again to update it.")

        st.divider()
        st.download_button("⬇️ Download everything (JSON)",
                           json.dumps(o.export_state(), indent=2, ensure_ascii=False),
                           file_name="studylens_session.json", mime="application/json")
