from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import sys, os
import numpy as np
import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.questionnaire.scoring import load_questionnaire, score_responses
from src.fusion.feature_extraction import (
    build_facial_embedder, get_facial_embedding, facial_transform,
    build_speech_embedder, get_speech_embedding, DEVICE
)
from src.fusion.fusion_scoring import compute_mental_health_score
from src.recommendation.recommend import get_recommendations, generate_empathetic_summary, generate_conversational_questions
from src.recommendation.session_history import save_session, get_trend, trend_message, get_history_dataframe
from PIL import Image
from evaluation.gradcam import generate_gradcam
import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime

st.set_page_config(page_title="Talk2Mind", layout="centered")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.markdown("""
<style>
    .stApp {
        background-color: #16141F;
        background-image:
            radial-gradient(at 0% 0%, rgba(167, 139, 250, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(32, 29, 44, 0.5) 0px, transparent 50%);
        background-attachment: fixed;
        color: #F3F4F6;
    }
    div[data-testid="stAppViewBlockContainer"] {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    p, span, li, .stMarkdown, .stText, div[role="alert"] {
        color: #F3F4F6;
    }
    h1, h2, h3 {
        color: #F3F4F6;
    }
    label, div[data-testid="stWidgetLabel"], div[data-testid="stFileUploader"] label,
    div[data-testid="stCameraInput"] label, div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #F3F4F6 !important;
        font-weight: 500;
    }

    /* --- Session ID input --- */
    div[data-baseweb="input"] {
        background-color: #201D2C !important;
        border: 1px solid #334155 !important;
        border-radius: 6px;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="input"] input::placeholder {
        background-color: #201D2C !important;
        color: #F3F4F6 !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
    }

    /* --- Card containers --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #201D2C;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
        border: 1px solid #334155;
        margin: 0.5rem 0;
    }

    /* --- Metric cards --- */
    div[data-testid="metric-container"] {
        background: #16141F;
        border-radius: 10px;
        padding: 0.75rem 0.5rem;
        border: 1px solid #334155;
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #F3F4F6;
        font-weight: 700;
    }

    /* --- Buttons --- */
    div.stButton button {
        border-radius: 8px;
        font-weight: 500;
        color: #ffffff;
        background-color: #A78BFA;
        border: none;
    }
    div.stButton button:hover {
        background-color: #8B74E0;
    }

    /* --- Radio labels (questionnaire) --- */
    div[data-testid="stRadio"] {
        color: #F3F4F6;
    }
    div[data-testid="stRadio"] label {
        background: #16141F;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        margin: 0.15rem;
        border: 1px solid #334155;
        color: #F3F4F6 !important;
        font-weight: 500;
        transition: border-color 0.15s, background 0.15s;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #A78BFA;
    }
    /* Hide Streamlit chrome */
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="stAppLoadingIndicator"],
    [data-testid="stLoadingBar"] {
        display: none !important;
    }
    [data-testid="stApp"]::before,
    [data-testid="stApp"]::after {
        display: none !important;
        content: none !important;
    }

    /* --- File uploader --- */
    div[data-testid="stFileUploader"] {
        color: #F3F4F6;
    }
    div[data-testid="stFileUploader"] section {
        color: #F3F4F6;
    }
    div[data-testid="stFileUploader"] section button {
        color: #F3F4F6;
        background: #16141F;
        border: 1px solid #334155;
    }

    /* --- Alerts --- */
    div[data-testid="stInfo"] {
        background-color: #1E1B2E;
        border: 1px solid #A78BFA;
        border-radius: 8px;
        color: #F3F4F6;
    }
    div[data-testid="stInfo"] p {
        color: #F3F4F6;
    }
    div[data-testid="stWarning"] {
        border-radius: 8px;
        color: #F3F4F6;
    }
    div[data-testid="stError"] {
        border-radius: 8px;
        color: #F3F4F6;
    }
    div[data-testid="stSuccess"] {
        border-radius: 8px;
        color: #F3F4F6;
    }

    /* --- Selectbox --- */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #F3F4F6;
        background-color: #201D2C;
        border: 1px solid #334155;
        border-radius: 6px;
    }

    hr { margin: 0.5rem 0; }
    div[data-testid="stSlider"] {
        color: #F3F4F6;
    }

    /* --- Sidebar navigation --- */
    a[data-testid="stPageLink-NavLink"] {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.02em;
        transition: background 0.15s;
    }
    a[data-testid="stPageLink-NavLink"]:hover {
        background: #201D2C !important;
    }
    a[data-testid="stPageLink-NavLink"][aria-current="page"] {
        background: #A78BFA !important;
        color: #16141F !important;
        font-weight: 600;
    }
    .sidebar-brand {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #A78BFA;
    }
    .sidebar-section {
        font-size: 11px;
        letter-spacing: 0.1rem;
        text-transform: uppercase;
        color: #94a3b8;
        margin-top: 0.5rem;
    }


</style>
""", unsafe_allow_html=True)

qdata = load_questionnaire("src/questionnaire/questions.json")


def _score_gauge(score):
    if score < 25:
        color, label = "#2ECC71", "Low concern"
    elif score < 50:
        color, label = "#F1C40F", "Mild concern"
    elif score < 70:
        color, label = "#C9895A", "Moderate concern"
    else:
        color, label = "#E74C3C", "High concern"
    return f"""
    <div style="margin:0.3em 0 0.5em 0;">
      <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-bottom:2px;">
        <span>0</span><span>25</span><span>50</span><span>70</span><span>100</span>
      </div>
      <div style="height:22px; background:#2A2636; border-radius:11px; overflow:hidden;">
        <div style="height:100%; width:{min(score,100)}%; background:{color}; border-radius:11px; transition:width 0.8s ease;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
        <span style="font-size:1.6rem; font-weight:700; color:#F3F4F6;">{score:.1f}
          <span style="font-size:0.9rem; font-weight:400; color:#94a3b8;">/100</span>
        </span>
        <span style="background:{color}; color:white; padding:3px 14px; border-radius:20px; font-size:0.8rem; font-weight:600;">{label}</span>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Page 1: Check-In
# ---------------------------------------------------------------------------
def _crop_to_largest_face(frame, margin=0.2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return frame, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * margin), int(h * margin)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(frame.shape[1] - x, w + 2 * pad_x)
    h = min(frame.shape[0] - y, h + 2 * pad_y)
    return frame[y:y+h, x:x+w], True

def _checkin_page():
    st.title("Talk2Mind - Mental Well-being Check-in")

    if "q_index" not in st.session_state:
        st.session_state.q_index = 0
        st.session_state.responses = {}

    if "conversational_questions" not in st.session_state:
        with st.spinner("Preparing questions..."):
            st.session_state.conversational_questions = generate_conversational_questions(qdata)

    st.text_input(
        "Enter a session ID (any name/number, used to track your history)",
        value="guest", key="user_id",
    )

    if st.session_state.q_index < 9:
        st.header("1. Quick Questionnaire (PHQ-9)")
        idx = st.session_state.q_index
        items = qdata["items"]
        conv_qs = st.session_state.conversational_questions
        response_options = qdata["response_options"]
        labels = [opt["label"] for opt in response_options]

        if st.session_state.get("_q_answered_idx", -1) != idx:
            if "phq_radio" in st.session_state:
                del st.session_state["phq_radio"]
            st.session_state._q_answered_idx = idx

        st.radio(conv_qs[idx], options=labels, key="phq_radio", horizontal=True)

        if st.button("Next"):
            choice = st.session_state.phq_radio
            value = next(opt["value"] for opt in response_options if opt["label"] == choice)
            st.session_state.responses[items[idx]["id"]] = value
            st.session_state.q_index += 1
            st.rerun()

        q_progress = st.session_state.q_index / 9
        st.markdown(f"""
        <div style="margin:0.5em 0;">
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-bottom:4px;">
            <span>Question {st.session_state.q_index + 1} of 9</span>
            <span>{int(q_progress * 100)}%</span>
          </div>
          <div style="height:10px; background:#334155; border-radius:5px; overflow:hidden;">
            <div style="height:100%; width:{q_progress * 100}%; background:#A78BFA; border-radius:5px; transition:width 0.3s ease;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- All questions answered: capture phase ---
    responses = st.session_state.responses

    st.header("2. Facial Snapshot")
    facial_img = st.camera_input("Take a photo")

    st.header("3. Voice Sample")
    audio_file = st.file_uploader("Upload a short voice recording", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if st.button("Run Assessment"):
        if facial_img is None or audio_file is None:
            st.warning("Please provide both a facial snapshot and a voice recording.")
            return

        with st.spinner("Analyzing..."):
            file_bytes = np.asarray(bytearray(facial_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame, face_found = _crop_to_largest_face(frame)
            if not face_found:
                st.warning("No face detected — using full image (results may be less accurate)")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            facial_model = build_facial_embedder()
            tensor = facial_transform(frame_rgb).unsqueeze(0).to(DEVICE)
            _, facial_probs, _ = get_facial_embedding(facial_model, tensor)
            st.session_state["last_facial_probs"] = facial_probs

            try:
                # Pass the same tensor used for the polar chart so GradCAM
                # caption always matches the polar chart prediction.
                FACIAL_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "neutral"]
                pred_idx = int(facial_probs.flatten().argmax())
                gradcam_img, _, _, _ = generate_gradcam(
                    Image.fromarray(frame_rgb), model=facial_model
                )
                # Use probs from feature_extraction (same run as polar chart)
                gradcam_class = FACIAL_CLASSES[pred_idx]
                gradcam_conf = float(facial_probs.flatten()[pred_idx])
                st.session_state["gradcam_img"] = gradcam_img
                st.session_state["gradcam_class"] = gradcam_class
                st.session_state["gradcam_conf"] = gradcam_conf
            except Exception:
                st.session_state["gradcam_img"] = None

            import librosa
            y, sr = librosa.load(audio_file, sr=16000)
            from src.speech.preprocess_speech import extract_features
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                import soundfile as sf
                sf.write(tmp.name, y, sr)
                feat = extract_features(tmp.name)
            speech_model = build_speech_embedder()
            speech_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, speech_probs, _ = get_speech_embedding(speech_model, speech_tensor)
            st.session_state["last_speech_probs"] = speech_probs

            q_total = sum(st.session_state.responses.values())
            questionnaire_normalized = q_total / 27 * 100
            score_result = compute_mental_health_score(
                facial_probs.flatten(), speech_probs.flatten(), questionnaire_normalized
            )
            # Pass risk_flag from PHQ-9 Q9 so crisis alerts are triggered
            q9_value = st.session_state.responses.get("q9", 0)
            risk_flag = q9_value > 0
            recs = get_recommendations(score_result["band"], risk_flag=risk_flag)
            summary = generate_empathetic_summary(score_result, recs)
            trend = get_trend(st.session_state["user_id"], score_result["final_score"])

            st.session_state["last_score_result"] = score_result
            st.session_state["last_user_id"] = st.session_state["user_id"]
            st.session_state["last_recs"] = recs
            st.session_state["last_trend"] = trend
            st.session_state["last_summary"] = summary

        st.success("Assessment complete! View detailed results on the **Results** page.")


# ---------------------------------------------------------------------------
# Page 2: Results
# ---------------------------------------------------------------------------
def _sanitize(text):
    table = str.maketrans({
        '\u2014': '-', '\u2013': '-',
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u2026': '...',
    })
    return text.translate(table)

def _build_pdf_report(user_id, score_result, recs, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "Talk2Mind - Mental Well-being Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"Session ID: {user_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Mental Well-being Score", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"Score: {score_result['final_score']:.1f} / 100", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, f"Band: {score_result.get('band', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Emotional Insights", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    bd = score_result.get("breakdown", {})
    pdf.cell(0, 7, f"Facial Distress:     {bd.get('facial_distress', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, f"Speech Distress:     {bd.get('speech_distress', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, f"Questionnaire Score: {bd.get('questionnaire_score', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, _sanitize(summary))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Recommendations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=10)
    if recs.get("crisis_alert"):
        pdf.cell(0, 7, "Please see in-app crisis resources", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for r in recs.get("recommendations", []):
        if r != recs.get("crisis_alert"):
            pdf.cell(0, 7, _sanitize(f"- {r}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())

def _results_page():
    st.title("Talk2Mind - Mental Well-being Results")

    if "last_score_result" not in st.session_state:
        st.info("No assessment data yet. Complete a check-in first to see your results.")
        return

    score_result = st.session_state["last_score_result"]
    recs = st.session_state["last_recs"]
    trend = st.session_state["last_trend"]
    summary = st.session_state.get("last_summary", "")
    user_id = st.session_state.get("user_id", "guest")

    with st.container(border=True):
        st.subheader("Score")
        st.write(summary)
        st.markdown(_score_gauge(score_result["final_score"]), unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Emotional Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Facial Distress", f"{score_result['breakdown']['facial_distress']}")
        col2.metric("Speech Distress", f"{score_result['breakdown']['speech_distress']}")
        col3.metric("Questionnaire Score", f"{score_result['breakdown']['questionnaire_score']}")

        facial_probs = st.session_state.get("last_facial_probs")
        speech_probs = st.session_state.get("last_speech_probs")
        if facial_probs is not None and speech_probs is not None:
            # Order must match facial model's CLASSES from preprocess_facial.py
            # Speech model probs are remapped to same order in feature_extraction
            categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Neutral"]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=facial_probs.flatten(), theta=categories, fill="toself",
                name="Facial", line_color="#A78BFA",
            ))
            fig.add_trace(go.Scatterpolar(
                r=speech_probs.flatten(), theta=categories, fill="toself",
                name="Speech", line_color="#D4A574",
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155"),
                    angularaxis=dict(gridcolor="#334155"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F3F4F6"),
                legend=dict(font=dict(color="#F3F4F6")),
                showlegend=True, height=280,
                margin=dict(l=30, r=30, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        gradcam_img = st.session_state.get("gradcam_img")
        if gradcam_img is not None:
            st.markdown("---")
            st.write("**What the model focused on**")
            gcol1, gcol2 = st.columns([1, 2])
            with gcol1:
                st.image(gradcam_img, width=224)
            with gcol2:
                st.caption(
                    "Heatmap shows which facial regions most influenced the "
                    f"emotion prediction. The model predicted "
                    f"**{st.session_state['gradcam_class']}** "
                    f"({st.session_state['gradcam_conf']:.1%} confidence). "
                    "Red/orange areas = strongest influence on the decision."
                )

    with st.container(border=True):
        st.subheader("Recommendations")
        if recs["crisis_alert"]:
            st.error(recs["crisis_alert"])
        for r in recs["recommendations"]:
            if r != recs["crisis_alert"]:
                st.write(f"- {r}")

    pdf_bytes = _build_pdf_report(user_id, score_result, recs, summary)
    st.download_button(
        label="Download Report",
        data=pdf_bytes,
        file_name=f"talk2mind_report_{user_id}.pdf",
        mime="application/pdf",
    )

    with st.container(border=True):
        st.subheader("Trend")
        st.info(trend_message(trend))
        history_df = get_history_dataframe(user_id, current_score=score_result["final_score"])
        if len(history_df) >= 2:
            st.line_chart(history_df.set_index("timestamp")["final_score"])

    st.header("Feedback")
    mood_change = st.selectbox(
        "Compared to your last check-in, how do you feel?",
        ["better", "same", "worse", "first session"],
    )
    accuracy_rating = st.slider("How accurate did this assessment feel?", 1, 5, 3)

    if st.button("Submit Feedback & Save Session"):
        feedback = {"mood_change": mood_change, "accuracy_rating": accuracy_rating}
        save_session(st.session_state["last_user_id"], st.session_state["last_score_result"], feedback)
        st.success("Session saved. Thank you!")
        del st.session_state["last_score_result"]


# ---------------------------------------------------------------------------
# Page 3: About
# ---------------------------------------------------------------------------
def _about_page():
    st.title("Talk2Mind - Mental Well-Being Check-In")

    st.markdown("""
    Talk2Mind is a multimodal mental wellbeing assessment platform designed to
    analyse facial expressions, speech patterns, and questionnaire responses,
    offering a comprehensive view of an individual's emotional state

    **Methodology**

    1. **Questionnaire** — A nine-item assessment inspired by established
       clinical screening tools evaluates mood patterns over the preceding
       two weeks.
    2. **Facial Analysis** — A photograph is processed through a deep
       learning model trained to detect emotional indicators from facial
       expressions.
    3. **Speech Analysis** — A voice recording is analysed to extract
       acoustic features correlated with emotional states.
    4. **Multimodal Fusion** — Outputs from the three modalities are
       integrated into a unified mental wellness score, categorised into
       low, mild, moderate, or high concern levels.
    5. **Recommendations** — Personalised guidance and an empathetic
       summary are generated based on the assessment results.

    All data is processed locally and associated solely with the session
    identifier you provide. This platform is intended for screening and
    self-reflection purposes and does not constitute a clinical diagnosis.
    """)

    st.divider()
    st.caption(
        "If you are in crisis or require immediate support, please contact "
        "a licensed mental health professional or call a crisis helpline "
        "in your region."
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
checkin = st.Page(_checkin_page, title="Check-In")
results = st.Page(_results_page, title="Results")
about = st.Page(_about_page, title="About")

with st.sidebar:
    st.markdown('<div class="sidebar-brand">TALK2MIND</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)
    st.page_link(checkin, label="Check-In", use_container_width=True)
    st.page_link(results, label="Results", use_container_width=True)
    st.page_link(about, label="About", use_container_width=True)

pg = st.navigation([checkin, results, about], position="hidden")
pg.run()
