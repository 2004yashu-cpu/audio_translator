import streamlit as st
import os
import subprocess
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Notes Transcriber",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Notes Transcriber")
st.caption("Multilingual Speech-to-Text (Cloud Stable)")

# -------------------------------------------------
# LOAD WHISPER MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8"
    )

model = load_model()

# -------------------------------------------------
# LANGUAGES
# -------------------------------------------------
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
}

# -------------------------------------------------
# AUDIO HELPERS
# -------------------------------------------------
def convert_audio(in_path: str) -> str:
    out = in_path.rsplit(".", 1)[0] + "_clean.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out

def is_audio_valid(path: str) -> bool:
    try:
        return os.path.getsize(path) > 10_000
    except Exception:
        return False

# -------------------------------------------------
# UI
# -------------------------------------------------
src_lang = st.selectbox(
    "Spoken Language",
    options=list(LANG_MAP.keys()),
    format_func=lambda x: LANG_MAP[x]
)

uploaded = st.file_uploader(
    "Upload audio (mp3 / wav / m4a)",
    type=["mp3", "wav", "m4a"]
)

if uploaded:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        raw_path = tmp.name

    clean_path = convert_audio(raw_path)

    if not is_audio_valid(clean_path):
        st.error("❌ Audio too short or silent.")
        st.stop()

    with st.spinner("🧠 Transcribing..."):
        segments, info = model.transcribe(
            clean_path,
            language=src_lang,
            vad_filter=True
        )

    text = " ".join(seg.text for seg in segments)

    st.subheader("📝 Transcription")
    st.text_area("Detected Text", text, height=300)
