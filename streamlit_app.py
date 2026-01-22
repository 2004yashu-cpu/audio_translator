import streamlit as st
import os
import subprocess
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel

st.set_page_config(page_title="AI Notes Transcriber")

st.title("🧠 AI Notes Transcriber")

@st.cache_resource
def load_model():
    return WhisperModel("medium", device="cpu", compute_type="int8")

model = load_model()

LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
}

def convert_audio(in_path):
    out = in_path.rsplit(".", 1)[0] + "_clean.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out

src_lang = st.selectbox(
    "Spoken Language",
    options=list(LANG_MAP.keys()),
    format_func=lambda x: LANG_MAP[x],
)

uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

if uploaded:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        raw = tmp.name

    clean = convert_audio(raw)

    with st.spinner("Transcribing..."):
        segments, info = model.transcribe(
            clean,
            language=src_lang,
            vad_filter=True
        )

    text = " ".join(s.text for s in segments)
    st.text_area("Transcription", text, height=300)
