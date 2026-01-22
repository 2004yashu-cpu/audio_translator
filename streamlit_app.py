import streamlit as st
import os
import subprocess
import soundfile as sf
from tempfile import NamedTemporaryFile
from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Notes Translator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Notes Translator")
st.caption("Speech • Notes • Translation • Voice Output")

# -------------------------------------------------------------------
# LOAD MODEL (STREAMLIT CLOUD SAFE)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    return WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8"
    )

model = load_model()

# -------------------------------------------------------------------
# LANGUAGES
# -------------------------------------------------------------------
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
}
NAME_TO_CODE = {v: k for k, v in LANG_MAP.items()}

# -------------------------------------------------------------------
# AUDIO HELPERS
# -------------------------------------------------------------------
def convert_audio(in_path: str) -> str:
    out = in_path.rsplit(".", 1)[0] + "_clean.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out

def is_audio_valid(path: str):
    try:
        data, sr = sf.read(path)
        if len(data) / sr < 0.5:
            return False
        if abs(data).mean() < 1e-4:
            return False
        return True
    except Exception:
        return False

# -------------------------------------------------------------------
# TRANSLATION (SAFE)
# -------------------------------------------------------------------
def translate_via_english(text, src, tgt):
    if src != "en":
        text = GoogleTranslator(source=src, target="en").translate(text)
    if tgt != "en":
        text = GoogleTranslator(source="en", target=tgt).translate(text)
    return text.strip()

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
mode = st.radio(
    "Choose Mode",
    ["🎙 Audio → Translation + Voice", "📄 Text → Translation + Voice"]
)

# ===================================================================
# AUDIO MODE
# ===================================================================
if mode.startswith("🎙"):
    src_lang = st.selectbox(
        "Spoken Language",
        options=list(LANG_MAP.keys()),
        format_func=lambda x: LANG_MAP[x]
    )

    tgt_lang = NAME_TO_CODE[
        st.selectbox("Translate To", list(LANG_MAP.values()))
    ]

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
            st.error("Audio too short or silent.")
            st.stop()

        with st.spinner("Transcribing..."):
            segments, info = model.transcribe(
                clean_path,
                language=src_lang,
                vad_filter=True
            )

        text = " ".join(seg.text for seg in segments)

        st.subheader("📝 Detected Text")
        edited = st.text_area("Edit text", text, height=200)

        if st.button("🌍 Translate"):
            translated = translate_via_english(edited, src_lang, tgt_lang)
            st.subheader("🌍 Translation")
            st.write(translated)

            tts = gTTS(translated, lang=tgt_lang)
            buf = BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf)

# ===================================================================
# TEXT MODE
# ===================================================================
else:
    src_lang = st.selectbox(
        "Text Language",
        options=list(LANG_MAP.keys()),
        format_func=lambda x: LANG_MAP[x]
    )

    tgt_lang = NAME_TO_CODE[
        st.selectbox("Translate To", list(LANG_MAP.values()))
    ]

    text = st.text_area("Enter text", height=250)

    if st.button("🌍 Translate Text"):
        translated = translate_via_english(text, src_lang, tgt_lang)
        st.write(translated)
