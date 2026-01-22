import streamlit as st
import whisper
import os
import subprocess
import soundfile as sf
from tempfile import NamedTemporaryFile
from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator

# -------------------------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Notes Translator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Notes Translator")
st.caption("Speech • Notes • Translation • Voice Output")

device = "cpu"

# -------------------------------------------------------------------
# LANGUAGE MAP
# -------------------------------------------------------------------
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}
NAME_TO_CODE = {v: k for k, v in LANG_MAP.items()}

# -------------------------------------------------------------------
# LOAD WHISPER MODEL
# -------------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium").to(device)

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

def is_audio_valid(path: str, min_sec: float = 0.5):
    try:
        data, sr = sf.read(path)
        if data is None or len(data) == 0 or sr <= 0:
            return False, 0.0

        duration = len(data) / sr
        if duration < min_sec:
            return False, duration

        if abs(data).mean() < 1e-4:
            return False, duration

        return True, duration
    except Exception:
        return False, 0.0

# -------------------------------------------------------------------
# TRANSLATION (PYTHON 3.13 SAFE)
# -------------------------------------------------------------------
def translate_via_english(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text.strip():
        return ""

    try:
        if src_lang != "en":
            text = GoogleTranslator(
                source=src_lang,
                target="en"
            ).translate(text)

        if tgt_lang != "en":
            text = GoogleTranslator(
                source="en",
                target=tgt_lang
            ).translate(text)

        text = text.replace(" ,", ",").replace(" .", ".")
        text = " ".join(text.split())
        return text.strip()
    except Exception:
        return ""

# -------------------------------------------------------------------
# TEXT TO SPEECH
# -------------------------------------------------------------------
def text_to_speech_block(text: str, lang_code: str):
    try:
        buffer = BytesIO()
        gTTS(text=text, lang=lang_code).write_to_fp(buffer)
        buffer.seek(0)
        st.audio(buffer, format="audio/mp3")
        st.download_button("⬇ Download Audio", buffer, "translated_audio.mp3")
    except Exception:
        st.warning("TTS not supported for this language.")

# -------------------------------------------------------------------
# MODE SELECTION
# -------------------------------------------------------------------
mode = st.radio(
    "Choose Mode",
    ["🎙 Audio → Translation + Voice", "📄 Text → Translation + Voice"],
)

# ===================================================================
# MODE 1: AUDIO PIPELINE
# ===================================================================
if mode == "🎙 Audio → Translation + Voice":

    src_lang = st.selectbox(
        "Spoken Language",
        options=list(LANG_MAP.keys()),
        index=2,
        format_func=lambda x: LANG_MAP[x],
    )

    tgt_lang = NAME_TO_CODE[
        st.selectbox("Translate To", list(LANG_MAP.values()))
    ]

    input_mode = st.radio(
        "Audio Source",
        ["📁 Upload audio file", "🎙 Record from microphone"]
    )

    raw_path = None

    if input_mode == "📁 Upload audio file":
        uploaded = st.file_uploader(
            "Upload audio (mp3 / wav / m4a)",
            type=["mp3", "wav", "m4a"]
        )
        if uploaded:
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                raw_path = tmp.name
            st.audio(uploaded)

    else:
        audio_value = st.audio_input("Record your voice (min 1 second)")
        if audio_value:
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_value.getbuffer())
                raw_path = tmp.name
            st.audio(audio_value)

    if raw_path:
        clean_path = convert_audio(raw_path)
        clean_ok, _ = is_audio_valid(clean_path)

        if not clean_ok:
            st.error(" Audio too short or silent. Record at least 1 second.")
            for p in [raw_path, clean_path]:
                if os.path.exists(p):
                    os.remove(p)
            st.stop()

        model = load_whisper_model()

        try:
            with st.spinner("Transcribing..."):
                result = model.transcribe(
                    clean_path,
                    fp16=False,
                    language=src_lang,
                    task="transcribe",
                    no_speech_threshold=0.3,
                )
        except RuntimeError:
            st.error("Whisper failed due to silent audio.")
            st.stop()

        detected_text = result.get("text", "").strip()

        st.subheader(" Detected Text (Editable)")
        edited_text = st.text_area(
            "Edit before translation",
            detected_text,
            height=220
        )

        if st.button(" Translate Audio"):
            translated = translate_via_english(
                edited_text, src_lang, tgt_lang
            )

            if translated:
                st.subheader(" Translation")
                st.text_area("Final Translation", translated, height=220)

                st.download_button("⬇ Download Transcript", edited_text, "transcript.txt")
                st.download_button("⬇ Download Translation", translated, "translation.txt")

                st.subheader("🔊 Voice Output")
                text_to_speech_block(translated, tgt_lang)

        for p in [raw_path, clean_path]:
            if os.path.exists(p):
                os.remove(p)

# ===================================================================
# MODE 2: TEXT PIPELINE
# ===================================================================
else:
    src_lang = st.selectbox(
        "Text Language",
        options=list(LANG_MAP.keys()),
        index=2,
        format_func=lambda x: LANG_MAP[x],
    )

    tgt_lang = NAME_TO_CODE[
        st.selectbox("Translate To", list(LANG_MAP.values()))
    ]

    manual_text = st.text_area(
        "Paste text here",
        height=260
    )

    if st.button(" Translate Text"):
        translated = translate_via_english(
            manual_text, src_lang, tgt_lang
        )

        if translated:
            st.subheader(" Translation")
            st.text_area("Final Translation", translated, height=260)

            st.download_button("⬇ Download Original", manual_text, "original.txt")
            st.download_button("⬇ Download Translation", translated, "translation.txt")

            st.subheader("🔊 Voice Output")
            text_to_speech_block(translated, tgt_lang)
