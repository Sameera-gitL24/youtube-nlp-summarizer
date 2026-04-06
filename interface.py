import streamlit as st
import os
import yt_dlp
import whisper
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------- Language Dictionary --------
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Tamil": "ta",
    "Chinese": "zh",
    "Japanese": "ja",
    "Arabic": "ar",
    "Russian": "ru",
    "Korean": "ko",
    "Italian": "it",
    "Portuguese": "pt",
    "Bengali": "bn",
    "Urdu": "ur",
    "Malayalam": "ml",
    "Kannada": "kn"
}
@st.cache_resource
def load_models():
    model_whisper = whisper.load_model("tiny")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model_sum = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return model_whisper, tokenizer, model_sum

model_whisper, tokenizer, model_sum = load_models()

def download_audio(url):
    for file in os.listdir():
        if file.startswith("audio"):
            os.remove(file)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
def speech_to_text():
    files = os.listdir()
    audio_file = [f for f in files if f.startswith("audio")][0]

    st.info("🔄 Transcribing audio...")
    result = model_whisper.transcribe(audio_file)

    return result["text"]
# -------- Step 3: Summarization --------
def summarize_text(text):
    st.info("🔄 Generating summary...")

    chunk_size = 1000
    summaries = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]

        input_text = "Summarize: " + chunk
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        outputs = model_sum.generate(inputs["input_ids"], max_new_tokens=100)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# -------- Step 4: Translation --------
def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return "❌ Translation failed"
# 5
def process_video(url, selected_languages):
    st.info("⬇️ Downloading audio...")
    download_audio(url)

    text = speech_to_text()

    if not text.strip():
        return {"Error": "No speech detected"}

    summary = summarize_text(text)

    results = {"English": summary}

    for lang in selected_languages:
        code = LANGUAGES[lang]
        results[lang] = translate_text(summary, code)

    return results
# -------- UI --------
st.set_page_config(page_title="YouTube Summarizer", layout="wide")

st.title("🎥 YouTube Multilingual Summarizer")
st.write("Convert YouTube videos into summaries and translate into multiple languages 🌍")

# Input
url = st.text_input("Enter YouTube Video URL")

# Language selection
selected_languages = st.multiselect(
    "Select Languages",
    list(LANGUAGES.keys())
)
# Button
if st.button("🚀 Generate Summary"):

    if not url:
        st.warning("Please enter a YouTube URL")
    else:
        with st.spinner("Processing... Please wait ⏳"):
            result = process_video(url, selected_languages)

        st.success("✅ Done!")

        for lang, output in result.items():
            st.subheader(f"🌐 {lang}")
            st.write(output)