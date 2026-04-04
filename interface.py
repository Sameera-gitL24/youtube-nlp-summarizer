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
