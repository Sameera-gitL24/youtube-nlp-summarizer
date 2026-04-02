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