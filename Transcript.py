import yt_dlp
import whisper
import os
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
def download_audio(url):
    # Remove old audio files
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
print("🔄 Loading Whisper model...")
model_whisper = whisper.load_model("tiny")  # fast
print("✅ Whisper loaded!")

# -------- Step 3: Speech to Text --------
def speech_to_text():
    files = os.listdir()
    audio_file = [f for f in files if f.startswith("audio")][0]

    print("🔄 Transcribing audio...")
    result = model_whisper.transcribe(audio_file)
    print("✅ Transcription done!")

    return result["text"]

print("🔄 Loading summarization model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model_sum = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print("✅ Model loaded!")

# -------- Step 5: Summarization --------
def summarize_text(text):
    print("🔄 Generating summary...")

    chunk_size = 1000
    summaries = []

    # Split text into chunks
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]

        input_text = "Summarize: " + chunk

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        outputs = model_sum.generate(
            inputs["input_ids"],
            max_new_tokens=100
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    # Combine all summaries
    final_summary = " ".join(summaries)

    print("✅ Summary generated!")
    return final_summary

# -------- Step 6: Translation --------
def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return "❌ Translation failed"
