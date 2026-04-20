import speech_recognition as sr
from pydub import AudioSegment
import os
from langchain_core.tools import tool

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an MP3 or WAV audio file to text using Google Speech Recognition.

    Parameters
    ----------
    file_path : str
        Path to the audio file (relative to LLMFiles directory).
        Supports .mp3 and .wav formats.

    Returns
    -------
    str  Transcribed text, or an error message.
    """
    try:
        full_path  = os.path.join("LLMFiles", file_path)
        final_path = full_path

        if full_path.lower().endswith(".mp3"):
            sound      = AudioSegment.from_mp3(full_path)
            final_path = full_path.replace(".mp3", ".wav")
            sound.export(final_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(final_path) as source:
            audio = recognizer.record(source)
            text  = recognizer.recognize_google(audio)

        if final_path != full_path and os.path.exists(final_path):
            os.remove(final_path)

        return text
    except Exception as e:
        return f"Transcription error: {e}"
