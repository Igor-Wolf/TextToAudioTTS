from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import Literal
from TTS.api import TTS
import numpy as np
import soundfile as sf
import torch
import io
import tempfile
import os

app = FastAPI()

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

@app.post("/create-audio")
def create_audio(text: str, language: Literal["en", "pt", "es"] = "pt", voice_to_be_cloned: UploadFile = File()):

# para requisições do postman ou outros sites, para passar pelo body os valores
#     def create_audio(
#     text: str = Form(...),
#     language: Literal["en", "pt", "es"] = Form("pt"),
#     voice_to_be_cloned: UploadFile = File(...)
# )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_clone_voice:
        temp_clone_voice.write(voice_to_be_cloned.file.read())
        temp_clone_voice_path = temp_clone_voice.name

    try:
        text = text.replace('.', ';\n')
        audio = tts.tts(
            text=text,
            speaker_wav=temp_clone_voice_path,
            language=language
        )
    finally:
        os.remove(temp_clone_voice_path)

    audio_array = np.array(audio, dtype=np.float32)
    audio_stream = io.BytesIO()
    sf.write(audio_stream, audio_array, samplerate=22050, format="WAV")
    audio_stream.seek(0)

    return StreamingResponse(
        audio_stream,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'}
    )
