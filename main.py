from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import io
from pydub import AudioSegment
from database import setup_database, save_profile, load_profiles, delete_profile, list_profiles
from identifier import get_embedding, identify

app = FastAPI()

def read_audio(contents: bytes) -> np.ndarray:
    """Convert any audio format (webm, wav, etc) to numpy array."""
    audio = AudioSegment.from_file(io.BytesIO(contents))
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / 32768.0
    return samples

@app.on_event("startup")
def on_startup():
    setup_database()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/profiles")
def get_profiles():
    return {"profiles": list_profiles()}

@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    audio = read_audio(contents)
    embedding = get_embedding(audio)
    save_profile(name, embedding)
    return {"message": f"✅ Profile saved for {name}!"}

@app.post("/identify")
async def identify_speaker(file: UploadFile = File(...)):
    contents = await file.read()
    audio = read_audio(contents)
    profiles = load_profiles()
    name, score = identify(audio, profiles)
    return {
        "speaker": name,
        "score": round(score, 3),
        "identified": name is not None
    }

@app.delete("/profiles/{name}")
def remove_profile(name: str):
    delete_profile(name)
    return {"message": f"🗑️ Deleted profile for {name}"}