from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import numpy as np
import io
import scipy.io.wavfile as wav
from database import setup_database, save_profile, load_profiles, delete_profile, list_profiles
from identifier import get_embedding, identify

app = FastAPI()

# Set up DB table on startup
@app.on_event("startup")
def on_startup():
    setup_database()

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import FileResponse

@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/profiles")
def get_profiles():
    return {"profiles": list_profiles()}


@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    
    # Read WAV audio
    sample_rate, audio = wav.read(io.BytesIO(contents))
    audio = audio.astype(np.float32) / 32768.0  # normalise to -1 to 1

    # Get embedding and save
    embedding = get_embedding(audio)
    profiles = load_profiles()
    save_profile(name, embedding)

    return {"message": f"✅ Profile saved for {name}!"}


@app.post("/identify")
async def identify_speaker(file: UploadFile = File(...)):
    contents = await file.read()

    sample_rate, audio = wav.read(io.BytesIO(contents))
    audio = audio.astype(np.float32) / 32768.0

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