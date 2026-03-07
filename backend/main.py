from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, cv2
from video_utils import extract_frames
from model import Videocaption

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded at startup — server won't crash if model fails to load
llava = None

@app.on_event("startup")
async def load_model():
    global llava
    try:
        llava = Videocaption()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Model failed to load: {e}")

@app.post("/caption")
async def caption(file: UploadFile = File(...)):

    # Return 503 if model didn't load instead of an unhandled crash
    if llava is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    filename = file.filename or "upload"
    path = "temp_" + filename

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Video
        if filename.endswith(("mp4", "avi", "mov")):
            frames = extract_frames(path)
        # Image
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames = [img]

        text = llava.generate_caption(
            frames,
            "Describe the crash scene in detail. Focus on vehicles involved, "
            "damage visible, road conditions, passenger condition, and any hazards."
        )
        return {"description": text}
    finally:
        # Always clean up temp file even if an exception occurs
        if os.path.exists(path):
            os.remove(path)

    