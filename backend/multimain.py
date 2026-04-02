from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, cv2
from video_utils import extract_frames
from multimodel import create_caption_model

app = FastAPI()
llava = None

@app.on_event("startup")
async def load_model():
    global llava
    try:
        # Switch model here: "blip2", "llava-1.5", "llava-next-video"
        llava = create_caption_model("blip2")  # ← BLIP-2 recommended for Mac CPU
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Model failed to load: {e}")

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    if llava is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    filename = file.filename or "upload"
    path = "temp_" + filename
    
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        if filename.endswith(("mp4", "avi", "mov")):
            frames = extract_frames(path)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames = [img]
        
        # Use crash scene prompt
        text = llava.generate_caption(
            frames,
            "a photograph of a car accident showing"
        )
        return {"description": text}
    finally:
        if os.path.exists(path):
            os.remove(path)