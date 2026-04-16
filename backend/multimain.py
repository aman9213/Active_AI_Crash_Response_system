from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, cv2
from video_utils import extract_frames
from multimodel import create_caption_model
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llava = None
current_model = None

@app.on_event("startup")
async def load_model():
    global llava, current_model
    try:
        # Default model for Mac CPU
        current_model = "llava-1.5-q4"
        llava = create_caption_model(current_model)
        print(f"[INFO] Model '{current_model}' loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Model failed to load: {e}")


@app.post("/caption")
async def caption(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None)
    model: Optional[str] = Form("llava-1.5-q4")
):
    """
    Analyze crash image/video with optional custom query.
    
    Args:
        file: Image or video file
        query: Optional question about the image (e.g., "What vehicles are involved?")
    
    Examples:
        POST /caption
        - file: crash.jpg
        - query: "Describe the crash scene"
        
        POST /caption
        - file: crash.jpg
        - query: "What is the damage level?"
        
        POST /caption
        - file: crash.mp4
        - query: "Who is at fault?"
    """

    global llava, current_model

    # Check if model needs to be switched
    if model != current_model:
        print(f"[INFO] Switching model from '{current_model}' to '{model}'...")
        try:
            llava = create_caption_model(model)
            current_model = model
            print(f"[INFO] Model switched to '{model}'.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model '{model}': {str(e)}")
            
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
        
        # Use custom query if provided, otherwise default prompt
        if query:
            prompt = query
        else:
            prompt = (
                "Analyze this accident scene. Describe: "
                "1) Vehicles involved (make, model, color, damage) "
                "2) Road conditions "
                "3) Injuries visible "
                "4) Emergency services presence "
                "5) Estimated severity"
            )
        
        text = llava.generate_caption(frames, prompt)
        return {
            "query": query or "default",
            "description": text
        }
    finally:
        if os.path.exists(path):
            os.remove(path)