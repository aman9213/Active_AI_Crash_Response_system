from fastapi import FastAPI, UploadFile, File
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

llava = Videocaption()

@app.post("/caption")
async def caption(file: UploadFile = File(...)):

    path = "temp_" + file.filename

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Video
    if file.filename.endswith(("mp4","avi","mov")):
        frames = extract_frames(path)

    # Image
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = [img]

    text = llava.generate_caption(frames)

    os.remove(path)

    return {"description": text}