"""
Google Colab-optimized model.py for AI Crash Response System

This file is designed to run on Google Colab with GPU support.
To use in Colab:

1. Upload this file to Colab or paste the code
2. Run each cell in order (or run all)
3. The model will auto-download and cache on Colab's GPU
"""

import sys
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np
import cv2


# ============================================================================
# Auto-install dependencies if running on Colab
# ============================================================================
def setup_colab():
    """Install required packages on Google Colab."""
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        print("[INFO] Detected Google Colab environment. Installing dependencies...")
        import subprocess
        # Install torch first (important for GPU support on Colab)
        subprocess.run(
            ["pip", "install", "-q", "-U", "torch", "torchvision", "torchaudio"],
            check=True
        )
        # Upgrade transformers to latest (fixes ImportError with CLIP)
        subprocess.run(
            ["pip", "install", "-q", "-U", "--force-reinstall", "transformers>=4.45.0"],
            check=True
        )
        # Other dependencies
        subprocess.run(
            ["pip", "install", "-q", "-U", "accelerate", "pillow", "opencv-python"],
            check=True
        )
        print("[INFO] Dependencies installed successfully.")
    else:
        # Also run setup if NOT in Colab (for local testing)
        print("[INFO] Running on local machine. Ensuring dependencies are up to date...")
        import subprocess
        subprocess.run(
            ["pip", "install", "-q", "-U", "transformers>=4.45.0", "accelerate"],
            check=True
        )
        print("[INFO] Dependencies updated.")
    
    return in_colab


def get_best_device() -> torch.device:
    """Auto-detect best available device: CUDA (GPU) → MPS → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] MPS (Apple Silicon) available.")
        return torch.device("mps")
    print("[INFO] Using CPU (slow).")
    return torch.device("cpu")


# ============================================================================
# Main Videocaption class
# ============================================================================
class Videocaption:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        """
        Initialize the image/video captioner using BLIP-2.
        
        BLIP-2 is lightweight (2.7B parameters) and much faster than LLaVA-NeXT.
        Other model options:
        - "Salesforce/blip2-opt-6.7b" (larger, slower)
        - "Salesforce/blip2-flan-t5-xl" (alternative architecture)
        
        Args:
            model_name: HuggingFace model ID
            device: torch.device or None (auto-detect)
        """
        # Auto-detect best device if none provided
        self.device = torch.device(device) if device else get_best_device()
        print(f"[INFO] Using device: {self.device}")

        # Load processor first — cheaper, fail fast if model name is wrong
        print(f"[INFO] Loading processor from '{model_name}'...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        print("[INFO] Processor loaded.")

        # Load model to detected device
        print(f"[INFO] Loading BLIP-2 model from '{model_name}'. This may take 1-2 minutes...")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("[INFO] Model loaded successfully.")

    def generate_caption(self, video_frames, prompts=None):
        """
        Generate a caption for image(s).
        
        Note: BLIP-2 is designed for images/static frames. For video, 
        it captions individual frames or we extract key frames.
        
        Args:
            video_frames: List of numpy arrays (BGR) or PIL Images (RGB)
            prompts: str, optional prompt (BLIP-2 works best without custom prompts)
                    If None, uses a generic prompt.
            
        Returns:
            str, generated caption
        """
        if prompts is None:
            prompts = "a photography of"

        # Convert BGR numpy arrays to RGB if needed
        frames_rgb = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                # Assume BGR (OpenCV format), convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(Image.fromarray(frame_rgb))
            else:
                # Assume already PIL Image in RGB
                frames_rgb.append(frame)

        # BLIP-2 generates a caption for each frame, concatenate them
        captions = []
        for img in frames_rgb:
            inputs = self.processor(images=img, text=prompts, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100)
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)

        # Return concatenated captions or just the first one
        return " ".join(captions) if len(captions) > 1 else captions[0]


# ============================================================================
# Testing code (runs on Colab)
# ============================================================================
if __name__ == "__main__":
    in_colab = setup_colab()
    
    device = get_best_device()
    print(f"\n{'='*60}")
    print(f"Preferred device: {device}")
    print(f"{'='*60}\n")

    # Initialize model
    print("Initializing Videocaption model...")
    video_captioner = Videocaption(device=str(device))

    # ========================================================================
    # Test 1: Load from URLs (works in Colab without uploading files)
    # ========================================================================
    print("\n" + "="*60)
    print("Test 1: Caption from URL image")
    print("="*60)
    
    try:
        from urllib.request import urlopen
        from PIL import Image
        import io

        # Sample car crash image (you can replace with your own URL)
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Car_accident_2009-by-RaSeLaSeD.jpg/640px-Car_accident_2009-by-RaSeLaSeD.jpg"
        
        print(f"[INFO] Downloading image from {url}...")
        img = Image.open(io.BytesIO(urlopen(url).read())).convert("RGB")
        print(f"[INFO] Image shape: {img.size}")

        prompts = "a photography of a car accident scene showing"
        print(f"[INFO] Prompt: {prompts}")
        print("[INFO] Generating caption... (this may take 30-60 sec on GPU, 2-5 min on CPU)")
        
        caption = video_captioner.generate_caption([np.array(img)], prompts)
        print(f"\n[RESULT] Generated Caption:\n{caption}\n")

    except Exception as e:
        print(f"[WARNING] URL test failed: {e}")
        print("[INFO] You can upload your own images and use them instead.")

    # ========================================================================
    # Test 2: Upload and process local file (if in Colab)
    # ========================================================================
    if in_colab:
        print("\n" + "="*60)
        print("Test 2: Upload and caption your own image (Colab only)")
        print("="*60)
        
        try:
            from google.colab import files
            print("[INFO] Uploading image files...")
            uploaded = files.upload()
            
            for filename in uploaded.keys():
                print(f"\n[INFO] Processing: {filename}")
                img = Image.open(filename).convert("RGB")
                
                prompts = "Describe this image in detail."
                caption = video_captioner.generate_caption([np.array(img)], prompts)
                print(f"[RESULT] {caption}\n")
        
        except Exception as e:
            print(f"[WARNING] Upload test failed: {e}")

    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)
