"""
Multi-model image/video captioning system for AI Crash Response.

Supported models:
1. BLIP-2 (Salesforce) — Fast, lightweight, recommended
2. LLaVA-1.5 (Meta) — Versatile vision-language model
3. LLaVA-NeXT-Video (LLaVA) — Heavy but most powerful
4. Qwen-VL (Alibaba) — Excellent visual understanding
5. ViLBERT (Facebook) — Lightweight alternative

Switch between models by changing the model_class in main.py:
    from model import BLIP2CaptionModel
    captioner = BLIP2CaptionModel(device="auto")
"""

import torch
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod
from typing import List, Union


def get_best_device() -> torch.device:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] CUDA available. GPU: {torch.cuda.get_device_name(0)}")
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] MPS (Apple Silicon) available.")
        return torch.device("mps")
    print("[INFO] Using CPU.")
    return torch.device("cpu")


# ============================================================================
# Base class for all caption models
# ============================================================================
class BaseCaptionModel(ABC):
    """Abstract base class for all caption models."""

    def __init__(self, device=None):
        self.device = torch.device(device) if device else get_best_device()
        print(f"[INFO] Using device: {self.device}")

    @abstractmethod
    def generate_caption(self, video_frames: List, prompts: str = None) -> str:
        """Generate caption for frames. Must be implemented by subclasses."""
        pass

    def _convert_frame_to_rgb(self, frame):
        """Convert BGR numpy array to RGB PIL Image."""
        if isinstance(frame, np.ndarray):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return frame


# ============================================================================
# 1. BLIP-2 (Recommended for speed & efficiency)
# ============================================================================
class BLIP2CaptionModel(BaseCaptionModel):
    """
    BLIP-2 by Salesforce (2.7B parameters).
    
    Pros:
    - Fast (30-60 sec on GPU, 2-5 min on CPU)
    - Small model (5 GB)
    - Works well on images
    - Stable on CPU
    
    Cons:
    - Not specialized for video
    - Captions individual frames
    
    Model options:
    - "Salesforce/blip2-opt-2.7b" (default, fastest)
    - "Salesforce/blip2-opt-6.7b" (larger, better quality)
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device=None):
        super().__init__(device)
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"[INFO] Loading BLIP-2 from '{model_name}'...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("[INFO] BLIP-2 loaded.")

    def generate_caption(self, video_frames: List, prompts: str = None) -> str:
        if prompts is None:
            prompts = "a photography of"

        captions = []
        for frame in video_frames:
            img = self._convert_frame_to_rgb(frame)
            inputs = self.processor(images=img, text=prompts, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100)

            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)

        return " ".join(captions) if len(captions) > 1 else captions[0]


# ============================================================================
# 2. LLaVA-1.5 (Balanced option)
# ============================================================================
class LLaVA15CaptionModel(BaseCaptionModel):
    """
    LLaVA-1.5 by Meta (7B parameters).
    
    Pros:
    - Better visual reasoning than BLIP-2
    - Good image understanding
    - Supports custom prompts well
    - Responsive to detailed prompts
    
    Cons:
    - Slower than BLIP-2 (1-2 min on GPU, 5-10 min on CPU)
    - Larger model (13 GB)
    
    Model options:
    - "llava-hf/llava-1.5-7b-hf" (faster)
    - "llava-hf/llava-1.5-13b-hf" (better quality, slower)
    """

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device=None):
        super().__init__(device)
        from transformers import LlavaProcessor, LlavaForConditionalGeneration

        print(f"[INFO] Loading LLaVA-1.5 from '{model_name}'...")
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("[INFO] LLaVA-1.5 loaded.")

    def generate_caption(self, video_frames: List, prompts: str = None) -> str:
        if prompts is None:
            prompts = "Describe this image."

        captions = []
        for frame in video_frames:
            img = self._convert_frame_to_rgb(frame)
            inputs = self.processor(text=prompts, images=img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200)

            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)

        return " ".join(captions) if len(captions) > 1 else captions[0]


# ============================================================================
# 3. LLaVA-NeXT-Video (Most powerful but heavy)
# ============================================================================
class LLaVANextVideoCaptionModel(BaseCaptionModel):
    """
    LLaVA-NeXT-Video by LLaVA team (7B parameters).
    
    Pros:
    - Specialized for video understanding
    - Temporal reasoning across frames
    - Excellent crash scene analysis
    - Best quality
    
    Cons:
    - Very slow (2-5 min on GPU, 15+ min on CPU)
    - Large model (14-16 GB)
    - May crash on weak CPUs
    - Requires trust_remote_code=True
    
    Note: Only recommended if you have a decent GPU.
    """

    def __init__(self, model_name: str = "llava-hf/LLaVA-NeXT-Video-7B-hf", device=None):
        super().__init__(device)
        from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

        print(f"[INFO] Loading LLaVA-NeXT-Video from '{model_name}'...")
        self.processor = LlavaNextVideoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("[INFO] LLaVA-NeXT-Video loaded.")

    def generate_caption(self, video_frames: List, prompts: str = None) -> str:
        if prompts is None:
            prompts = "Describe this video scene in detail."

        # Convert to RGB for processor
        frames_rgb = [self._convert_frame_to_rgb(f) for f in video_frames]

        inputs = self.processor(text=prompts, videos=frames_rgb, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption


# ============================================================================
# 4. Qwen-VL (Excellent visual understanding)
# ============================================================================
class QwenVLCaptionModel(BaseCaptionModel):
    """
    Qwen-VL by Alibaba (4B parameters).
    
    Pros:
    - Excellent visual understanding
    - Good balance of speed and quality
    - Smaller than LLaVA (8 GB)
    - Supports Chinese and English
    
    Cons:
    - Less common, fewer examples
    - Medium speed (45 sec - 2 min on GPU)
    
    Note: Requires qwen-vl-chat or similar variant.
    """

    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat", device=None):
        super().__init__(device)
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError:
            print("[WARNING] Qwen-VL not available. Install: pip install qwen-vl")
            raise

        print(f"[INFO] Loading Qwen-VL from '{model_name}'...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("[INFO] Qwen-VL loaded.")

    def generate_caption(self, video_frames: List, prompts: str = None) -> str:
        if prompts is None:
            prompts = "Describe the scene."

        captions = []
        for frame in video_frames:
            img = self._convert_frame_to_rgb(frame)
            inputs = self.processor(
                text=prompts, images=[img], return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=150)

            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)

        return " ".join(captions) if len(captions) > 1 else captions[0]


# ============================================================================
# Model factory — easy switching
# ============================================================================
MODEL_REGISTRY = {
    "blip2": BLIP2CaptionModel,
    "llava-1.5": LLaVA15CaptionModel,
    "llava-next-video": LLaVANextVideoCaptionModel,
    "qwen-vl": QwenVLCaptionModel,
}


def create_caption_model(model_type: str = "blip2", device=None):
    """
    Factory function to create a caption model.
    
    Args:
        model_type: One of ["blip2", "llava-1.5", "llava-next-video", "qwen-vl"]
        device: torch.device or None (auto-detect)
    
    Returns:
        BaseCaptionModel instance
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](device=device)


# Alias for backward compatibility
Videocaption = BLIP2CaptionModel


if __name__ == "__main__":
    device = get_best_device()
    print(f"\nAvailable models: {list(MODEL_REGISTRY.keys())}\n")

    # Example: Switch models here
    print("="*60)
    print("Testing BLIP-2 model (default, fastest)")
    print("="*60)

    # Create model using factory
    captioner = create_caption_model("blip2", device=str(device))

    # Test with local images
    test_images = [
        "/Users/vision/Desktop/pythoncode/AI_Crash Response System/data/car_accident.jpg",
        "/Users/vision/Desktop/pythoncode/AI_Crash Response System/data/car_accident1.jpg",
    ]

    for img_path in test_images:
        try:
            img = Image.open(img_path).convert("RGB")
            caption = captioner.generate_caption(
                [np.array(img)],
                "a photograph of a car accident showing"
            )
            print(f"\n{img_path}:")
            print(f"  Caption: {caption}")
        except FileNotFoundError:
            print(f"[WARNING] Image not found: {img_path}")

    print("\n" + "="*60)            



# import torch
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from PIL import Image

# device = "cpu"

# processor = Blip2Processor.from_pretrained(
#     "Salesforce/blip2-opt-2.7b"
# )

# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float32
# ).to(device)

# def caption_image(img: Image.Image):
#     inputs = processor(images=img, return_tensors="pt").to(device)

#     out = model.generate(**inputs, max_new_tokens=60)
#     return processor.decode(out[0], skip_special_tokens=True)
