import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


def get_best_device() -> torch.device:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Videocaption:
    def __init__(self, model_name="llava-hf/LLaVA-NeXT-Video-7B-hf", device=None):
        # Auto-detect best device if none provided
        self.device = torch.device(device) if device else get_best_device()
        print(f"[INFO] Using device: {self.device}")

        # Load processor first — cheaper, fail fast if model name is wrong
        self.processor = LlavaNextVideoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Load model to detected device (no device_map="auto" — avoids input/device mismatch)
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def generate_caption(self, video_frames, prompts):
        inputs = self.processor(text=prompts, videos=video_frames, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption


# if __name__ == "__main__":
#     device = get_best_device()
#     print("Preferred device:", device)
#     video_captioner = Videocaption(device=str(device))
#     video_frames = [
#         "/Users/vision/Desktop/pythoncode/AI_Crash Response System/data/car_accident.jpg",
#         "/Users/vision/Desktop/pythoncode/AI_Crash Response System/data/car_accident1.jpg",
#     ]
#     prompts = "Describe the crash scene in detail."
#     caption = video_captioner.generate_caption(video_frames, prompts)
#     print("Generated Caption:", caption)            



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
