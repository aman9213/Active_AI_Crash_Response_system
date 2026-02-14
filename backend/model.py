import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

class Videocaption:
    def __init__(self, model_name="llava-hf/LLaVA-NeXT-Video-7B-hf", device="cpu"):
        self.device = device
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_name,device_map="auto")
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)
        

    def generate_caption(self, video_frames,prompts):
        inputs = self.processor(text=prompts,videos=video_frames, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Preferred device:", device)
    video_captioner = Videocaption(device=device)
    video_frames = ["/Users/vision/Desktop/pythoncode/AI_Crash Response System/data/car_accident.jpg", "/Users/vision/Desktop/pythoncode/AI_Crash Response System/data/car_accident2.jpg"]  # Replace with actual paths to video frames
    prompts = "Describe the video content."
    caption = video_captioner.generate_caption(video_frames, prompts)
    print("Generated Caption:", caption)            



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
