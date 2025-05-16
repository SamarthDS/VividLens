# src/actionclip_model.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ActionCLIPWrapper:
    def __init__(self, prompts):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.prompts = prompts

    def predict(self, frame_path):
        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(text=self.prompts, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape (1, len(prompts))
        probs = logits_per_image.softmax(dim=1)
        pred_idx = torch.argmax(probs).item()
        return self.prompts[pred_idx], probs[0][pred_idx].item()
