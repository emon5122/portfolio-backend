import io

import torch
from fastapi import APIRouter, UploadFile, status
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from api.validators.story import Story

router = APIRouter(prefix="/story-teller", tags=["chatbot"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
img_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


@router.post("/", status_code=status.HTTP_200_OK, response_model=Story)
async def Chat(file: UploadFile):
    image_content = await file.read()
    raw_image = Image.open(io.BytesIO(image_content)).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = img_model.generate(**inputs, max_length=50).to(device)
    val = processor.decode(out[0], skip_special_tokens=True)
    text = f"Write a story on the following scenerio:-> {val}"
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200).to(device)
    text = tokenizer.batch_decode(outputs)
    return {"text": text[0]}
