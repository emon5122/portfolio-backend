import os
import random

import torch
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.models.response import Tags
from api.validators.chat import Chat, ChatResponse
from core.config import ROOT_PATH
from core.database import get_db
from utils.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
from utils.train import generate_model

router = APIRouter(prefix="/chat", tags=["chatbot"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_location = None if torch.cuda.is_available() else torch.device("cpu")


MODEL = f"{ROOT_PATH}/data.pth"
if not os.path.exists(MODEL):
    generate_model()
model_data = torch.load(MODEL, map_location=map_location)
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


@router.post("/", status_code=status.HTTP_200_OK, response_model=ChatResponse)
def Chat(chat: Chat, db: Session = Depends(get_db)):
    sentence = tokenize(chat.text)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        tags = db.query(Tags).filter(Tags.tag == tag).first()
        responses = [r for r in tags.responses]
        return random.choice(responses)
    else:
        return {"response": "I don't understand"}
