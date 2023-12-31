import json
import random

import torch
from fastapi import APIRouter, status

router = APIRouter(prefix="/chat", tags=["chatbot"])

from api.validators.chat import Chat
from core.config import ROOT_PATH
from utils.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_location = None if torch.cuda.is_available() else torch.device("cpu")


with open(f"{ROOT_PATH}/data.json", "r") as f:
    data = json.load(f)
MODEL = f"{ROOT_PATH}/data.pth"
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


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=Chat)
def Chat(chat: Chat):
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
        for intent in data["intents"]:
            if tag == intent["tag"]:
                return {"text": random.choice(intent["responses"])}
        else:
            for command in data.get("commands", []):
                if tag == command["action"]:
                    return {"text": random.choice(command["responses"])}
            else:
                for trait in data.get("personality_traits", []):
                    if tag == trait["trait"]:
                        return {"text": random.choice(trait["responses"])}
                else:
                    return {"text": "I do not understand..."}
    else:
        return {"text": "I do not understand..."}
