import json

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.models.response import Responses, Tags
from core.config import ROOT_PATH
from core.database import get_db

router = APIRouter(prefix="/data", tags=["data_loaders"])


@router.get("/load_data", status_code=status.HTTP_201_CREATED, response_model=str)
def load_data(db: Session = Depends(get_db)):
    with open(f"{ROOT_PATH}/data.json", "r") as f:
        data = json.load(f)
        for intent in data["intents"]:
            new_tag = Tags(tag=intent["tag"])
            db.add(new_tag)
            db.commit()
            db.refresh(new_tag)
            tag_id = new_tag.id
            for response in intent["responses"]:
                new_response = Responses(tag_id=tag_id, response=response)
                db.add(new_response)
                db.commit()
                db.refresh(new_response)
    return "data loaded"
