from pydantic import BaseModel


class Story(BaseModel):
    text: str
