from pydantic import BaseModel


class Chat(BaseModel):
    text: str


class ChatResponse(BaseModel):
    response: str
