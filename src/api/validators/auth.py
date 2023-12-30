from typing import Optional

from pydantic import BaseModel, EmailStr, conint


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    id: Optional[str]
