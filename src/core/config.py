import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv("Secret_Key")
    ALGORITHM: str = os.getenv("Algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv("TokenExpire") or 24 * 60
    DB_USER: str = os.getenv("DBUSER")
    DB_PASS: str = os.getenv("DBPASS")
    DB_HOST: str = os.getenv("DBHOST")
    DB_NAME: str = os.getenv("DBNAME")


load_dotenv()
settings = Settings()
FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
