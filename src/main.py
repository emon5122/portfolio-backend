from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints.business_idea_generator import (
    router as business_idea_generator_router,
)
from api.endpoints.chat import router as chat_router
from api.endpoints.data_loader import router as data_loader_router
from api.models import response as response_model
from api.models import user as user_model
from core.database import engine

load_dotenv()

user_model.Base.metadata.create_all(bind=engine)
response_model.Base.metadata.create_all(bind=engine)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(business_idea_generator_router)
app.include_router(chat_router)
app.include_router(data_loader_router)
