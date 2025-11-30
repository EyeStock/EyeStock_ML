from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, chat, news, coin_predict

app = FastAPI(title="EyeStock Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(news.router)
app.include_router(coin_predict.router)
