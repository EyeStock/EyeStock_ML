from fastapi import FastAPI
from app.routers import chatting_router

app = FastAPI(title="My FastAPI App")
app.include_router(chatting_router.router)