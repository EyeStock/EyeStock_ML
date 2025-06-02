from fastapi import FastAPI
from app.routers import router

app = FastAPI(title="My FastAPI App")
app.include_router(router)