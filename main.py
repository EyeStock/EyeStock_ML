from fastapi import FastAPI
from app.routers import router

app = FastAPI(title="My FastAPI App")
app.include_router(router)

# 테스트용 루트 경로
@router.get("/")
def root():
    return {"message": "Hello, FastAPI"}