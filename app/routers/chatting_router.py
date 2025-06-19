from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat_service import run_chat

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/chatting", response_model=ChatResponse)
async def chatting(request: ChatRequest):
    answer = await run_chat(request.question)
    return ChatResponse(answer=answer)


# 테스트용 루트 경로
@router.get("/")
def root():
    return {"message": "Hello, FastAPI"}