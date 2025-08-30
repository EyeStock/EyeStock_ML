from fastapi import APIRouter, Header, HTTPException
from typing import Optional
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag_service import process_chat

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, x_user_id: Optional[str] = Header(default="anon", alias="X-User-Id")):
    try:
        return await process_chat(req, x_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
