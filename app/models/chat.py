from pydantic import BaseModel, Field
from typing import List

class ChatRequest(BaseModel):
    message: str = Field(..., description="질문/프롬프트")
    top_k: int = 10
    temperature: float = 0.3
    enable_news_collect: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    latency_ms: int
