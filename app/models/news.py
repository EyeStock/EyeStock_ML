from pydantic import BaseModel, Field
from typing import List

class NewsRequest(BaseModel):
    question: str = Field(..., description="질문/프롬프트")
    days: int = Field(14, description="최근 며칠치 뉴스 수집할지 (기본=14일)")
    max_links: int = Field(200, description="최대 수집 뉴스 개수")

class NewsItem(BaseModel):
    keyword: str
    url: str
    title: str
    date: str

class NewsResponse(BaseModel):
    results: List[NewsItem]
