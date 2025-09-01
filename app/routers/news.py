from fastapi import APIRouter, Query
from typing import List, Optional
from app.services.card_news_service import collect_news_as_json
from app.models.news import NewsRequest, NewsResponse, NewsItem

router = APIRouter(prefix="/news", tags=["news"])

@router.post("", response_model=NewsResponse)
def get_news(req: NewsRequest) -> NewsResponse:
    data = collect_news_as_json(req.question, req.days, req.max_links)
    return NewsResponse(results=[NewsItem(**item) for item in data])
