from fastapi import APIRouter, Query
from typing import List, Optional
from app.ml.collect_url import collect_news_as_json
from app.models.news import NewsRequest, NewsResponse, NewsItem

router = APIRouter(prefix="/news", tags=["news"])

@router.post("", response_model=NewsResponse)
def get_news(req: NewsRequest):
    data = collect_news_as_json(req.keywords, req.days, req.max_links)
    return {"results": [NewsItem(**item) for item in data]}
