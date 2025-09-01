from app.ml.collect_url import get_news_for_keywords, extract_keywords, clean_title

def collect_news_as_json(question: str, days: int = 14, max_links: int = 200):
    keywords = extract_keywords(question)
    results = []
    for kw in keywords:
        urls_data = get_news_for_keywords([kw], days=days, max_links=max_links)
        for item in urls_data:
            results.append({
                "keyword": kw,
                "url": item.get("link"),
                "title": clean_title(item.get("title")),
                "date": item.get("date"),
            })
    return results
