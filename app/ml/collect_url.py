import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import json
import os
import re

# === NAVER API 인증 ===
CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")


# 네이버 뉴스 API 호출
def get_request_url(url: str) -> str | None:
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", CLIENT_ID)
    req.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            return response.read().decode("utf-8")
    except Exception as e:
        print(f"[ERROR] {e}")
    return None

# 네이버 뉴스 검색 API 호출
def get_naver_search_result(sNode: str, search_text: str, page_start: int, display: int):
    base_url = "https://openapi.naver.com/v1/search"
    node = f"/{sNode}.json"
    parameters = f"?query={urllib.parse.quote(search_text)}&start={page_start}&display={display}&sort=date"
    url = base_url + node + parameters
    response_data = get_request_url(url)
    return json.loads(response_data) if response_data else None

# API 응답에서 링크/메타데이터 추출
def get_post_data(post: dict, json_result: list):
    org_link = post.get("originallink", "")
    link = post.get("link", "")
    title = post.get("title", "")
    pubdate = post.get("pubDate", "")

    if link:
        # pubDate → YYYY-MM-DD 형식 변환 시도
        try:
            date_str = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d")
        except Exception:
            date_str = ""
        json_result.append({
            "org_link": org_link,
            "link": link,
            "title": title,
            "date": date_str,
        })

# 링크 중복 제거
def collect_links(json_result: list) -> list[str]:
    return list(set(
        item["link"] for item in json_result
        if "link" in item and item["link"]
    ))


# URL 수집 (Chat/RAG)
def get_urls_for_keywords(keywords: list[str], max_per_query: int = 200, display_per_page: int = 100) -> list[str]:
    url_list = []
    for q in keywords:
        q = q.strip()
        if not q:
            continue
        json_result, page_start, collected = [], 1, 0
        while True:
            json_search = get_naver_search_result("news", q, page_start, display_per_page)
            if not json_search or json_search.get("display", 0) == 0:
                break
            for post in json_search.get("items", []):
                get_post_data(post, json_result)
                collected += 1
                if collected >= max_per_query:
                    break
            if collected >= max_per_query:
                break
            page_start += json_search.get("display", 0)
            if page_start > 1000:
                break
        links = collect_links(json_result)
        url_list.extend(links)
    return list(set(url_list))


# 메타 포함 뉴스 수집 (카드뉴스/News API)
def get_news_for_keywords(
    keywords: list[str], days: int = 14, max_links: int = 200, display_per_page: int = 100
) -> list[dict]:
    """키워드별 뉴스 기사 메타데이터 수집"""
    results = []
    cutoff_date = datetime.now() - timedelta(days=days)

    for q in keywords:
        q = q.strip()
        if not q:
            continue
        json_result, page_start, collected = [], 1, 0
        while True:
            json_search = get_naver_search_result("news", q, page_start, display_per_page)
            if not json_search or json_search.get("display", 0) == 0:
                break
            for post in json_search.get("items", []):
                get_post_data(post, json_result)
                collected += 1
                if collected >= max_links:
                    break
            if collected >= max_links:
                break
            page_start += json_search.get("display", 0)
            if page_start > 1000:
                break

        # 날짜 필터
        for item in json_result:
            try:
                if item["date"]:
                    pubdate = datetime.strptime(item["date"], "%Y-%m-%d")
                    if pubdate < cutoff_date:
                        continue
            except Exception:
                pass
            results.append(item)

    return results

# 카드뉴스 반환
def collect_news_as_json(keywords: list[str], days: int = 14, max_links: int = 200) -> list[dict]:
    results = []
    for kw in keywords:
        print(f"[수집] 키워드='{kw}', 기간={days}일, 최대={max_links}개")
        urls_data = get_news_for_keywords([kw], days=days, max_links=max_links)
        for item in urls_data:
            results.append({
                "keyword": kw,
                "url": item.get("link"),
                "title": item.get("title"),
                "date": item.get("date"),
            })
    return results


# 질문 → 키워드 추출
def extract_keywords(question: str) -> list[str]:
    """질문에서 한글 키워드(2글자 이상)를 정규식으로 추출"""
    words = re.findall(r"[\uac00-\ud7a3]{2,}", question)
    return list(set(words))
