import urllib.request
from datetime import datetime, timedelta
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from langchain_community.llms import HuggingFacePipeline
import torch
import os

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# NAVER API 요청 함수
def get_request_url(url):
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

# 뉴스 검색 요청 함수
def get_naver_search_result(sNode, search_text, page_start, display):
    base_url = "https://openapi.naver.com/v1/search"
    node = f"/{sNode}.json"
    parameters = f"?query={urllib.parse.quote(search_text)}&start={page_start}&display={display}&sort=date"
    url = base_url + node + parameters
    response_data = get_request_url(url)
    return json.loads(response_data) if response_data else None

def get_post_data(post, json_result):
    org_link = post.get("originallink", "")
    link = post.get("link", "")
    if org_link and link:
        json_result.append({"org_link": org_link, "link": link})

def collect_links(json_result):
    return list(set(
        item["link"] for item in json_result
        if "link" in item and item["link"]
    ))
    
# 키워드와 기간을 입력받아 뉴스 URL과 메타정보를 JSON 형태로 반환
def collect_urls_as_json(keywords, days, max_links=200):
    results = []
    for kw in keywords:
        print(f"[수집] 키워드='{kw}', 기간={days}일, 최대={max_links}개")
        # 기존 get_urls_for_keywords 내부 로직 활용
        urls_data = get_urls_for_keywords(kw, days=days, max_links=max_links)
        # urls_data는 [{"url": ..., "title": ..., "date": ...}, ...] 형태라고 가정
        for item in urls_data:
            results.append({
                "keyword": kw,
                "url": item.get("url"),
                "title": item.get("title"),
                "date": item.get("date"),
            })
    return results

# 단순 정규식 기반 키워드 추출 
def extract_keywords(question):
    words = re.findall(r"[\uac00-\ud7a3]{2,}", question)
    return list(set(words))

def build_queries_from_selected_keywords(prefs, max_queries=12):
    keys = [k for k in prefs.get("selected_keywords", []) if k]
    # 단일 + 2-그램 조합(너무 많아지지 않게 제한)
    queries = set(keys)
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pair = f"{keys[i]} {keys[j]}"
            queries.add(pair)
            if len(queries) >= max_queries:
                break
        if len(queries) >= max_queries:
            break
    return list(queries)[:max_queries]

def get_urls_for_keywords(keywords, max_per_query=200, display_per_page=100):
    sNode = "news"
    url_list = []
    for q in keywords:
        q = q.strip()
        if not q:
            continue
        json_result, page_start, collected = [], 1, 0
        while True:
            json_search = get_naver_search_result(sNode, q, page_start, display_per_page)
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
