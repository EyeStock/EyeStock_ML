import time, asyncio, re
from typing import List
from app.ml import rag
from app.utils.user_command_logger import CommandPatternLogger
from app.models.chat import ChatRequest, ChatResponse
from app.ml.collect_url import extract_keywords, get_news_for_keywords, clean_title
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 주가/거래/공시 같은 금융 키워드 + 주요 기업/산업 키워드가 있으면 통과
def _is_stock_chunk(text: str) -> bool:
    if not text:
        return False

    finance_keys = [
        "주가","공시","실적","거래량","상한가","하한가","배당",
        "증자","감자","자사주","리포트","목표가","투자의견",
        "코스피","코스닥","나스닥","m&a","수주","신제품"
    ]
    company_keys = [
        "삼성","LG","현대","SK","네이버","카카오","테슬라","애플",
        "엔비디아","MS","구글","아마존"
    ]
    sector_keys = [
        "AI","반도체","바이오","전기차","로봇","2차전지","클라우드","디스플레이"
    ]

    keys = finance_keys + company_keys + sector_keys
    t = text.lower()
    return any(k.lower() in t for k in keys)

def _postprocess_chatty(text: str) -> str:
    t = re.sub(r"(?m)^\s*([\-–•\*\d]+\s*[.)]?)\s*", "", text)
    t = re.sub(r"\s*\n+\s*", " ", t).strip()
    return " ".join(re.split(r"(?<=[.!?。])\s+", t)[:3])

# 질문에서 키워드 추출 → 네이버 뉴스 수집 → 기사 파싱 → 벡터스토어 저장 (메타데이터 포함: url, title, date)
async def _fetch_and_store_news(question: str, duration_seconds: int = 60):
    keys = extract_keywords(question or "") or [question]
    news_items = get_news_for_keywords(keys, days=14, max_links=200)

    seen_hashes = set()
    start = time.time()

    for item in news_items:
        if time.time() - start > duration_seconds:
            print(f"[DEBUG] 수집 중단: {duration_seconds}초 초과")
            break
        url = item.get("link")
        title = clean_title(item.get("title") or "")
        date = item.get("date") or ""
        try:
            article = Article(url)
            article.download(); article.parse()
            text = (article.text or "").strip()
            if len(text) < 300 or "무단전재" in text:
                continue
            h = hash(text[:1000])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            ).split_text(text)

            if chunks:
                rag.vectorstore.add_texts(
                    chunks,
                    metadatas=[{"source": url, "title": title, "date": date} for _ in chunks]
                )
                print(f"[DEBUG] 저장됨: {title} | {date} | {url} | {len(chunks)} 청크")
        except Exception:
            print(f"[ERROR] 기사 처리 실패: {url} | {e}")
            pass
        await asyncio.sleep(0)

# RAG 기반 챗봇 처리
async def process_chat(req: ChatRequest, user_id: str) -> ChatResponse:
    t0 = time.time()
    logger = CommandPatternLogger(user_id=user_id)
    is_similar = await logger.process_command(req.message)

    # 뉴스 수집
    if req.enable_news_collect:
        duration = 10 if is_similar else 60
        await _fetch_and_store_news(req.message, duration_seconds=duration)

    # 검색 및 필터링
    retriever = rag.vectorstore.as_retriever(search_kwargs={"k": req.top_k})
    docs = retriever.invoke(req.message)
    print(f"[DEBUG] 검색된 문서 수: {len(docs)}")
    filtered = [d for d in docs if _is_stock_chunk(getattr(d, "page_content", ""))]
    print(f"[DEBUG] 필터링 후 문서 수: {len(filtered)}")

    # 답변 생성할 때 참고할 뉴스 문서가 0개라면 상위 3개를 fallback으로 사용
    if not filtered and docs:
        filtered = docs[:3]

    # 본문 + 메타데이터 기반 context 구성
    doc_context = "\n\n".join(
        f"[{d.metadata.get('date','')}] {d.metadata.get('title','')}\n{d.page_content[:300]}..."
        for d in filtered[:30] if d.page_content
    )

    # 프롬프트 입력
    prompt_input = {"patterns": "", "documents": doc_context, "question": req.message}
    raw = rag.rag_chain.invoke(prompt_input)

    answer = raw.split("###답변:")[-1].strip() if "###답변:" in raw else raw.strip()
    answer = _postprocess_chatty(answer)

    # 출처 URL 배열
    sources: List[str] = []
    for d in filtered:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)

    return ChatResponse(
        answer=answer,
        sources=sources,
        latency_ms=int((time.time() - t0) * 1000)
    )
