import time, asyncio, re
from typing import List
from app.ml import rag
from app.utils.user_command_logger import CommandPatternLogger
from app.models.chat import ChatRequest, ChatResponse
from app.ml.collect_url import extract_keywords, get_urls_for_keywords
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _is_stock_chunk(text: str) -> bool:
    if not text:
        return False
    keys = [
        "주가","공시","실적","거래량","상한가","하한가","배당",
        "증자","감자","자사주","리포트","목표가","투자의견",
        "코스피","코스닥","나스닥","m&a","수주","신제품"
    ]
    return any(k in text.lower() for k in keys)

def _postprocess_chatty(text: str) -> str:
    t = re.sub(r"(?m)^\s*([\-–•\*\d]+\s*[.)]?)\s*", "", text)
    t = re.sub(r"\s*\n+\s*", " ", t).strip()
    return " ".join(re.split(r"(?<=[.!?。])\s+", t)[:3])

async def _fetch_and_store_news(question: str, duration_seconds: int = 60):
    keys = extract_keywords(question or "") or [question]
    urls = get_urls_for_keywords(keys, max_per_query=200)

    seen_hashes = set()
    start = time.time()

    for url in urls:
        if time.time() - start > duration_seconds:
            break
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
                chunk_size=500, chunk_overlap=0
            ).split_text(text)
            if chunks:
                rag.vectorstore.add_texts(chunks, metadatas=[{"source": url} for _ in chunks])
        except Exception:
            pass
        await asyncio.sleep(0)

async def process_chat(req: ChatRequest, user_id: str) -> ChatResponse:
    t0 = time.time()
    logger = CommandPatternLogger(user_id=user_id)
    is_similar = await logger.process_command(req.message)

    if req.enable_news_collect:
        duration = 10 if is_similar else 60
        await _fetch_and_store_news(req.message, duration_seconds=duration)

    retriever = rag.vectorstore.as_retriever(search_kwargs={"k": req.top_k})
    docs = retriever.invoke(req.message)
    filtered = [d for d in docs if _is_stock_chunk(getattr(d, "page_content", ""))]
    doc_context = "\n\n".join(d.page_content for d in filtered[:30] if d.page_content)

    prompt_input = {"patterns": "", "documents": doc_context, "question": req.message}
    raw = rag.rag_chain.invoke(prompt_input)
    answer = raw.split("###답변:")[-1].strip() if "###답변:" in raw else raw.strip()
    answer = _postprocess_chatty(answer)

    sources: List[str] = []
    for d in filtered:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("url")
        if src and src not in sources:
            sources.append(src)

    return ChatResponse(answer=answer, sources=sources, latency_ms=int((time.time() - t0) * 1000))
