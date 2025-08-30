import asyncio
import time
import os
import numpy as np
import uuid
import gc
import torch
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
    GenerationConfig,
    TextStreamer,
    TextIteratorStreamer
)
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from newspaper import Article
from app.ml.collect_url import get_urls_from_question
from app.utils.user_command_logger import CommandPatternLogger
from app.ml.embed_loader import get_embedder

MODEL_PATH = os.getenv("MODEL_PATH")
EMBEDDING_PATH = os.getenv("EMBEDDING_PATH")

torch.cuda.empty_cache()

# === 1. 모델 로딩 (1회) ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
)

# === 2. 임베딩 모델 (sentence-transformers) ===
# embedder = SentenceTransformer(EMBEDDING_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
embedder = get_embedder()

class EmbeddingWrapper:
    def __init__(self, model):
        self.model = model
    def embed_query(self, text):
        return self.model.encode([text])[0]
    def embed_documents(self, texts):
        return self.model.encode(texts)

embeddings = EmbeddingWrapper(embedder)

# === 3. 벡터스토어 초기화 ===
vectorstore = Chroma(
    persist_directory="vectorstore_data",
    collection_name="my_collection",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(k=5)

# === 4. 뉴스 수집 및 벡터스토어 추가 ===
async def fetch_and_store_news(question, duration_seconds):
    urls = get_urls_from_question(question)
    seen_hashes = set()
    total = 0
    start = time.time()
    for url in urls:
        if time.time() - start > duration_seconds:
            print(f"[중단] {duration_seconds}초 초과")
            break
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()
            if len(text) < 300 or "무단전재" in text:
                continue
            text_hash = hash(text[:1000])
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_text(text)
            vectorstore.add_texts(
                chunks,
                metadatas=[{"source": url} for _ in chunks]
            )
            total += len(chunks)
            print(f"[추가] 기사: {url[:60]}... ({len(chunks)} 청크)")
        except Exception as e:
            print(f"[오류] 기사 수집 실패: {e}")
        await asyncio.sleep(0)
    print(f"[수집 종료] 총 청크: {total}")

# === 5. 최종 실행 ===
async def main():
    user_id = str(uuid.uuid4())
    logger = CommandPatternLogger(user_id=user_id)

    while True:
        question = input("\n질문 입력 ('e' 입력 시 종료): ")
        if question.strip().lower() == "e":
            break
        
        try:
            print("[로그] 질문 기록")
            start_time = time.time()
            is_similar = await logger.process_command(question)
        except Exception as e:
            print(f"[오류] 질문 처리 중 예외 발생: {e}")
            continue

        print("[로그] 유사 패턴 검색")
        similar_patterns = logger.retrieve_patterns(question=question, k=5)
        # 상위 질문 3개까지만 판단
        pattern_context = "\n".join([f"- {doc.page_content}" for doc in similar_patterns[:3]])

        print("[수집] 뉴스 시작...")
        duration = 10 if is_similar else 60
        await fetch_and_store_news(question, duration_seconds=duration)

        print("[검색] 유사 문서 검색...")
        docs = retriever.invoke(question)
        # 상위 문서 10개,, 
        doc_context = "\n\n".join([doc.page_content for doc in docs[:10] if doc.page_content.strip()])

        print("[생성] 스트리밍 시작...\n")
        prompt = f"""

당신은 주식 시장 뉴스를 분석하고, 사용자의 질문에 맞는 정보를 설명하는 챗봇입니다. 아래 정보를 바탕으로, 사용자의 질문과 관련된 주식의 주요 이슈, 가격 흐름, 변동성 요인 등을 객관적으로 설명하세요.

- 응답은 투자 조언이 아닌, 뉴스와 과거 패턴에 기반한 정보 제공에 집중합니다.
- "전문가와 상담하세요", "예측할 수 없습니다" 등의 회피 문장은 사용하지 마세요.
- 변동성, 상승/하락 요인, 시장 반응 등의 구체적인 내용을 간결하게 서술하세요.
- 전체 응답은 1~2문장으로 구성하며, 정보 밀도는 높게 유지하세요.
- 숫자, 시점, 추세, 주의사항이 있다면 구체적으로 언급하세요.
- 명확하고 중립적인 어조를 유지하되, 필요 시 간단한 투자 판단의 힌트를 포함해도 됩니다.

[이전 질문 패턴]
{pattern_context}

[뉴스 내용]
{doc_context}

[사용자 질문]
{question}

### 답변:"""
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=2.,
                                        )
        
        generation_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=True,
        repetition_penalty=1.2,
        temperature=0.7,
        top_p=0.9,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        t = Thread(target=model.generate, kwargs={
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "generation_config": generation_config,
        "streamer": streamer,
        })
        t.start()
        t.join()

        
        print("\n\n[완료] 전체 출력 완료.")
        print(f"[소요 시간] {time.time() - start_time:.2f}초")

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())