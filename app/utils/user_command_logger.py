import asyncio
import time
import numpy as np
from langchain_chroma import Chroma
from datetime import datetime, timedelta
import os
from sentence_transformers import SentenceTransformer
from app.ml.embed_loader import get_embedder

EMBEDDING_PATH = os.getenv("EMBEDDING_PATH")

# === Embedding 모델 ===
usr_embedder = get_embedder()

class EmbeddingWrapper:
    def __init__(self, usr_embedder):
        self.usr_embedder = usr_embedder

    def embed_query(self, text):
        return self.usr_embedder.encode([text])[0]

    def embed_documents(self, texts):
        return self.usr_embedder.encode(texts)

embeddings = EmbeddingWrapper(usr_embedder)

# === 사용자 패턴 벡터스토어 ===
vectorstore = Chroma(
    persist_directory="vectorstore_user_logs",
    collection_name="user_command_logs",
    embedding_function=embeddings,
)


# === 사용자 패턴 로거 ===
class CommandPatternLogger:
    def __init__(self, user_id, base_dir="vectorstore_user_logs"):
        self.user_id = user_id
        self.embeddings = embeddings
        user_store_dir = os.path.join(base_dir, user_id)

        self.vectorstore = Chroma(
            persist_directory=user_store_dir,
            collection_name=f"user_logs_{user_id}",
            embedding_function=embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(k=5)

        self.previous_commands = []
        self.previous_embeddings = {}

        try:
            stored_docs = self.vectorstore.get()["documents"]
            for text in stored_docs:
                if text.strip():
                    self.previous_commands.append(text)
                    self.previous_embeddings[text] = embeddings.embed_query(text)
            print(f"[초기화] 유저 '{user_id}' 질문 {len(self.previous_commands)}개 로드 완료")
        except Exception as e:
            print(f"[오류] 유저 '{user_id}' 벡터스토어 로딩 실패: {e}")

    def similarity_to_previous(self, command, threshold=0.85):
        if not self.previous_commands:
            return False, 0.0

        if command not in self.previous_embeddings:
            self.previous_embeddings[command] = self.embeddings.embed_query(command)

        query_emb = self.previous_embeddings[command]
        prev_embs = np.array(
            [self.previous_embeddings[c] for c in self.previous_commands]
        )
        sim = np.dot(prev_embs, query_emb) / (
            np.linalg.norm(prev_embs, axis=1) * np.linalg.norm(query_emb)
        )
        max_sim = np.max(sim)
        return max_sim >= threshold, max_sim

    def add_command(self, command):
        self.previous_commands.append(command)
        if command not in self.previous_embeddings:
            self.previous_embeddings[command] = self.embeddings.embed_query(command)
        if len(self.previous_commands) > 50:
            self.previous_commands.pop(0)
            
    def store_command(self, command):
        tagged_command = command 
        self.vectorstore.add_texts([tagged_command])
        self.previous_commands.append(tagged_command)
        self.previous_embeddings[tagged_command] = self.embeddings.embed_query(tagged_command)
        
    async def process_command(self, command):
        similar, sim_val = self.similarity_to_previous(command)
        print(f"[유사도] 이전 명령과 {sim_val:.2f}")

        if not similar:
            self.add_command(command)
            self.store_command(command) 
            print("[저장] 새 패턴을 벡터스토어에 기록.")
        else:
            print("[스킵] 유사 패턴이므로 중복 저장 생략.")

        return similar
        
    def retrieve_patterns(self, question, k=5, user_id=None):
        retriever = self.vectorstore.as_retriever(k=k)
        
        query = f"[{user_id}] {question}" if user_id else question
        results = retriever.invoke(query)

        print(f"[패턴 검색 결과] 총 {len(results)}건:")
        for doc in results:
            print(f"- {doc.page_content.strip()[:100]}")

        return results


# === export 정의 ===
__all__ = ["CommandPatternLogger", "vectorstore", "embeddings"]
