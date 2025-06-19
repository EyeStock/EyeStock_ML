from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import datetime

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="vectorstore_user_logs",
    collection_name="user_command_logs",
    embedding_function=embeddings,
)

class CommandPatternLogger:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.retriever = vectorstore.as_retriever(k=5)
        self.previous_commands = []
        self.previous_embeddings = {}

    def similarity_to_previous(self, command, threshold=0.8):
        if not self.previous_commands:
            return False, 0.0
        if command not in self.previous_embeddings:
            self.previous_embeddings[command] = self.embeddings.embed_query(command)

        query_emb = self.previous_embeddings[command]
        prev_embs = np.array([self.previous_embeddings[c] for c in self.previous_commands])
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
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "command_text": command,
        }
        self.vectorstore.add_texts([command], metadatas=[meta])

    async def process_command(self, command):
        similar, sim_val = self.similarity_to_previous(command)
        print(f"[유사도] 이전 명령과 {sim_val:.2f}")

        if not similar:
            self.add_command(command)
            self.store_command(command)
            print("[저장] 새 패턴을 벡터스토어에 기록.")
        else:
            print("[스킵] 유사 패턴이므로 중복 저장 생략.")

    def retrieve_patterns(self, question):
        return self.retriever.invoke(question)
