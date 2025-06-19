import asyncio, time, numpy as np
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from app.services.collect_url import get_urls_from_question
from app.services.user_command_logger import embeddings, vectorstore

MODEL_PATH = "app/Magistral-Small-2506-UD-IQ2_XXS.gguf"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    verbose=True,
    n_ctx=2048,
    n_batch=16,
    temperature=0.7,
)

# # === 뉴스/문서 RAG 용 Vectorstore ===
# vectorstore = Chroma(
#     persist_directory="vectorstore_data",
#     collection_name="my_collection",
#     embedding_function=embeddings,
# )


retriever = vectorstore.as_retriever(k=30)

prompt = PromptTemplate(
    template="""
[과거 질문 패턴]
{patterns}

[질문]
{question}

기업 실적, 주가 흐름, 재무 지표 등을 바탕으로 최소 5문장 이상으로 상세히 설명하고,
필요하다면 투자 주의사항도 포함하세요.

답변:
""",
    input_variables=["patterns", "question"],
)

rag_chain = prompt | llm | StrOutputParser()

class RAGSession:
    def __init__(self):
        self.previous_question = None
        self.previous_embedding = None

    def is_similar_to_previous(self, question, threshold=0.9):
        if not self.previous_question:
            return False
        query_emb = embeddings.embed_query(question)
        sim = np.dot(self.previous_embedding, query_emb) / (
            np.linalg.norm(self.previous_embedding) * np.linalg.norm(query_emb)
        )
        print(f"[유사도] 이전 질문과 {sim:.2f}")
        return sim >= threshold

    def update_previous(self, question):
        self.previous_question = question
        self.previous_embedding = embeddings.embed_query(question)

    async def collect_data_for_3_minutes(self, question):
        print(f"[INFO] '{question}' 에 대해 3분간 데이터 수집 시작")
        urls = get_urls_from_question(question)
        start = time.time()
        for url in urls:
            if time.time() - start > 180:
                break
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_text(text)
                if splits:
                    vectorstore.add_texts(splits)
            except Exception as e:
                print(f"[ERROR] URL 처리 실패: {e}")
            await asyncio.sleep(0)

    async def answer(self, question, pattern_context):
        print("=== START answer() ===")  # 로그 추가
        docs = retriever.invoke(question)
        doc_context = "\n\n".join([doc.page_content for doc in docs])
        prompt_input = {
            "patterns": pattern_context,
            "question": f"{question}\n\n{doc_context}",
        }
        answer = rag_chain.invoke(prompt_input)
        return answer
