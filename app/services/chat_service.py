from app.services.rag_qa import RAGSession
from app.services.user_command_logger import CommandPatternLogger, vectorstore, embeddings

logger = CommandPatternLogger(vectorstore, embeddings)
session = RAGSession()

async def run_chat(question: str) -> str:
    await logger.process_command(question)
    similar_patterns = logger.retrieve_patterns(question)
    pattern_context = "\n".join([f"- {doc.page_content}" for doc in similar_patterns])

    if session.is_similar_to_previous(question):
        print("[INFO] 유사 질문 → 즉시 응답")
        answer = await session.answer(question, pattern_context)
    else:
        print("[INFO] 새로운 질문 → 뉴스 수집 후 응답")
        session.update_previous(question)
        await session.collect_data_for_3_minutes(question)
        answer = await session.answer(question, pattern_context)

    return answer
