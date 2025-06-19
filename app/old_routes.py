from fastapi import APIRouter
from app.models import *
from app.services import run_prediction

router = APIRouter()

# 테스트용 루트 경로
@router.get("/")
def root():
    return {"message": "Hello, FastAPI"}

# 임시로 하드코딩된 더미 응답 반
def run_prediction(request: PredictRequest) -> PredictResponse:
    return PredictResponse(
        response=f"'{request.text}'에 대한 더미 응답입니다.",
        command_vector=[0.1, 0.2, 0.3]
    )

# 채팅 질문 → 응답 반환
@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = run_prediction(request)
    return result

# 채팅 종료(1) → 취향 벡터 저장 요청
@router.post("/fst_end_chat", response_model=EndChatPreferenceResponse)
def end_chat(request: EndChatPreferenceRequest):
    # 사용자 질문 텍스트 더미
    return EndChatPreferenceResponse(user_question=["q1","q2","q3"])

# 채팅 종료(2) → 명령 벡터 저장 요청
@router.post("/scnd_end_chat")
def end_chat(request: EndChatCommandRequest):
    return "ok"

# 새 채팅 시작 시 이전 벡터 여부 확인
@router.post("/use_prev_vector", response_model=UsePrevVectorResponse)
def use_prev_vector(request: UsePrevVectorRequest):
    # 이전 벡터 사용 여부 더미
    return UsePrevVectorResponse(
        response="이전 대화를 불러옵니다."
    )