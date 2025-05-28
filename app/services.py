from app.models import PredictRequest, PredictResponse

def run_prediction(request: PredictRequest) -> PredictResponse:
    # 여기에 AI 모델 로딩 및 예측 로직을 넣으면 됨
    dummy_result = sum(request.input_data)  # 예시로 합을 반환
    return PredictResponse(prediction=dummy_result)
