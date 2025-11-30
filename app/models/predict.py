from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    coinTicker: str = Field(..., description="예측하고자 하는 코인 ticker")


class PredictResponse(BaseModel):
    coinTicker: str = Field(..., description="예측하고자 하는 코인 ticker")
    prediction: str = Field(..., description="0(하락), 1(상승)로 이루어진 예측 방향")
    prob: float = Field(..., description="모델 출력 sigmoid 값")
    timestamp: datetime = Field(..., description="예측 시각")
