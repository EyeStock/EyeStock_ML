from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    coinTicker: str = Field(..., description="예측하고자 하는 코인 ticker")


class PredictResponse(BaseModel):
    prediction: str
