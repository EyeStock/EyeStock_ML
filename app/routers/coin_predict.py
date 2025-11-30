from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from app.models.predict import PredictResponse, PredictRequest
from app.services.coin_predict_service import predict_coin

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, x_user_id: Optional[str] = Header(default="anon", alias="X-User-Id")):
    try:
        return await predict_coin(req, x_user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
