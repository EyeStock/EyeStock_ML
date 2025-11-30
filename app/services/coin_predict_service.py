import traceback

import numpy as np
import pandas as pd
import pyupbit
import torch
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from app.ml.coin_predict_loader import get_model
from app.models.predict import PredictRequest
from app.utils.user_command_logger import coin_predict_log

WINDOW_SIZE = 120
RSI_PERIOD = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRICE_COLS = ["시가", "고가", "저가", "종가"]
MA_COLS = ["MA5", "MA20", "MA60", "MA120"]
VOL_COL = "거래량"
RSI_COL = "RSI14"

THRESHOLD = 0.483673


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "날짜" in df.columns:
        df = df.sort_values("날짜").reset_index(drop=True)

    close = df["종가"]

    # 이동평균 계산
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    # RSI 계산
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI14"] = rsi

    # 지표 계산 때문에 생긴 앞쪽 NaN 제거
    df = df.dropna().reset_index(drop=True)
    return df


def scale_window(window: pd.DataFrame) -> np.ndarray:
    price_data = window[PRICE_COLS + MA_COLS].to_numpy(dtype=float)
    vol_data = window[[VOL_COL]].to_numpy(dtype=float)
    rsi_data = window[[RSI_COL]].to_numpy(dtype=float)

    # 가격 min-max
    p_min = price_data.min()
    p_max = price_data.max()
    if p_max > p_min:
        price_scaled = (price_data - p_min) / (p_max - p_min)
    else:
        price_scaled = np.zeros_like(price_data)

    # 거래량 min-max
    v_min = vol_data.min()
    v_max = vol_data.max()
    if v_max > v_min:
        vol_scaled = (vol_data - v_min) / (v_max - v_min)
    else:
        vol_scaled = np.zeros_like(vol_data)

    # RSI (0~100 -> 0~1)
    rsi_scaled = rsi_data / 100.0

    # 합치기
    x_window = np.concatenate([price_scaled, vol_scaled, rsi_scaled], axis=1)
    return x_window


async def get_model_input(ticker: str):
    try:
        df = await run_in_threadpool(pyupbit.get_ohlcv, ticker, interval="minute1", count=300)
    except Exception as e:
        print(f"API 요청 실패: {e}")
        return None

    if df is None or len(df) < 200:
        print("Error: 데이터를 가져오지 못했거나 너무 짧습니다.")
        return None

    df = df.reset_index()
    rename_map = {
        "index": "날짜",
        "open": "시가",
        "high": "고가",
        "low": "저가",
        "close": "종가",
        "volume": "거래량"
    }
    df = df.rename(columns=rename_map)

    df_feat = add_indicators(df)

    if len(df_feat) < WINDOW_SIZE:
        print(f"[SKIP]: 데이터 부족 (길이 {len(df_feat)})")
        return None

    last_window = df_feat.iloc[-WINDOW_SIZE:]

    X_numpy = scale_window(last_window)
    X_tensor = torch.from_numpy(X_numpy).float().unsqueeze(0).to(DEVICE)

    return X_tensor


async def predict_coin(req: PredictRequest, x_user_id: str):
    try:
        input_tensor = await get_model_input(req.coinTicker)

        if input_tensor is None:
            raise HTTPException(status_code=400, detail="데이터 부족 또는 티커 오류")

        try:
            model = get_model()
        except Exception as e:
            raise HTTPException(status_code=400, detail="데이터 부족 또는 티커 오류")

        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).cpu().item()

        buy_signal = "1" if prob >= THRESHOLD else "0"

        coin_predict_log(x_user_id, req.coinTicker, buy_signal)

        return {
            "prediction": buy_signal
        }
    except Exception as e:
        traceback.print_exc()
        print(f"에러 메시지: {e}")
        raise HTTPException(status_code=500, detail=str(e))
