import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ================== 사용자 설정 ==================
DATA_DIR = r"plus_120/데이터셋/KOSPI200"
MODEL_PATH = r"plus_120/코인20/transformer_target5.pth"  # Transformer 모델 파라미터 파일

WINDOW_SIZE = 120
RSI_PERIOD = 14
X_FEAT_DIM = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.483673

PRICE_COLS = ["시가", "고가", "저가", "종가"]
VOL_COL = "거래량"
MA_COLS = ["MA5", "MA20", "MA60", "MA120"]
RSI_COL = "RSI14"


# ================== 모델 정의 ==================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerModel(nn.Module):
    def __init__(self, input_dim=X_FEAT_DIM,
                 d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        last_step = x[-1]
        out = self.fc(last_step)
        return out


# ================== 전처리 함수 ==================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 날짜 오름차순 정렬 및 이동평균/RSI14 추가
    df = df.sort_values("날짜").reset_index(drop=True)
    close = df["종가"]

    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI14"] = rsi

    df = df.dropna().reset_index(drop=True)
    return df


def scale_window(window: pd.DataFrame) -> np.ndarray:
    # 윈도우 스케일링
    price_data = window[PRICE_COLS + MA_COLS].to_numpy(dtype=float)
    vol_data = window[[VOL_COL]].to_numpy(dtype=float)
    rsi_data = window[[RSI_COL]].to_numpy(dtype=float)

    p_min = price_data.min()
    p_max = price_data.max()
    if p_max > p_min:
        price_scaled = (price_data - p_min) / (p_max - p_min)
    else:
        price_scaled = np.zeros_like(price_data)

    v_min = vol_data.min()
    v_max = vol_data.max()
    if v_max > v_min:
        vol_scaled = (vol_data - v_min) / (v_max - v_min)
    else:
        vol_scaled = np.zeros_like(vol_data)

    rsi_scaled = rsi_data / 100.0

    x_window = np.concatenate([price_scaled, vol_scaled, rsi_scaled], axis=1)
    return x_window


def get_last_window(df: pd.DataFrame):
    # 마지막 윈도우만 추출
    n = len(df)
    if n < WINDOW_SIZE:
        return None, None, None

    # 마지막 윈도우
    start = n - WINDOW_SIZE
    end = n
    window = df.iloc[start:end]

    x_window = scale_window(window)
    X = x_window[np.newaxis, :, :]  # (1, 120, 10)

    last_date = df.iloc[-1]["날짜"]
    last_close = df.iloc[-1]["종가"]

    return X, last_date, last_close


def load_csv(csv_path: str) -> pd.DataFrame:
    # CSV 로드 및 컬럼 이름 변경
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    # 코인 데이터 컬럼명 변경
    df = df.rename(columns={
        df.columns[0]: "날짜",
        "open": "시가",
        "high": "고가",
        "low": "저가",
        "close": "종가",
        "volume": "거래량"
    })

    required_cols = ["날짜", "시가", "고가", "저가", "종가", "거래량"]
    if not all(col in df.columns for col in required_cols):
        return None

    return df


# ================== 메인 예측 함수 ==================
def predict_last_windows(data_dir: str, model_path: str, threshold: float = THRESHOLD):
    # n개 종목의 마지막 윈도우를 전처리하고 Transformer 모델로 예측
    # 모델 로드
    model = TransformerModel()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("폴더 안에 CSV 파일이 없습니다.")
        return None

    results = []

    for csv_path in csv_files:
        symbol = os.path.splitext(os.path.basename(csv_path))[0]

        # CSV 로드
        df = load_csv(csv_path)
        if df is None:
            print(f"[SKIP] {symbol}: 필수 컬럼 부족")
            continue

        # 지표 추가
        df_feat = add_indicators(df)
        if len(df_feat) < WINDOW_SIZE:
            print(f"[SKIP] {symbol}: 데이터 부족 (길이 {len(df_feat)})")
            continue

        # 마지막 윈도우 추출
        X, last_date, last_close = get_last_window(df_feat)
        if X is None:
            print(f"[SKIP] {symbol}: 윈도우 추출 실패")
            continue

        # 예측
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(DEVICE)
            logit = model(X_tensor)
            prob = torch.sigmoid(logit).cpu().numpy().item()

        signal = 1 if prob >= threshold else 0

        csv_filename = os.path.basename(csv_path)
        buy_signal = 1 if prob >= threshold else 0

        results.append({
            "종목명": symbol,
            "signal": buy_signal
        })

        print(f"[{'BUY' if buy_signal else 'HOLD'}] {csv_filename}: prob={prob:.4f}, signal={buy_signal}")

    # 결과 DataFrame 생성
    if results:
        df_results = pd.DataFrame(results)
        return df_results
    else:
        print("처리할 종목이 없습니다.")
        return None


# ================== 실행 ==================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD}")
    print("=" * 60)

    df_results = predict_last_windows(DATA_DIR, MODEL_PATH, THRESHOLD)

    if df_results is not None:
        print("\n" + "=" * 60)
        print("=== 예측 결과 ===")
        print(df_results.to_string(index=False))

        # CSV로 저장
        output_path = "prediction_results.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n결과 저장: {output_path}")

        # 요약
        buy_count = df_results["signal"].sum()
        total_count = len(df_results)
        print(f"\nBUY 시그널: {buy_count}개 / 전체 {total_count}개")