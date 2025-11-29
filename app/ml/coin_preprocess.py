import os
import glob
import numpy as np
import pandas as pd

# ===== 사용자 설정 =====
DATA_DIR = r"plus_120/데이터셋/코인20"  # <-- 여기만 폴더 경로로 바꿔 주세요.
WINDOW_SIZE = 120
RSI_PERIOD = 14

PRICE_COLS = ["시가", "고가", "저가", "종가"]
VOL_COL = "거래량"
MA_COLS = ["MA5", "MA20", "MA60", "MA120"]
RSI_COL = "RSI14"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: columns = [날짜, 시가, 고가, 저가, 종가, 거래량]
    날짜 오름차순 정렬 후 이동평균/RSI14 추가하고 NaN 제거.
    """
    # 날짜 오름차순 정렬 (csv가 내림차순일 가능성 있으므로)
    
    df = df.sort_values("날짜").reset_index(drop=True)

    close = df["종가"]

    # 이동평균
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    # RSI14 계산 (단순 평균 버전)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df["RSI14"] = rsi

    # 지표가 다 생기지 않은 구간 제거
    df = df.dropna().reset_index(drop=True)

    return df


def scale_window(window: pd.DataFrame) -> np.ndarray:

    """
    한 윈도우(길이 120, 여러 컬럼 포함)를 받아서
    시가/고가/저가/종가/MA들(8개)은 공통 min-max,
    거래량은 별도 min-max,
    RSI14는 0~100 고정 min-max로 스케일링하여
    shape (WINDOW_SIZE, 10) ndarray와 사용한 p_min, p_max를 반환.
    """
    # 가격 + 이동평균 (8개)
    price_data = window[PRICE_COLS + MA_COLS].to_numpy(dtype=float)
    # 거래량 (1개)
    vol_data = window[[VOL_COL]].to_numpy(dtype=float)
    # RSI14 (1개)
    rsi_data = window[[RSI_COL]].to_numpy(dtype=float)

    # 가격+이동평균 공통 min-max
    p_min = price_data.min()
    p_max = price_data.max()
    if p_max > p_min:
        price_scaled = (price_data - p_min) / (p_max - p_min)
    else:
        price_scaled = np.zeros_like(price_data)

    # 거래량 별도 min-max
    v_min = vol_data.min()
    v_max = vol_data.max()
    if v_max > v_min:
        vol_scaled = (vol_data - v_min) / (v_max - v_min)
    else:
        vol_scaled = np.zeros_like(vol_data)

    # RSI: 0~100 고정 min-max
    rsi_scaled = rsi_data / 100.0

    # (120, 8) + (120, 1) + (120, 1) -> (120, 10)
    x_window = np.concatenate([price_scaled, vol_scaled, rsi_scaled], axis=1)
    return x_window




def make_xy_from_df(df: pd.DataFrame):
    """
    지표가 추가된 df에서 X, y 생성.
    X: (window_num, 120, 10)

    y: (window_num, 5) =
       [당일 종가(close_t),
        5일후 종가(close_t5),
        10일후 종가(close_t10),
        당일 대비 5일후 종가 변화량(close_t5 - close_t),
        당일 대비 10일후 종가 수익률((close_t10 - close_t) / close_t)]
    """
    n = len(df)
    X_list = []
    y_list = []

    for i in range(0, n - WINDOW_SIZE):
        start = i
        end = i + WINDOW_SIZE          # 윈도우 마지막 index+1
        idx_t = end - 1                # 윈도우에서 "당일" 인덱스 (120번째 날)
        idx_5 = idx_t + 5              # 5일후
        idx_10 = idx_t + 10            # 10일후

        # 10일후 종가까지 있어야 하므로 체크
        if idx_10 >= n:
            break

        window = df.iloc[start:end]

        # X 윈도우 (스케일링)
        x_window = scale_window(window)
        X_list.append(x_window)


        # y 타깃 계산
        close_t = df.iloc[idx_t]["종가"]
        close_t5 = df.iloc[idx_5]["종가"]
        close_t10 = df.iloc[idx_10]["종가"]

        if close_t != 0:
            return_5  = (close_t5  - close_t) / close_t
            return_10 = (close_t10 - close_t) / close_t
        else:
            return_5  = 0.0
            return_10 = 0.0

        y_list.append([close_t, close_t5, close_t10, return_5, return_10])
        
    if not X_list:
        return None, None

    X = np.stack(X_list, axis=0)  # (window_num, 120, 10)
    y = np.stack(y_list, axis=0)  # (window_num, 5)
    return X, y


def process_folder(data_dir: str):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print("폴더 안에 csv 파일이 없습니다.")
        return

    for csv_path in csv_files:
        print(f"처리 중: {csv_path}")
        try:
            # 인코딩은 상황에 맞게 변경 (utf-8-sig, cp949 등)
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp949")
        #=============코인사용할시!!!    
        df = df.rename(columns={
            df.columns[0]: "날짜",   # 첫 번째 컬럼 (보통 DatetimeIndex 였던 것)
            "open": "시가",
            "high": "고가",
            "low": "저가",
            "close": "종가",
            "volume": "저장량" if False else "거래량"  # 오타 방지용 예시
        })
        #=========================
        # 필수 컬럼이 있는지 확인
        required_cols = ["날짜", "시가", "고가", "저가", "종가", "거래량"]
        if not all(col in df.columns for col in required_cols):
            print(f"  필수 컬럼 부족: {csv_path}, 건너뜀.")
            continue

        # 지표 추가
        df_feat = add_indicators(df)

        if len(df_feat) < WINDOW_SIZE + 1:
            print(f"  데이터가 너무 짧음 (지표 추가 후 길이 {len(df_feat)}), 건너뜀.")
            continue

        X, y = make_xy_from_df(df_feat)
        if X is None:
            print(f"  유효한 윈도우가 없음, 건너뜀.")
            continue

        # 파일 이름 기반으로 저장
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        x_path = os.path.join("plus_120/코인20", f"{base_name}_X.npy")
        y_path = os.path.join("plus_120/코인20", f"{base_name}_y.npy")

        np.save(x_path, X)
        np.save(y_path, y)

        print(f"  저장 완료: X {X.shape}, y {y.shape}")
        print(f"  -> {x_path}")
        print(f"  -> {y_path}")


if __name__ == "__main__":
    process_folder(DATA_DIR)
