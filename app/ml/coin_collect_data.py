import os
import time
import math
import traceback
from datetime import datetime, timedelta

import pandas as pd
import pyupbit

# -------- 설정 --------
SAVE_DIR = "D:\졸업프로젝트\CEEMD\코인 전종목"
REQUEST_INTERVAL = 0.15   # 초당 요청 제한 완화(공용 API 안전빵)
RETRY = 3                 # 요청 실패 시 재시도 횟수
RETRY_WAIT = 1.5          # 재시도 대기(초)
COUNT_PER_CALL = 200      # pyupbit 한 번에 가져올 수 있는 최대 캔들 수
# ----------------------

os.makedirs(SAVE_DIR, exist_ok=True)

def get_krw_tickers():
    # 1) 새 버전 스타일 시도
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        if tickers:
            return sorted(tickers)
    except TypeError:
        pass
    # 2) 구버전 호환: 전부 받아서 필터
    tickers = pyupbit.get_tickers()  # 전체
    return sorted([t for t in tickers if t.startswith("KRW-")])


def fetch_day_chunk(ticker, to=None, count=COUNT_PER_CALL):
    """
    ticker의 일봉을 최대 count개 가져온다.
    to: 'YYYY-MM-DD HH:MM:SS' 또는 datetime. 이 시각 '이전까지'의 캔들을 반환.
    반환: DataFrame (index=Datetime, columns=['open','high','low','close','volume','value'])
    """
    for i in range(RETRY):
        try:
            df = pyupbit.get_ohlcv(ticker=ticker, interval="day", to=to, count=count)
            return df
        except Exception:
            if i == RETRY - 1:
                raise
            time.sleep(RETRY_WAIT)

def load_existing_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df.set_index("timestamp", inplace=True)
            # 중복/정렬 정리
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df
        except Exception:
            # 손상 파일이면 백업 후 새로 생성
            backup = path + ".broken_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(path, backup)
            print(f"[경고] 손상된 CSV를 백업했습니다 -> {backup}")
    return None

def save_csv(df, path):
    out = df.copy()
    out = out.sort_index()
    out.reset_index().rename(columns={"index": "timestamp"}).to_csv(path, index=False)

def backfill_full_history(ticker):
    """
    해당 티커의 일봉을 가능한 과거까지 역방향으로 모두 수집.
    기존 CSV가 있으면 이어받기(백필)하고, 없으면 처음부터.
    """
    path = os.path.join(SAVE_DIR, f"{ticker.replace('-', '_')}.csv")
    existing = load_existing_csv(path)

    # 수집 시작점(to) 결정
    if existing is not None and not existing.empty:
        oldest = existing.index.min()
        # oldest 하루 전까지 더 가져오도록 'to'를 oldest로 설정
        cursor = oldest.strftime("%Y-%m-%d %H:%M:%S")
        full_df = existing
        # 안전상: 중복 제거
        full_df = full_df[~full_df.index.duplicated(keep="last")]
    else:
        cursor = None  # 최신부터 시작
        full_df = pd.DataFrame()

    print(f"[{ticker}] 수집 시작. 기존행={0 if existing is None else len(existing)}")

    # 루프: 과거로 이동하며 200개씩 수집
    total_added = 0
    last_oldest = None

    while True:
        time.sleep(REQUEST_INTERVAL)
        df = fetch_day_chunk(ticker, to=cursor, count=COUNT_PER_CALL)

        if df is None or df.empty:
            break

        # pyupbit는 컬럼 이름이 고정. 인덱스는 tz-naive KST 기준(일반적으로 KST).
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # 기존과 합치기
        if full_df is None or full_df.empty:
            full_df = df.copy()
        else:
            full_df = pd.concat([df, full_df]).sort_index()
            full_df = full_df[~full_df.index.duplicated(keep="last")]

        total_added += len(df)

        # 다음 페이지 커서(가장 오래된 캔들의 하루 전)
        oldest_now = df.index.min()
        if last_oldest is not None and oldest_now >= last_oldest:
            # 더 이상 과거로 못 감(보호조건)
            break
        last_oldest = oldest_now

        # Upbit의 'to'는 해당 시각 '이전' 캔들까지 반환하므로, oldest_now 하루 전으로 이동
        cursor = (oldest_now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        # 덩어리가 200 미만이면 더 과거가 없다는 신호일 가능성 큼
        if len(df) < COUNT_PER_CALL:
            break

    if full_df is None or full_df.empty:
        print(f"[{ticker}] 가져올 데이터가 없습니다.")
        return 0

    save_csv(full_df, path)
    print(f"[{ticker}] 완료: 총 {len(full_df)}행 (이번에 추가 {total_added}행) -> {path}")
    return total_added

def main():
    tickers = get_krw_tickers()
    print(f"KRW 마켓 종목 수: {len(tickers)}")
    print("예시 5개:", tickers[:5])

    # 전체 수집
    failures = []
    for i, t in enumerate(tickers, 1):
        try:
            print(f"\n[{i}/{len(tickers)}] {t}")
            backfill_full_history(t)
        except Exception as e:
            print(f"[에러] {t}: {e}")
            traceback.print_exc()
            failures.append(t)

    if failures:
        print("\n다음 티커에서 실패했습니다(재시도 권장):")
        print(", ".join(failures))
    else:
        print("\n전 종목 수집 완료!")

if __name__ == "__main__":
    main()
