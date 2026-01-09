# -*- coding: utf-8 -*-
"""
V4 모델 추론 테스트
"""
import sys
sys.path.append('K:/gdrive_mirror/연구분석업무/김광일/항적데이터 분석 프로젝트/선박교통분석 모듈 만들기')

import pandas as pd
import numpy as np
from glob import glob
import os

# V4 추론기 로드
from trajectory_inference_v4 import GridTrajectoryInferenceV4

print("=" * 60)
print("V4 격자 기반 모델 추론 테스트")
print("=" * 60)

# 모델 로드
model_path = "model/yeosu/model_v4/model_v4_best.pth"
vocab_path = "prepared_data_v4/grid_vocab.json"
scaler_path = "model/yeosu/model_v4/scaler_v4.npz"

print(f"\n[1] 모델 로드 중...")
inferencer = GridTrajectoryInferenceV4(
    model_path=model_path,
    vocab_path=vocab_path,
    scaler_path=scaler_path
)
print(f"  - 시퀀스 길이: {inferencer.seq_len}")
print(f"  - 예측 길이: {inferencer.pred_len}")
print(f"  - Vocab 크기: {inferencer.vocab_size}")

# 테스트 데이터 로드
print(f"\n[2] 테스트 데이터 로드...")
data_folder = "G:/NIA_ai_project/항적데이터 추출/여수"

# 파일 검색 (기본 폴더에서 직접)
csv_files = glob(os.path.join(data_folder, "*.csv"))
print(f"  - 총 파일 수: {len(csv_files)}")

# 충분히 긴 파일 찾기
test_file = None
for f in csv_files[:1000]:  # 처음 1000개에서 검색
    df_temp = pd.read_csv(f, encoding='utf-8')
    if len(df_temp) >= 70:  # 50 + 20 이상
        test_file = f
        break

if test_file:
    filename = os.path.basename(test_file)
    parts = filename.replace('.csv', '').split('_')

    if len(parts) >= 3:
        start_area = parts[1]
        end_area = parts[2]
    else:
        start_area = "남쪽 진입"
        end_area = "여수정박지B"

    print(f"  - 테스트 파일: {filename}")
    print(f"  - 출발: {start_area}, 도착: {end_area}")

    # 데이터 로드
    df = pd.read_csv(test_file, encoding='utf-8')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"  - 데이터 포인트: {len(df)}")
    print(f"  - 컬럼: {list(df.columns)}")

    # 60개 포인트 이상인지 확인 (50 입력 + 10 비교)
    if len(df) >= 60:
        # 처음 50개로 예측, 다음 10개와 비교
        input_df = df.iloc[:50].copy()
        actual_future = df.iloc[50:60].copy()

        print(f"\n[3] 예측 수행...")
        print(f"  - 입력 시작: {input_df['datetime'].iloc[0]}")
        print(f"  - 입력 끝: {input_df['datetime'].iloc[-1]}")
        print(f"  - 입력 위치: ({input_df['lat'].iloc[-1]:.4f}, {input_df['lon'].iloc[-1]:.4f})")

        # 예측
        predictions = inferencer.predict(
            input_df,
            n_steps=10,
            start_area=start_area,
            end_area=end_area
        )

        print(f"\n[4] 예측 결과:")
        print(predictions[['datetime', 'lat', 'lon', 'grid_id', 'sog', 'cog', 'confidence']].to_string())

        # 실제값과 비교
        if actual_future is not None and len(actual_future) > 0:
            print(f"\n[5] 실제값과 비교:")
            print("=" * 80)
            print(f"{'Step':>4} | {'예측 lat':>10} | {'실제 lat':>10} | {'예측 lon':>10} | {'실제 lon':>10} | {'거리오차(m)':>10}")
            print("-" * 80)

            errors = []
            for i, (pred_row, actual_row) in enumerate(zip(predictions.itertuples(), actual_future.itertuples())):
                pred_lat, pred_lon = pred_row.lat, pred_row.lon
                actual_lat, actual_lon = actual_row.lat, actual_row.lon

                # 거리 계산 (단순 유클리드, 도 -> 미터 변환)
                dlat = (pred_lat - actual_lat) * 111000  # 위도 1도 ≈ 111km
                dlon = (pred_lon - actual_lon) * 111000 * np.cos(np.radians(actual_lat))
                dist_error = np.sqrt(dlat**2 + dlon**2)
                errors.append(dist_error)

                print(f"{i+1:>4} | {pred_lat:>10.4f} | {actual_lat:>10.4f} | {pred_lon:>10.4f} | {actual_lon:>10.4f} | {dist_error:>10.1f}")

            print("-" * 80)
            print(f"평균 거리 오차: {np.mean(errors):.1f} m")
            print(f"최대 거리 오차: {np.max(errors):.1f} m")
            print(f"최소 거리 오차: {np.min(errors):.1f} m")
    else:
        print(f"  [WARN] 데이터가 50개 미만입니다: {len(df)}")
else:
    print("[ERROR] 테스트 파일을 찾을 수 없습니다.")

print("\n" + "=" * 60)
print("테스트 완료")
print("=" * 60)
