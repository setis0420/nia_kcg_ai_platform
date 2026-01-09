# -*- coding: utf-8 -*-
"""
Heading Inertia 테스트 스크립트
==============================
COG 방향 유지 효과를 테스트합니다.
"""
import sys
sys.path.append('K:/gdrive_mirror/연구분석업무/김광일/항적데이터 분석 프로젝트/선박교통분석 모듈 만들기')

import pandas as pd
import numpy as np
from trajectory_inference_v2 import TrajectoryInferenceV2

# 모델 로드
model_path = "model/yeosu/model_lstm_attn/model_lstm_attn.pth"
scaler_path = "model/yeosu/model_lstm_attn/scaler_lstm_attn.npz"

inferencer = TrajectoryInferenceV2(model_path, scaler_path)
print(f"모델 로드 완료 (seq_len={inferencer.seq_len})")
print(f"cog_mirror: {inferencer.cog_mirror}")
print(f"heading_inertia_lambda: {inferencer.heading_inertia_lambda}")

# 테스트 데이터 생성 (북쪽 방향으로 이동 중인 선박)
np.random.seed(42)
seq_len = inferencer.seq_len

# 북쪽으로 이동하는 선박 (COG ~350도)
base_lat = 34.5
base_lon = 127.7
base_cog = 350  # 북쪽 방향
base_sog = 12.0

# 시퀀스 데이터 생성
timestamps = pd.date_range('2024-01-01 00:00:00', periods=seq_len, freq='1min')
cog_values = base_cog + np.random.randn(seq_len) * 3  # 약간의 변동
cog_values = (cog_values + 360) % 360
sog_values = base_sog + np.random.randn(seq_len) * 0.5

# 위치 계산 (COG 방향으로 이동)
lat_values = [base_lat]
lon_values = [base_lon]
for i in range(1, seq_len):
    cog_rad = np.radians(cog_values[i-1])
    dist_nm = sog_values[i-1] / 60  # 1분 이동 거리 (해리)
    dlat = dist_nm / 60 * np.cos(cog_rad)
    dlon = dist_nm / 60 * np.sin(cog_rad) / np.cos(np.radians(lat_values[-1]))
    lat_values.append(lat_values[-1] + dlat)
    lon_values.append(lon_values[-1] + dlon)

test_df = pd.DataFrame({
    'datetime': timestamps,
    'lat': lat_values,
    'lon': lon_values,
    'sog': sog_values,
    'cog': cog_values
})

print(f"\n테스트 데이터: 북쪽으로 이동 (COG 평균: {test_df['cog'].mean():.1f}도)")
print(f"시작 위치: ({test_df['lat'].iloc[-1]:.4f}, {test_df['lon'].iloc[-1]:.4f})")

# 다양한 heading_inertia 값으로 테스트
n_steps = 30
inertia_values = [0.0, 0.3, 0.5, 0.7, 0.9]

print("\n" + "="*60)
print("Heading Inertia 테스트 (입력 COG: 350도 / 북쪽)")
print("="*60)

for inertia in inertia_values:
    print(f"\n▶ heading_inertia = {inertia}")

    preds = inferencer.predict_multi_from_df(
        test_df,
        n_steps=n_steps,
        heading_inertia=inertia
    )

    # COG 변화 분석
    cog_values_pred = preds['pred_cog'].values
    cog_5min = cog_values_pred[:5]
    cog_15min = cog_values_pred[10:15]
    cog_25min = cog_values_pred[25:30]

    print(f"  처음 5분 COG: {', '.join([f'{c:.0f}' for c in cog_5min])}도")
    print(f"  10-15분 COG:  {', '.join([f'{c:.0f}' for c in cog_15min])}도")
    print(f"  25-30분 COG:  {', '.join([f'{c:.0f}' for c in cog_25min])}도")

    # 방향 편차 계산 (350도와의 차이)
    def angle_diff(a, b):
        """두 각도 사이의 최소 차이"""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    mean_drift = np.mean([angle_diff(c, 350) for c in cog_values_pred])
    max_drift = max([angle_diff(c, 350) for c in cog_values_pred])
    print(f"  평균 방향 편차: {mean_drift:.1f}도, 최대 편차: {max_drift:.1f}도")

# 시각화
print("\n" + "="*60)
print("시각화 생성...")
print("="*60)

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 경로 비교
    ax1 = axes[0]
    ax1.plot(test_df['lon'], test_df['lat'], 'b-', linewidth=2, label='입력 경로')

    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for inertia, color in zip(inertia_values, colors):
        preds = inferencer.predict_multi_from_df(
            test_df, n_steps=n_steps, heading_inertia=inertia
        )
        ax1.plot(preds['pred_lon'], preds['pred_lat'],
                 linestyle='--', linewidth=1.5, color=color,
                 label=f'inertia={inertia}')

    ax1.set_xlabel('경도')
    ax1.set_ylabel('위도')
    ax1.set_title('경로 비교 (입력: 북쪽 방향)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. COG 변화
    ax2 = axes[1]
    for inertia, color in zip(inertia_values, colors):
        preds = inferencer.predict_multi_from_df(
            test_df, n_steps=n_steps, heading_inertia=inertia
        )
        ax2.plot(range(1, n_steps+1), preds['pred_cog'],
                 color=color, linewidth=1.5,
                 label=f'inertia={inertia}')

    ax2.axhline(y=350, color='blue', linestyle=':', linewidth=2, label='입력 COG (350°)')
    ax2.set_xlabel('예측 스텝 (분)')
    ax2.set_ylabel('COG (도)')
    ax2.set_title('COG 변화 비교')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 360)

    plt.tight_layout()
    plt.savefig('heading_inertia_test.png', dpi=150)
    print("그래프 저장: heading_inertia_test.png")
    plt.close()  # show() 대신 close() 사용

except ImportError:
    print("matplotlib가 없어서 시각화 생략")

print("\n테스트 완료!")
