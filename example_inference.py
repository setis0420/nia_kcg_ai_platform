# -*- coding: utf-8 -*-
"""
간단한 예측 예시 코드
======================
이 파일을 참고하여 직접 코드에서 예측 수행
"""
# 현재 폴더 path
import sys
sys.path.append('K:/gdrive_mirror/연구분석업무/김광일/항적데이터 분석 프로젝트/선박교통분석 모듈 만들기')
sys.path.append('K:/gdrive_mirror/연구분석업무/김광일/항적데이터 분석 프로젝트/선박교통분석 모듈 만들기')

from load_trj import load_trj

import pandas as pd
import numpy as np
from trajectory_inference_v2 import TrajectoryInferenceV2


def interpolate_trajectory(trj, dt_minutes=1, datetime_col='datetime',
                           lat_col='lat', lon_col='lon',
                           sog_col='sog', cog_col='cog'):
    """
    불규칙한 AIS 데이터를 일정 시간 간격으로 보간

    Parameters
    ----------
    trj : pd.DataFrame
        AIS 항적 데이터 (DateTime, Latitude, Longitude, SOG, COG 컬럼 필요)
    dt_minutes : int
        보간 시간 간격 (분)
    datetime_col, lat_col, lon_col, sog_col, cog_col : str
        컬럼명

    Returns
    -------
    pd.DataFrame
        보간된 데이터 (datetime, lat, lon, sog, cog 컬럼)
    """
    trj = trj.copy()
    trj[datetime_col] = pd.to_datetime(trj[datetime_col], errors='coerce')
    trj = trj.dropna(subset=[datetime_col, lat_col, lon_col, sog_col, cog_col])
    trj = trj.sort_values(datetime_col).reset_index(drop=True)

    if len(trj) < 2:
        raise ValueError("보간을 위해 최소 2개 이상의 데이터 포인트가 필요합니다.")

    # 시작/종료 시간
    start_time = trj[datetime_col].iloc[0]
    end_time = trj[datetime_col].iloc[-1]

    # 일정 간격 시간 생성
    time_range = pd.date_range(start=start_time, end=end_time, freq=f'{dt_minutes}min')

    # 숫자형 시간 인덱스 (보간용)
    trj['_time_num'] = (trj[datetime_col] - start_time).dt.total_seconds()
    target_times = (time_range - start_time).total_seconds().values

    # 선형 보간
    lat_interp = np.interp(target_times, trj['_time_num'].values, trj[lat_col].values)
    lon_interp = np.interp(target_times, trj['_time_num'].values, trj[lon_col].values)
    sog_interp = np.interp(target_times, trj['_time_num'].values, trj[sog_col].values)

    # COG는 각도이므로 sin/cos로 보간 후 복원
    cog_rad = np.radians(trj[cog_col].values)
    sin_cog_interp = np.interp(target_times, trj['_time_num'].values, np.sin(cog_rad))
    cos_cog_interp = np.interp(target_times, trj['_time_num'].values, np.cos(cog_rad))
    cog_interp = np.degrees(np.arctan2(sin_cog_interp, cos_cog_interp))
    cog_interp = (cog_interp + 360) % 360

    return pd.DataFrame({
        'datetime': time_range,
        'lat': lat_interp,
        'lon': lon_interp,
        'sog': sog_interp,
        'cog': cog_interp
    })


def interpolate_and_predict(trj, inferencer, n_steps=30,
                            datetime_col='datetime', lat_col='lat',
                            lon_col='lon', sog_col='sog', cog_col='cog',
                            mmsi=None, start_area=None, end_area=None,
                            **predict_kwargs):
    """
    AIS 데이터를 1분 간격으로 보간하고 예측 수행

    Parameters
    ----------
    trj : pd.DataFrame
        AIS 항적 데이터
    inferencer : TrajectoryInferenceV2
        학습된 추론기
    n_steps : int
        예측할 스텝 수
    datetime_col, lat_col, lon_col, sog_col, cog_col : str
        입력 데이터의 컬럼명
    mmsi, start_area, end_area : optional
        Categorical 변수
    **predict_kwargs :
        predict_multi_from_df에 전달할 추가 인자

    Returns
    -------
    tuple (interpolated_df, prediction_df)
        보간된 데이터와 예측 결과
    """
    # 1. 보간 (1분 간격)
    interpolated = interpolate_trajectory(
        trj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lon_col=lon_col,
        sog_col=sog_col,
        cog_col=cog_col
    )

    print(f"[보간] 원본: {len(trj)}개 → 보간: {len(interpolated)}개 (1분 간격)")

    # 2. 데이터 충분성 체크
    if len(interpolated) < inferencer.seq_len:
        raise ValueError(f"보간된 데이터({len(interpolated)})가 시퀀스 길이({inferencer.seq_len})보다 적습니다.")

    # 3. 예측
    predictions = inferencer.predict_multi_from_df(
        interpolated,
        n_steps=n_steps,
        mmsi=mmsi,
        start_area=start_area,
        end_area=end_area,
        **predict_kwargs
    )

    return interpolated, predictions








# ============================================
# 1. 추론기 로드
# ============================================
model_path = "global_model_v2/lstm_global_v2.pth"
scaler_path = "global_model_v2/scaler_global_v2.npz"

inferencer = TrajectoryInferenceV2(model_path, scaler_path)
print(f"시퀀스 길이: {inferencer.seq_len}")


# ============================================
# 2. 예시 데이터 생성 (실제로는 CSV에서 로드)
# ============================================
# 실제 사용시: df = pd.read_csv("your_trajectory.csv")


# 공통 필터 조건
latlmt = [34.3, 35]
lonlmt = [127 + 40/60, 128.5]
soglmt = [0, 25]
length_range = [50, 400]



mmsi = 312454000
start_time = '2018-01-01 10:17:02'
end_time = '2018-01-02 23:27:20'	


# 항적 불러오기
trj = load_trj(
    latlmt=latlmt,
    lonlmt=lonlmt,
    soglmt=soglmt,
    datetimelmt=[start_time,end_time],
    mmsi = mmsi
)

print(f"\n원본 AIS 데이터: {len(trj)}개")
print(trj.head())


# ============================================
# 3. 보간 수행 (1분 간격)
# ============================================
interpolated_full = interpolate_trajectory(trj)
print(f"\n전체 보간 데이터: {len(interpolated_full)}개")

# 예측 비교를 위해 n_steps 설정
n_predict_steps = 30  # 예측할 스텝 수

# 입력용 데이터: 마지막 n_predict_steps를 제외한 부분
input_data = interpolated_full.iloc[:-n_predict_steps].copy()
# 실제 정답 데이터: 마지막 n_predict_steps
ground_truth = interpolated_full.iloc[-n_predict_steps:].copy()

print(f"입력 데이터: {len(input_data)}개 (예측 시작점까지)")
print(f"정답 데이터: {len(ground_truth)}개 (비교용)")

# 입력 데이터 마지막 부분(모델이 보는 부분)의 SOG 확인
seq_len = inferencer.seq_len
print(f"\n입력 데이터 마지막 {seq_len}개의 SOG 통계:")
last_sog = input_data['sog'].iloc[-seq_len:]
print(f"  평균: {last_sog.mean():.2f} knots")
print(f"  최소: {last_sog.min():.2f} knots")
print(f"  최대: {last_sog.max():.2f} knots")
print(f"\n정답 데이터 SOG 통계:")
print(f"  평균: {ground_truth['sog'].mean():.2f} knots")
print(f"  최소: {ground_truth['sog'].min():.2f} knots")
print(f"  최대: {ground_truth['sog'].max():.2f} knots")


# ============================================
# 4. 예측 수행
# ============================================
preds = inferencer.predict_multi_from_df(
    input_data,
    n_steps=n_predict_steps,
)

print("\n예측 결과:")
print(preds.head(10))

print(f"\n예측 SOG 통계:")
print(f"  평균: {preds['pred_sog'].mean():.2f} knots")
print(f"  최소: {preds['pred_sog'].min():.2f} knots")
print(f"  최대: {preds['pred_sog'].max():.2f} knots")


# ============================================
# 5. 결과 저장
# ============================================
preds.to_csv("prediction_result.csv", index=False, encoding='utf-8-sig')
print("\n결과 저장: prediction_result.csv")


# ============================================
# 6. 시각화: 원본 vs 예측 비교
# ============================================
try:
    import matplotlib
    matplotlib.use('Agg')  # 임시: 파일 저장만
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # 입력 데이터 (모델에 들어간 부분)
    ax.plot(input_data['lon'], input_data['lat'], 'b-', label='Input (History)', linewidth=2, alpha=0.7)

    # 실제 정답 (Ground Truth)
    ax.plot(ground_truth['lon'], ground_truth['lat'], 'g-', label='Ground Truth', linewidth=3)
    ax.scatter(ground_truth['lon'].iloc[-1], ground_truth['lat'].iloc[-1],
               c='green', s=200, marker='o', edgecolors='white', linewidths=2, zorder=5, label='GT End')

    # 예측 결과
    ax.plot(preds['pred_lon'], preds['pred_lat'], 'r--', label='Predicted', linewidth=2)
    ax.scatter(preds['pred_lon'].iloc[-1], preds['pred_lat'].iloc[-1],
               c='red', s=200, marker='*', edgecolors='white', linewidths=1, zorder=5, label='Pred End')

    # 예측 시작점 표시
    ax.scatter(input_data['lon'].iloc[-1], input_data['lat'].iloc[-1],
               c='blue', s=150, marker='s', edgecolors='white', linewidths=2, zorder=5, label='Prediction Start')

    # 시간 라벨 표시 (5스텝 간격으로)
    label_interval = 5
    for i in range(0, len(ground_truth), label_interval):
        time_str = ground_truth['datetime'].iloc[i].strftime('%H:%M')
        ax.annotate(time_str,
                    (ground_truth['lon'].iloc[i], ground_truth['lat'].iloc[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color='green')

    for i in range(0, len(preds), label_interval):
        time_str = preds['datetime'].iloc[i].strftime('%H:%M')
        ax.annotate(time_str,
                    (preds['pred_lon'].iloc[i], preds['pred_lat'].iloc[i]),
                    textcoords="offset points", xytext=(5, -10), fontsize=8, color='red')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Trajectory Prediction vs Ground Truth ({n_predict_steps} steps)')
    ax.legend(loc='best')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=150)
    print("시각화 저장: prediction_comparison.png")
    plt.close()

except ImportError:
    print("\n[INFO] matplotlib가 없어 시각화 생략")
