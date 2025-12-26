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
# 2. 데이터 로드 설정
# ============================================
# 공통 필터 조건
latlmt = [33.3, 36]
lonlmt = [126 + 40/60, 129.5]
soglmt = [0, 25]

# 데이터 조회용 시간 범위
mmsi = 312454000
data_start_time = '2018-01-01 03:17:02'
data_end_time = '2018-01-02 12:40:20'

# 예측 구간 설정 (이 구간 내에서 10분마다 예측)
predict_start_time = '2018-01-02 02:00:00'  # 예측 시작 시점
predict_end_time = '2018-01-02 09:00:00'    # 예측 종료 시점
predict_interval_min = 10                    # 예측 간격 (분)
n_predict_steps = 30                         # 각 예측의 스텝 수 (분)

# 항적 불러오기
trj = load_trj(
    latlmt=latlmt,
    lonlmt=lonlmt,
    soglmt=soglmt,
    datetimelmt=[data_start_time, data_end_time],
    mmsi=mmsi
)

print(f"\n원본 AIS 데이터: {len(trj)}개")
print(trj.head())


# ============================================
# 3. 보간 수행 (1분 간격)
# ============================================
interpolated_full = interpolate_trajectory(trj)
print(f"\n전체 보간 데이터: {len(interpolated_full)}개")
print(f"데이터 범위: {interpolated_full['datetime'].iloc[0]} ~ {interpolated_full['datetime'].iloc[-1]}")


# ============================================
# 4. 10분 간격 다중 예측 수행
# ============================================
predict_start = pd.to_datetime(predict_start_time)
predict_end = pd.to_datetime(predict_end_time)
seq_len = inferencer.seq_len

# 예측 시점 생성
prediction_times = pd.date_range(start=predict_start, end=predict_end, freq=f'{predict_interval_min}min')
print(f"\n예측 시점 수: {len(prediction_times)}개 ({predict_interval_min}분 간격)")

# 각 시점별 예측 결과 저장
all_predictions = []

for pred_time in prediction_times:
    # 예측 시점까지의 데이터 추출
    input_data = interpolated_full[interpolated_full['datetime'] <= pred_time].copy()

    # 시퀀스 길이보다 적으면 건너뜀
    if len(input_data) < seq_len:
        print(f"  [SKIP] {pred_time}: 데이터 부족 ({len(input_data)} < {seq_len})")
        continue

    # 예측 수행
    try:
        preds = inferencer.predict_multi_from_df(
            input_data,
            n_steps=n_predict_steps,
        )

        # 실제 경로 (Ground Truth) 추출
        gt_start_idx = interpolated_full[interpolated_full['datetime'] == pred_time].index
        if len(gt_start_idx) > 0:
            gt_start = gt_start_idx[0]
            gt_end = min(gt_start + n_predict_steps, len(interpolated_full))
            ground_truth = interpolated_full.iloc[gt_start:gt_end].copy()
        else:
            ground_truth = None

        all_predictions.append({
            'pred_time': pred_time,
            'predictions': preds,
            'ground_truth': ground_truth,
            'start_lat': input_data['lat'].iloc[-1],
            'start_lon': input_data['lon'].iloc[-1],
        })
        print(f"  [OK] {pred_time.strftime('%Y-%m-%d %H:%M')}: 예측 완료")

    except Exception as e:
        print(f"  [ERROR] {pred_time}: {e}")

print(f"\n총 예측 수행: {len(all_predictions)}개")


# ============================================
# 5. 결과 저장
# ============================================
# 모든 예측 결과를 하나의 CSV로 저장
all_preds_df = []
for item in all_predictions:
    preds_df = item['predictions'].copy()
    preds_df['pred_start_time'] = item['pred_time']
    all_preds_df.append(preds_df)

if all_preds_df:
    combined_preds = pd.concat(all_preds_df, ignore_index=True)
    combined_preds.to_csv("prediction_result.csv", index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: prediction_result.csv ({len(combined_preds)}행)")


# ============================================
# 6. 시각화: Folium 지도 (클릭하면 예측 경로 표시)
# ============================================
try:
    import folium

    # 지도 중심점: 마지막 예측의 종료점 기준
    if all_predictions:
        last_pred = all_predictions[-1]
        center_lat = last_pred['predictions']['pred_lat'].iloc[-1]
        center_lon = last_pred['predictions']['pred_lon'].iloc[-1]
    else:
        center_lat = interpolated_full['lat'].mean()
        center_lon = interpolated_full['lon'].mean()

    # Folium 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='CartoDB positron'
    )

    # 1. 전체 실제 항적 (파란색 실선)
    full_coords = list(zip(interpolated_full['lat'], interpolated_full['lon']))
    folium.PolyLine(
        full_coords,
        color='blue',
        weight=2,
        opacity=0.5,
        popup='실제 전체 항적'
    ).add_to(m)

    # 2. 각 예측 시점별 마커와 예측 경로
    colors = ['red', 'orange', 'purple', 'darkred', 'cadetblue', 'darkgreen', 'pink', 'darkblue']

    for idx, item in enumerate(all_predictions):
        pred_time = item['pred_time']
        preds = item['predictions']
        gt = item['ground_truth']
        color = colors[idx % len(colors)]

        time_str = pred_time.strftime('%Y-%m-%d %H:%M')

        # 예측 시작점 마커 (클릭하면 예측 경로 표시)
        # FeatureGroup으로 예측 경로를 감싸서 토글 가능하게
        fg = folium.FeatureGroup(name=f"예측 {time_str}", show=False)

        # 예측 경로 (점선)
        pred_coords = [(item['start_lat'], item['start_lon'])] + list(zip(preds['pred_lat'], preds['pred_lon']))
        folium.PolyLine(
            pred_coords,
            color=color,
            weight=3,
            opacity=0.8,
            dash_array='5',
        ).add_to(fg)

        # 실제 경로 (GT, 실선)
        if gt is not None and len(gt) > 0:
            gt_coords = list(zip(gt['lat'], gt['lon']))
            folium.PolyLine(
                gt_coords,
                color='green',
                weight=3,
                opacity=0.8,
            ).add_to(fg)

            # GT 종료점
            folium.CircleMarker(
                [gt['lat'].iloc[-1], gt['lon'].iloc[-1]],
                radius=6,
                color='green',
                fill=True,
                fill_opacity=0.8,
                popup=f"실제 종료<br>{gt['datetime'].iloc[-1].strftime('%H:%M')}"
            ).add_to(fg)

        # 예측 종료점
        folium.CircleMarker(
            [preds['pred_lat'].iloc[-1], preds['pred_lon'].iloc[-1]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"예측 종료<br>SOG: {preds['pred_sog'].iloc[-1]:.1f} kts<br>COG: {preds['pred_cog'].iloc[-1]:.1f}°"
        ).add_to(fg)

        # 5분 간격 포인트
        for i in range(0, len(preds), 5):
            row = preds.iloc[i]
            folium.CircleMarker(
                [row['pred_lat'], row['pred_lon']],
                radius=3,
                color=color,
                fill=True,
                popup=f"+{i}분<br>SOG: {row['pred_sog']:.1f}<br>COG: {row['pred_cog']:.1f}°"
            ).add_to(fg)

        fg.add_to(m)

        # 예측 시작점 마커 (항상 표시)
        popup_html = f"""
        <b>예측 시점: {time_str}</b><br>
        위치: ({item['start_lat']:.4f}, {item['start_lon']:.4f})<br>
        예측 {n_predict_steps}분<br>
        <i>좌측 레이어 패널에서 "{time_str}" 클릭하여 경로 표시</i>
        """
        folium.Marker(
            [item['start_lat'], item['start_lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color='blue', icon='circle', prefix='fa'),
        ).add_to(m)

    # 범례 추가
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; font-size: 12px;">
        <b>범례</b><br>
        <i style="background: blue; width: 20px; height: 2px; display: inline-block;"></i> 실제 전체 항적<br>
        <i style="background: green; width: 20px; height: 3px; display: inline-block;"></i> 실제 경로 (GT)<br>
        <i style="border-top: 2px dashed red; width: 20px; display: inline-block;"></i> 예측 경로<br>
        <br>
        <b>설정</b><br>
        예측 간격: {predict_interval_min}분<br>
        예측 길이: {n_predict_steps}분<br>
        총 예측: {len(all_predictions)}개
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 레이어 컨트롤 추가 (예측 경로 토글)
    folium.LayerControl(collapsed=False).add_to(m)

    # 저장
    output_html = 'prediction_map.html'
    m.save(output_html)
    print(f"\nFolium 지도 저장: {output_html}")
    print("※ 좌측 레이어 패널에서 각 예측 시점을 클릭하면 해당 예측 경로가 표시됩니다.")

except ImportError:
    print("\n[INFO] folium이 없어 지도 시각화 생략. pip install folium 으로 설치하세요.")
