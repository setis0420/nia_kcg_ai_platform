# -*- coding: utf-8 -*-
"""
V13 선박 항적 예측 테스트
- 수심 제약 + 과거 항적 방향 반영
- 보정 전/후 비교 시각화
"""

import os
import glob
import pandas as pd
import numpy as np
import folium
import webbrowser
from trajectory_predictor_v13 import predict_trajectory_v13, TrajectoryPredictorV13

# 폴더 설정 (스크립트 위치 기준)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, 'test_data')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results_v13')
os.makedirs(RESULT_DIR, exist_ok=True)


def create_comparison_map(df, result, name, mmsi, region):
    """보정 전/후 비교 지도 생성"""
    last_pos = result['last_position']

    # 지도 생성
    m = folium.Map(location=[last_pos[0], last_pos[1]], zoom_start=12)

    # 과거 경로 (초록)
    input_coords = df[['lat', 'lon']].values.tolist()
    folium.PolyLine(input_coords, color='green', weight=4,
                    popup='과거 30분', opacity=0.8).add_to(m)
    folium.CircleMarker(input_coords[0], radius=6, color='green',
                        fill=True, popup='30분 전').add_to(m)

    # 보정 전 예측 경로 (회색 점선) - 보정 적용된 경우만
    if result.get('correction_applied') and 'original_coords' in result:
        original_coords = result['original_coords'].tolist()
        folium.PolyLine(original_coords, color='gray', weight=3,
                        dash_array='5,10', popup='보정 전 (원본)', opacity=0.5).add_to(m)

        # 원본 종점 마커
        folium.CircleMarker(original_coords[-1], radius=5, color='gray',
                            fill=True, fill_opacity=0.5, popup='보정 전 종점').add_to(m)

    # 보정 후 예측 경로 (빨강)
    pred_coords = result['predicted_coords'].tolist()
    folium.PolyLine(pred_coords, color='red', weight=4,
                    popup='예측 60분 (보정)', opacity=0.9).add_to(m)

    # 10분 단위 마커
    for idx, label in [(9, '+10분'), (19, '+20분'), (29, '+30분'),
                       (39, '+40분'), (49, '+50분'), (59, '+60분')]:
        if idx < len(pred_coords):
            folium.CircleMarker(pred_coords[idx], radius=5, color='white',
                                fill=True, fill_color='red', fill_opacity=1,
                                popup=label).add_to(m)

    # 현재 위치 (노랑)
    folium.CircleMarker([last_pos[0], last_pos[1]], radius=10, color='orange',
                        fill=True, fill_color='yellow', popup='현재 위치').add_to(m)

    # 보정 정보
    correction_info = ""
    if result.get('correction_applied'):
        avg_dist = np.mean(result.get('correction_distance_km', [0]))
        max_dist = np.max(result.get('correction_distance_km', [0]))
        correction_info = f" | 보정: 평균 {avg_dist:.2f}km, 최대 {max_dist:.2f}km"
    else:
        correction_info = " | 보정 미적용"

    # 타이틀
    title = f"""<div style='position:fixed;top:10px;left:50px;z-index:1000;
                background:white;padding:10px;border-radius:5px;font-size:14px;'>
        <b>V13 예측</b> | {name} | MMSI: {mmsi} |
        {result['shiptype_name']} | {result['length_name']}{correction_info}
        <br><span style='color:green'>■</span> 과거 30분
        <span style='color:gray'>- -</span> 보정 전
        <span style='color:red'>■</span> 보정 후
    </div>"""
    m.get_root().html.add_child(folium.Element(title))

    return m


def main():
    # 테스트 파일 목록
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.csv')))

    print("=" * 60)
    print("V13 선박 항적 예측 테스트")
    print("- 수심 제약: depth >= 10m")
    print("- 과거 항적 방향 반영")
    print("=" * 60)
    print(f"테스트 파일 수: {len(test_files)}")
    print()

    if not test_files:
        print("[오류] test_data 폴더에 CSV 파일이 없습니다.")
        return

    for csv_file in test_files:
        filename = os.path.basename(csv_file)
        name = os.path.splitext(filename)[0]

        # 지역 판별
        if 'ulsan' in name.lower():
            region = '울산'
        elif 'incheon' in name.lower():
            region = '인천'
        elif 'mokpo' in name.lower():
            region = '목포'
        else:
            region = '목포'  # 기본값

        print(f"[{name}] 처리 중... (지역: {region})")

        # 데이터 로드
        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        mmsi = df['mmsi'].iloc[0]
        shiptype = df['shiptype'].iloc[0] if 'shiptype' in df.columns else None
        length = df['length'].iloc[0] if 'length' in df.columns else None

        print(f"  MMSI: {mmsi}, 선종: {shiptype}, 길이: {length}m")

        try:
            # V13 예측 실행 (보정 적용)
            result = predict_trajectory_v13(
                input_data=df,
                region=region,
                shiptype=shiptype,
                length=length,
                device='cuda',
                interpolate=False,
                enable_correction=True
            )

            # 보정 정보 출력
            if result.get('correction_applied'):
                avg_dist = np.mean(result.get('correction_distance_km', [0]))
                max_dist = np.max(result.get('correction_distance_km', [0]))
                print(f"  -> 보정 적용됨: 평균 {avg_dist:.3f}km, 최대 {max_dist:.3f}km")
            else:
                print(f"  -> 보정 미적용")

            # 지도 생성
            m = create_comparison_map(df, result, name, mmsi, region)

            # 저장
            html_file = os.path.join(RESULT_DIR, f'{name}_v13.html')
            m.save(html_file)
            print(f"  -> 저장: {html_file}")

        except Exception as e:
            print(f"  -> 오류: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("테스트 완료!")
    print(f"결과 폴더: {RESULT_DIR}")
    print("=" * 60)

    # 첫 번째 결과 열기
    if test_files:
        first_name = os.path.splitext(os.path.basename(test_files[0]))[0]
        first_result = os.path.join(RESULT_DIR, f'{first_name}_v13.html')
        if os.path.exists(first_result):
            webbrowser.open('file://' + os.path.realpath(first_result))


def build_track_grid(region='울산'):
    """과거 항적 그리드 생성 유틸리티"""
    print(f"[{region}] 과거 항적 그리드 생성 중...")

    predictor = TrajectoryPredictorV13(region=region, enable_correction=True)
    predictor.build_track_grid()

    print("완료!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--build-grid':
        region = sys.argv[2] if len(sys.argv) > 2 else '울산'
        build_track_grid(region)
    else:
        main()
