# -*- coding: utf-8 -*-
"""
V7 멀티 임베딩 추론 및 시각화 (스무딩 적용)
==========================================
- H3 + SOG + COG 멀티 임베딩
- 예측 경로 Bezier 스무딩
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import h3
import pickle
import warnings
from scipy.interpolate import splprep, splev

warnings.filterwarnings("ignore")

# 설정
DATA_FOLDER = "G:/NIA_ai_project/항적데이터 추출/여수"
TRANSITION_FOLDER = "area_transition_results"
MODEL_DIR = "global_model_v7_multi"
H3_RESOLUTION = 9
MIN_SOG = 8.0
MAX_SOG = 21.0
SEQ_LEN = 50
NUM_SOG_BINS = 14
NUM_COG_BINS = 24


def sog_to_bin(sog):
    """SOG를 구간 인덱스로 변환"""
    if sog < MIN_SOG:
        return 0
    bin_idx = int(sog - MIN_SOG)
    return min(bin_idx, NUM_SOG_BINS - 1)


def cog_to_bin(cog):
    """COG를 15도 단위 구간 인덱스로 변환"""
    cog = cog % 360
    return int(cog / 15) % NUM_COG_BINS


class H3MultiEmbedModel(nn.Module):
    """H3 + SOG + COG 멀티 임베딩 Transformer 모델"""
    def __init__(self, num_cells, num_sog_bins=NUM_SOG_BINS, num_cog_bins=NUM_COG_BINS,
                 embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.sog_embed = nn.Embedding(num_sog_bins, embed_dim)
        self.cog_embed = nn.Embedding(num_cog_bins, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_cells)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_ids, sog_bins, cog_bins):
        B, T = cell_ids.shape
        pos = torch.arange(T, device=cell_ids.device).unsqueeze(0).expand(B, T)
        x = self.cell_embed(cell_ids) + self.sog_embed(sog_bins) + self.cog_embed(cog_bins) + self.pos_embed(pos)
        x = self.dropout(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


def interpolate_1min(df):
    """1분 간격 보간 (COG 포함)"""
    if len(df) < 2:
        return None
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates('datetime')

    start_time = df['datetime'].iloc[0]
    end_time = df['datetime'].iloc[-1]
    time_range = pd.date_range(start=start_time, end=end_time, freq='1min')

    if len(time_range) < 2:
        return None

    df['_t'] = (df['datetime'] - start_time).dt.total_seconds()
    target_t = (time_range - start_time).total_seconds().values

    # COG 보간 (각도 특성 고려)
    cog_rad = np.radians(df['cog'].values)
    cog_sin = np.interp(target_t, df['_t'].values, np.sin(cog_rad))
    cog_cos = np.interp(target_t, df['_t'].values, np.cos(cog_rad))
    cog_interp = np.degrees(np.arctan2(cog_sin, cog_cos)) % 360

    return pd.DataFrame({
        'datetime': time_range,
        'lat': np.interp(target_t, df['_t'].values, df['lat'].values),
        'lon': np.interp(target_t, df['_t'].values, df['lon'].values),
        'sog': np.interp(target_t, df['_t'].values, df['sog'].values),
        'cog': cog_interp,
    })


def haversine(lat1, lon1, lat2, lon2):
    """두 점 사이 거리 (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_adjacent_cells(cell):
    """현재 셀의 인접 셀들 반환 (자신 포함)"""
    neighbors = h3.grid_disk(cell, 1)  # k=1: 자신 + 6개 인접 셀
    return set(neighbors)


def select_best_adjacent_cell(current_cell, logits, idx_to_cell, cell_to_idx):
    """
    모델 출력에서 인접 셀 중 가장 높은 확률의 셀 선택
    인접 셀 중 학습된 셀이 없으면:
    1. 모델 최고 확률 셀 방향으로 인접 셀 중 이동
    """
    adjacent = get_adjacent_cells(current_cell)

    # logits를 확률로 변환
    probs = torch.softmax(logits, dim=-1).squeeze()

    best_cell = None
    best_prob = -1

    # 1차: 인접 셀 중 학습된 셀에서 선택
    for adj_cell in adjacent:
        if adj_cell in cell_to_idx:
            idx = cell_to_idx[adj_cell]
            prob = probs[idx].item()
            if prob > best_prob:
                best_prob = prob
                best_cell = adj_cell

    # 2차: 인접 셀 중 학습된 셀이 없으면, 최고 확률 셀 방향으로 인접 셀 이동
    if best_cell is None:
        top_idx = probs.argmax().item()
        target_cell = idx_to_cell[top_idx]
        target_lat, target_lon = h3.cell_to_latlng(target_cell)
        current_lat, current_lon = h3.cell_to_latlng(current_cell)

        # 인접 셀 중 목표 방향에 가장 가까운 셀 선택
        min_dist = float('inf')
        for adj_cell in adjacent:
            if adj_cell == current_cell:
                continue  # 자기 자신 제외
            adj_lat, adj_lon = h3.cell_to_latlng(adj_cell)
            dist = haversine(adj_lat, adj_lon, target_lat, target_lon)
            if dist < min_dist:
                min_dist = dist
                best_cell = adj_cell

        # 그래도 없으면 현재 셀 유지
        if best_cell is None:
            best_cell = current_cell

    return best_cell


def smooth_path(points, num_smooth=100):
    """경로를 스플라인으로 스무딩"""
    if len(points) < 4:
        return points

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    try:
        # B-스플라인 피팅
        tck, u = splprep([lats, lons], s=0.0001, k=min(3, len(points)-1))
        u_new = np.linspace(0, 1, num_smooth)
        smooth_lats, smooth_lons = splev(u_new, tck)
        return list(zip(smooth_lats, smooth_lons))
    except:
        # 스플라인 실패시 원본 반환
        return points


def generate_html(mmsi, history_points, actual_points, predicted_points,
                  predicted_smooth, h3_cells_predicted, output_path):
    """HTML 시각화 생성 (스무딩 경로 포함)"""

    all_lats = [p[0] for p in history_points + actual_points + predicted_points]
    all_lons = [p[1] for p in history_points + actual_points + predicted_points]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Hexagon 경계 생성
    hex_boundaries_js = []
    for cell in h3_cells_predicted:
        boundary = h3.cell_to_boundary(cell)
        coords = [[lat, lon] for lat, lon in boundary]
        hex_boundaries_js.append(coords)

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>V7 Multi-Embed 예측 - MMSI {mmsi}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        .info-box {{
            position: absolute;
            top: 10px;
            left: 50px;
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-width: 380px;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 10px; }}
        .legend-hex {{ width: 20px; height: 20px; margin-right: 10px; background: rgba(255,100,100,0.2); border: 2px solid rgba(255,100,100,0.5); }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3 style="margin-top:0;">V7 Multi-Embed 예측</h3>
        <p><strong>MMSI:</strong> {mmsi}</p>
        <p><strong>입력:</strong> H3 셀 + SOG(1노트) + COG(15도)</p>
        <p><strong>H3 Resolution:</strong> {H3_RESOLUTION} (~700m)</p>
        <p><strong>시퀀스 길이:</strong> {SEQ_LEN}분</p>
        <hr>
        <p><strong>히스토리:</strong> {len(history_points)}개 점</p>
        <p><strong>실제 경로:</strong> {len(actual_points)}개 점</p>
        <p><strong>예측 (스무딩):</strong> {len(predicted_smooth)}개 점</p>
    </div>
    <div class="legend">
        <h4 style="margin-top:0;">범례</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: blue;"></div>
            <span>히스토리 (입력)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: green;"></div>
            <span>실제 경로</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: red; height: 3px;"></div>
            <span>예측 (스무딩)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: orange; border-style: dashed;"></div>
            <span>예측 (원본)</span>
        </div>
        <div class="legend-item">
            <div class="legend-hex"></div>
            <span>예측 H3 셀</span>
        </div>
    </div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 13);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        // 히스토리 경로 (파란색)
        var historyPoints = {[list(p) for p in history_points]};
        L.polyline(historyPoints, {{
            color: 'blue',
            weight: 3,
            opacity: 0.8
        }}).addTo(map);

        // 실제 경로 (녹색)
        var actualPoints = {[list(p) for p in actual_points]};
        L.polyline(actualPoints, {{
            color: 'green',
            weight: 3,
            opacity: 0.8
        }}).addTo(map);

        // 예측 원본 경로 (주황색 점선)
        var predictedPoints = {[list(p) for p in predicted_points]};
        L.polyline(predictedPoints, {{
            color: 'orange',
            weight: 2,
            opacity: 0.5,
            dashArray: '5, 5'
        }}).addTo(map);

        // 예측 스무딩 경로 (빨간색 실선)
        var predictedSmooth = {[list(p) for p in predicted_smooth]};
        L.polyline(predictedSmooth, {{
            color: 'red',
            weight: 3,
            opacity: 0.9
        }}).addTo(map);

        // 예측 H3 Hexagon (연한 빨간색)
        var hexBoundaries = {hex_boundaries_js};
        hexBoundaries.forEach(function(coords) {{
            L.polygon(coords, {{
                color: 'rgba(255,100,100,0.5)',
                weight: 1,
                fillColor: 'rgba(255,100,100,0.15)',
                fillOpacity: 0.15
            }}).addTo(map);
        }});

        // 마커들
        if (historyPoints.length > 0) {{
            L.circleMarker(historyPoints[0], {{
                radius: 8, color: 'blue', fillColor: 'white', fillOpacity: 1
            }}).addTo(map).bindPopup('히스토리 시작');
        }}

        if (predictedSmooth.length > 0) {{
            L.circleMarker(predictedSmooth[0], {{
                radius: 8, color: 'orange', fillColor: 'orange', fillOpacity: 1
            }}).addTo(map).bindPopup('예측 시작');

            L.circleMarker(predictedSmooth[predictedSmooth.length - 1], {{
                radius: 8, color: 'red', fillColor: 'red', fillOpacity: 1
            }}).addTo(map).bindPopup('예측 종료 (30분)');
        }}

        if (actualPoints.length > 0) {{
            L.circleMarker(actualPoints[actualPoints.length - 1], {{
                radius: 8, color: 'green', fillColor: 'green', fillOpacity: 1
            }}).addTo(map).bindPopup('실제 종료');
        }}

        // 지도 범위 맞추기
        var allPoints = historyPoints.concat(actualPoints).concat(predictedSmooth);
        if (allPoints.length > 0) {{
            map.fitBounds(allPoints);
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  HTML 저장: {output_path}")


def run_inference(mmsi):
    """특정 MMSI에 대해 추론 실행"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"MMSI {mmsi} 추론")
    print(f"{'='*60}")

    model_path = os.path.join(MODEL_DIR, str(mmsi))
    if not os.path.exists(model_path):
        print(f"  [ERROR] 모델 없음: {model_path}")
        return

    # 설정 로드
    config = np.load(os.path.join(model_path, 'config.npz'), allow_pickle=True)
    num_cells = int(config['num_cells'])
    print(f"  고유 H3 셀 수: {num_cells}")

    # 셀 매핑 로드
    with open(os.path.join(model_path, 'cell_mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)
    cell_to_idx = mapping['cell_to_idx']
    idx_to_cell = mapping['idx_to_cell']

    # 모델 로드
    model = H3MultiEmbedModel(num_cells=num_cells).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=device))
    model.eval()

    # 테스트 데이터 찾기
    transition_files = [f for f in os.listdir(TRANSITION_FOLDER) if f.endswith('.csv')]
    all_trans = pd.concat([pd.read_csv(os.path.join(TRANSITION_FOLDER, f)) for f in transition_files])
    target_trans = all_trans[all_trans['mmsi'] == mmsi]

    print(f"  전이 정보: {len(target_trans)} 건")

    # 테스트할 파일 찾기
    test_df = None
    checked = 0
    for _, row in target_trans.iterrows():
        s_area = row['start_area']
        e_area = row['end_area']
        start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

        filename = f"{mmsi}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            continue

        checked += 1
        try:
            df = pd.read_csv(filepath, encoding='cp949')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            for c in ['lat', 'lon', 'sog', 'cog']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['datetime', 'lat', 'lon', 'sog', 'cog'])

            df_fast = df[df['sog'] >= MIN_SOG]
            if len(df_fast) < SEQ_LEN + 30:
                continue

            intp = interpolate_1min(df_fast)
            if intp is None or len(intp) < SEQ_LEN + 30:
                continue

            test_df = intp.reset_index(drop=True)
            print(f"  테스트 파일 찾음: {filename[:60]}...")
            break
        except Exception as e:
            continue

        if checked > 100:
            break

    if test_df is None:
        print(f"  [ERROR] 적합한 테스트 데이터 없음")
        return

    print(f"  테스트 데이터: {len(test_df)} 포인트")

    # 히스토리 (입력 시퀀스)
    history = test_df.iloc[:SEQ_LEN]
    history_points = [(row['lat'], row['lon']) for _, row in history.iterrows()]

    # 실제 미래 경로
    actual = test_df.iloc[SEQ_LEN:SEQ_LEN+30]
    actual_points = [(row['lat'], row['lon']) for _, row in actual.iterrows()]

    # H3 셀, SOG, COG 변환
    history_cells = [h3.latlng_to_cell(lat, lon, H3_RESOLUTION) for lat, lon in history_points]
    history_sog = [sog_to_bin(row['sog']) for _, row in history.iterrows()]
    history_cog = [cog_to_bin(row['cog']) for _, row in history.iterrows()]

    # 마지막 SOG, COG (예측시 유지)
    last_sog = test_df.iloc[SEQ_LEN-1]['sog']
    last_cog = test_df.iloc[SEQ_LEN-1]['cog']

    # 예측
    predicted_points = []
    predicted_cells = []
    current_cells = history_cells.copy()
    current_sog_bins = history_sog.copy()
    current_cog_bins = history_cog.copy()

    with torch.no_grad():
        for step in range(30):
            # 현재 시퀀스의 인덱스 변환
            cell_indices = []
            for cell in current_cells[-SEQ_LEN:]:
                if cell in cell_to_idx:
                    cell_indices.append(cell_to_idx[cell])
                else:
                    lat, lon = h3.cell_to_latlng(cell)
                    min_dist = float('inf')
                    nearest_idx = 0
                    for known_cell, idx in cell_to_idx.items():
                        klat, klon = h3.cell_to_latlng(known_cell)
                        dist = haversine(lat, lon, klat, klon)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = idx
                    cell_indices.append(nearest_idx)

            cell_x = torch.tensor([cell_indices], dtype=torch.long, device=device)
            sog_x = torch.tensor([current_sog_bins[-SEQ_LEN:]], dtype=torch.long, device=device)
            cog_x = torch.tensor([current_cog_bins[-SEQ_LEN:]], dtype=torch.long, device=device)

            logits = model(cell_x, sog_x, cog_x)

            # 인접 셀 제약: 현재 셀의 인접 셀 중에서만 선택
            current_cell = current_cells[-1]
            pred_cell = select_best_adjacent_cell(current_cell, logits, idx_to_cell, cell_to_idx)

            pred_lat, pred_lon = h3.cell_to_latlng(pred_cell)
            predicted_points.append((pred_lat, pred_lon))
            predicted_cells.append(pred_cell)

            # 다음 스텝을 위해 업데이트
            current_cells.append(pred_cell)

            # COG 업데이트: 이전 위치에서 현재 위치로의 방향
            if len(predicted_points) >= 2:
                prev_lat, prev_lon = predicted_points[-2]
                dlat = pred_lat - prev_lat
                dlon = pred_lon - prev_lon
                new_cog = np.degrees(np.arctan2(dlon, dlat)) % 360
                current_cog_bins.append(cog_to_bin(new_cog))
            else:
                current_cog_bins.append(current_cog_bins[-1])

            # SOG는 마지막 값 유지
            current_sog_bins.append(sog_to_bin(last_sog))

    # 스무딩 적용
    predicted_smooth = smooth_path(predicted_points, num_smooth=100)

    # 오차 계산
    errors = []
    for i, (pred, act) in enumerate(zip(predicted_points, actual_points)):
        err = haversine(pred[0], pred[1], act[0], act[1])
        errors.append(err)

    print(f"\n  예측 오차:")
    print(f"    평균: {np.mean(errors):.2f} km")
    print(f"    10분 후: {errors[9]:.2f} km")
    print(f"    20분 후: {errors[19]:.2f} km")
    print(f"    30분 후: {errors[29]:.2f} km")

    # HTML 생성
    output_path = f"prediction_v7_multi_{mmsi}.html"
    generate_html(mmsi, history_points, actual_points, predicted_points,
                  predicted_smooth, predicted_cells, output_path)

    return errors


def main():
    print("=" * 60)
    print("V7 Multi-Embed 추론 및 시각화 (스무딩 적용)")
    print("=" * 60)

    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] 모델 디렉토리 없음: {MODEL_DIR}")
        return

    mmsi_list = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    print(f"\n학습된 MMSI: {mmsi_list}")

    for mmsi in mmsi_list:
        try:
            run_inference(int(mmsi))
        except Exception as e:
            print(f"  [ERROR] {mmsi}: {e}")


if __name__ == "__main__":
    main()
