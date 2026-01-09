# -*- coding: utf-8 -*-
"""
V5 추론 테스트 및 시각화 HTML 생성
==================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 설정
DATA_FOLDER = "G:/NIA_ai_project/항적데이터 추출/여수"
TRANSITION_FOLDER = "area_transition_results"
MODEL_DIR = "global_model_v5"
TARGET_MMSI = 440315060


class GridSequenceModel(nn.Module):
    def __init__(self, num_grids, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.grid_embed = nn.Embedding(num_grids + 1, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_grids)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x_emb = self.grid_embed(x) + self.pos_embed(pos)
        x_emb = self.dropout(x_emb)
        out = self.transformer(x_emb)
        return self.fc(out[:, -1, :])


def grid_id_to_latlon(grid_id, config):
    """격자 ID → 위경도 중심점"""
    num_cols = int(config['num_cols'])
    grid_row = grid_id // num_cols
    grid_col = grid_id % num_cols
    lat = float(config['lat_min']) + (grid_row + 0.5) * float(config['grid_size'])
    lon = float(config['lon_min']) + (grid_col + 0.5) * float(config['grid_size'])
    return lat, lon


def latlon_to_grid_id(lat, lon, config):
    """위경도 → 격자 ID"""
    grid_size = float(config['grid_size'])
    lat_min = float(config['lat_min'])
    lon_min = float(config['lon_min'])
    num_rows = int(config['num_rows'])
    num_cols = int(config['num_cols'])

    row = int((lat - lat_min) / grid_size)
    col = int((lon - lon_min) / grid_size)
    row = max(0, min(row, num_rows - 1))
    col = max(0, min(col, num_cols - 1))
    return row * num_cols + col


def interpolate_1min(df):
    """1분 간격 보간"""
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

    return pd.DataFrame({
        'datetime': time_range,
        'lat': np.interp(target_t, df['_t'].values, df['lat'].values),
        'lon': np.interp(target_t, df['_t'].values, df['lon'].values),
        'sog': np.interp(target_t, df['_t'].values, df['sog'].values),
    })


def generate_html(actual_track, pred_track, history_track, mmsi, output_path):
    """Folium 스타일 HTML 생성"""

    # 중심점 계산
    all_lats = [p[0] for p in actual_track + pred_track + history_track]
    all_lons = [p[1] for p in actual_track + pred_track + history_track]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # 실제 항적 좌표
    actual_coords = [[p[0], p[1]] for p in actual_track]
    pred_coords = [[p[0], p[1]] for p in pred_track]
    history_coords = [[p[0], p[1]] for p in history_track]

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>V5 LLM 스타일 항적 예측 - MMSI {mmsi}</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        .info-box {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        }}
        .info-box h3 {{ margin: 0 0 10px 0; }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-line {{
            width: 30px;
            height: 4px;
            margin-right: 10px;
            border-radius: 2px;
        }}
        .history {{ background: #3388ff; }}
        .actual {{ background: #00aa00; }}
        .predicted {{ background: #ff4444; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3>V5 LLM 스타일 예측</h3>
        <p><strong>MMSI:</strong> {mmsi}</p>
        <p><strong>입력:</strong> {len(history_track)}분 (과거 항적)</p>
        <p><strong>예측:</strong> {len(pred_track)}분</p>
        <hr>
        <div class="legend-item">
            <div class="legend-line history"></div>
            <span>입력 항적 (과거)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line actual"></div>
            <span>실제 항적 (정답)</span>
        </div>
        <div class="legend-item">
            <div class="legend-line predicted"></div>
            <span>예측 항적</span>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 13);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap'
        }}).addTo(map);

        // 입력 항적 (파란색)
        var historyCoords = {history_coords};
        var historyLine = L.polyline(historyCoords, {{
            color: '#3388ff',
            weight: 4,
            opacity: 0.8
        }}).addTo(map);

        // 실제 항적 (녹색)
        var actualCoords = {actual_coords};
        var actualLine = L.polyline(actualCoords, {{
            color: '#00aa00',
            weight: 4,
            opacity: 0.8
        }}).addTo(map);

        // 예측 항적 (빨간색)
        var predCoords = {pred_coords};
        var predLine = L.polyline(predCoords, {{
            color: '#ff4444',
            weight: 4,
            opacity: 0.8,
            dashArray: '10, 5'
        }}).addTo(map);

        // 시작점 마커
        if (historyCoords.length > 0) {{
            L.circleMarker(historyCoords[0], {{
                radius: 8,
                color: '#3388ff',
                fillColor: '#3388ff',
                fillOpacity: 1
            }}).addTo(map).bindPopup('입력 시작점');
        }}

        // 예측 시작점 (입력 끝점)
        if (historyCoords.length > 0) {{
            L.circleMarker(historyCoords[historyCoords.length - 1], {{
                radius: 10,
                color: '#ff8800',
                fillColor: '#ff8800',
                fillOpacity: 1
            }}).addTo(map).bindPopup('예측 시작점');
        }}

        // 실제 도착점
        if (actualCoords.length > 0) {{
            L.circleMarker(actualCoords[actualCoords.length - 1], {{
                radius: 8,
                color: '#00aa00',
                fillColor: '#00aa00',
                fillOpacity: 1
            }}).addTo(map).bindPopup('실제 도착점');
        }}

        // 예측 도착점
        if (predCoords.length > 0) {{
            L.circleMarker(predCoords[predCoords.length - 1], {{
                radius: 8,
                color: '#ff4444',
                fillColor: '#ff4444',
                fillOpacity: 1
            }}).addTo(map).bindPopup('예측 도착점');
        }}

        // 지도 범위 자동 조정
        var allCoords = historyCoords.concat(actualCoords).concat(predCoords);
        if (allCoords.length > 0) {{
            map.fitBounds(allCoords);
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML 저장: {output_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print(f"V5 추론 테스트 - MMSI {TARGET_MMSI}")
    print("=" * 60)

    # 1. 모델 및 설정 로드
    print("\n[1] 모델 로드")
    model_path = os.path.join(MODEL_DIR, str(TARGET_MMSI), 'model.pth')
    config_path = os.path.join(MODEL_DIR, str(TARGET_MMSI), 'config.npz')

    config = np.load(config_path, allow_pickle=True)
    seq_len = int(config['seq_len'])
    total_grids = int(config['total_grids'])
    min_sog = float(config['min_sog'])

    print(f"  seq_len: {seq_len}")
    print(f"  total_grids: {total_grids}")
    print(f"  min_sog: {min_sog}")

    model = GridSequenceModel(num_grids=total_grids, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  모델 로드 완료")

    # 2. 테스트 데이터 로드
    print("\n[2] 테스트 데이터 로드")
    transition_files = [f for f in os.listdir(TRANSITION_FOLDER) if f.endswith('.csv')][:50]
    all_trans = pd.concat([pd.read_csv(os.path.join(TRANSITION_FOLDER, f)) for f in transition_files])

    # 해당 MMSI의 모든 항적 시도
    target_trans = all_trans[all_trans['mmsi'] == TARGET_MMSI]
    print(f"  MMSI {TARGET_MMSI} 전이 정보: {len(target_trans)} 건")

    test_df = None
    for idx, row in target_trans.iterrows():
        s_area = row['start_area']
        e_area = row['end_area']
        start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

        filename = f"{TARGET_MMSI}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, encoding='cp949')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            for c in ['lat', 'lon', 'sog']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['datetime', 'lat', 'lon', 'sog'])

            # 보간 먼저 (속력 필터 전)
            intp = interpolate_1min(df)
            if intp is None:
                continue

            # 속력 필터 (8노트 이상)
            intp_filtered = intp[intp['sog'] >= min_sog].reset_index(drop=True)

            # 충분한 길이 확인 (입력 50 + 예측 30 = 80 이상)
            if len(intp_filtered) >= seq_len + 30:
                test_df = intp_filtered
                print(f"  테스트 파일: {filename}")
                print(f"  보간 후 길이: {len(intp)}")
                print(f"  속력 필터 후: {len(test_df)}")
                break
        except Exception as e:
            continue

    if test_df is None:
        print("[ERROR] 테스트 데이터 없음 - 조건을 완화합니다")
        # 조건 완화: 속력 필터 없이 시도
        for idx, row in target_trans.iterrows():
            s_area = row['start_area']
            e_area = row['end_area']
            start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
            end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

            filename = f"{TARGET_MMSI}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
            filepath = os.path.join(DATA_FOLDER, filename)

            if not os.path.exists(filepath):
                continue

            try:
                df = pd.read_csv(filepath, encoding='cp949')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                for c in ['lat', 'lon', 'sog']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna(subset=['datetime', 'lat', 'lon', 'sog'])

                intp = interpolate_1min(df)
                if intp is not None and len(intp) >= seq_len + 30:
                    test_df = intp
                    print(f"  테스트 파일 (완화): {filename}")
                    print(f"  데이터 길이: {len(test_df)}")
                    break
            except:
                continue

    if test_df is None:
        print("[ERROR] 테스트 데이터를 찾을 수 없습니다")
        return

    # 3. 입력 시퀀스 준비 및 예측
    print("\n[3] 예측 수행")
    n_predict = 30  # 30분 예측

    # 입력: 처음 seq_len개
    history_df = test_df.iloc[:seq_len]
    actual_future_df = test_df.iloc[seq_len:seq_len + n_predict]

    # 격자 ID 시퀀스 생성
    grid_ids = []
    for _, row in history_df.iterrows():
        gid = latlon_to_grid_id(row['lat'], row['lon'], config)
        grid_ids.append(gid)

    # Autoregressive 예측
    pred_grid_ids = []
    current_seq = np.array(grid_ids, dtype=np.int64)

    with torch.no_grad():
        for _ in range(n_predict):
            x = torch.from_numpy(current_seq[-seq_len:]).unsqueeze(0).to(device)
            logits = model(x)
            next_gid = logits.argmax(dim=1).item()
            pred_grid_ids.append(next_gid)
            current_seq = np.append(current_seq, next_gid)

    # 예측 격자 ID → 위경도
    pred_track = []
    for gid in pred_grid_ids:
        lat, lon = grid_id_to_latlon(gid, config)
        pred_track.append((lat, lon))

    # 실제 항적
    actual_track = [(row['lat'], row['lon']) for _, row in actual_future_df.iterrows()]

    # 입력 항적
    history_track = [(row['lat'], row['lon']) for _, row in history_df.iterrows()]

    print(f"  입력: {len(history_track)}개 포인트")
    print(f"  예측: {len(pred_track)}개 포인트")
    print(f"  실제: {len(actual_track)}개 포인트")

    # 4. 오차 계산
    print("\n[4] 오차 분석")
    errors = []
    for i, (pred, actual) in enumerate(zip(pred_track, actual_track)):
        # Haversine 거리 (km)
        lat1, lon1 = np.radians(pred[0]), np.radians(pred[1])
        lat2, lon2 = np.radians(actual[0]), np.radians(actual[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_km = 6371 * c
        errors.append(dist_km)

    print(f"  평균 오차: {np.mean(errors):.3f} km")
    print(f"  최대 오차: {np.max(errors):.3f} km")
    print(f"  10분 시점 오차: {errors[9]:.3f} km")
    print(f"  20분 시점 오차: {errors[19]:.3f} km")
    print(f"  30분 시점 오차: {errors[29]:.3f} km")

    # 5. HTML 생성
    print("\n[5] HTML 생성")
    output_path = f"prediction_v5_mmsi_{TARGET_MMSI}.html"
    generate_html(actual_track, pred_track, history_track, TARGET_MMSI, output_path)

    print("\n" + "=" * 60)
    print("완료!")
    print(f"  HTML: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
