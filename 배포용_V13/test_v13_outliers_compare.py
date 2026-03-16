# -*- coding: utf-8 -*-
"""
V13 이상선박 비교 테스트 - V13 예측 vs 다른 플랫폼 예측
=======================================================
- ais.csv: 이상선박 원본 AIS 데이터 (2026-03-03/04)
- pred_route.csv: 다른 플랫폼 예측 결과
- V13 모델로 동일 시간대 예측 → 두 예측을 함께 시각화
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from tqdm import tqdm

from trajectory_predictor_v13 import (
    TrajectoryPredictorV12, DepthChecker, HistoricalTrackGrid, PathCorrector,
    REGION_BOUNDS, SHIPTYPE_NAMES, LENGTH_NAMES
)

ALL_REGION_BOUNDS = {
    '울산': {'lat_min': 35.0, 'lat_max': 36.0, 'lon_min': 129.0, 'lon_max': 130.0},
    '인천': {'lat_min': 36.5, 'lat_max': 37.8, 'lon_min': 125.5, 'lon_max': 127.0},
    '목포': {'lat_min': 33.5, 'lat_max': 35.7, 'lon_min': 125.2, 'lon_max': 126.8},
    '부산': {'lat_min': 34.8, 'lat_max': 35.5, 'lon_min': 128.5, 'lon_max': 129.5},
    '여수': {'lat_min': 34.0, 'lat_max': 35.0, 'lon_min': 127.0, 'lon_max': 128.0},
    '통영': {'lat_min': 34.5, 'lat_max': 35.2, 'lon_min': 127.8, 'lon_max': 129.0},
    '완도': {'lat_min': 33.8, 'lat_max': 34.8, 'lon_min': 126.0, 'lon_max': 127.5},
    '제주': {'lat_min': 33.0, 'lat_max': 34.0, 'lon_min': 126.0, 'lon_max': 127.5},
    '군산': {'lat_min': 35.5, 'lat_max': 36.5, 'lon_min': 125.5, 'lon_max': 126.8},
    '포항': {'lat_min': 35.5, 'lat_max': 36.5, 'lon_min': 129.0, 'lon_max': 130.0},
    '동해': {'lat_min': 37.0, 'lat_max': 38.0, 'lon_min': 129.0, 'lon_max': 130.0},
}

DISPLAY_LEN = 20  # 20분 예측


def parse_js_coords(s):
    """JS object 형식의 좌표 목록을 파싱"""
    if pd.isna(s) or not s:
        return []
    s = re.sub(r'dateTime: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', r'"dateTime": "\1"', s)
    s = re.sub(r'(\b(?:lat|lon)\b):', r'"\1":', s)
    try:
        return json.loads(s)
    except:
        return []


def interpolate_1min(df):
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


def find_region(lat, lon):
    """좌표로 지역 판별"""
    for region, bounds in ALL_REGION_BOUNDS.items():
        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
            bounds['lon_min'] <= lon <= bounds['lon_max']):
            return region
    return None


def load_depth_grids_for_display(depth_checker, region):
    """수심 격자 데이터 로드 (육지 + 얕은 물) - L.rectangle 표시용"""
    if not depth_checker or not depth_checker.loaded:
        return []
    grids = depth_checker.get_depth_grid_for_display(region=region)
    print(f"  수심 격자 (표시용): {len(grids):,}개 (육지+얕은물)")
    return grids


def process_pred_route_entry(pred_row, ais_df, predictor, corrector):
    """pred_route 한 행을 처리: V13 예측 + 다른 플랫폼 예측 비교"""
    mmsi = pred_row['mmsi']
    base_time = pd.to_datetime(pred_row['data_req_time']).floor('1min')

    # 다른 플랫폼 예측 파싱
    other_coords_raw = parse_js_coords(pred_row['pred_route'])
    if not other_coords_raw:
        return None
    other_coords = np.array([[c['lat'], c['lon']] for c in other_coords_raw])
    other_coords_display = other_coords[:DISPLAY_LEN]

    # 이 선박의 AIS 데이터
    ship_df = ais_df[ais_df['mmsi'] == mmsi].copy()
    if len(ship_df) < 5:
        return None

    input_start = base_time - timedelta(minutes=30)
    input_end = base_time
    actual_end = base_time + timedelta(minutes=DISPLAY_LEN)

    # 입력 구간 (SOG 30노트 이상 비정상 제거)
    input_df = ship_df[(ship_df['datetime'] >= input_start) & (ship_df['datetime'] <= input_end)]
    input_df = input_df[input_df['sog'] < 30]
    if len(input_df) < 5:
        return None

    input_interp = interpolate_1min(input_df)
    if input_interp is None or len(input_interp) < 30:
        return None
    input_interp = input_interp.tail(30).reset_index(drop=True)

    # 현재 위치: base_time 시점 (입력 마지막 포인트)
    current_pos = [input_interp['lat'].iloc[-1], input_interp['lon'].iloc[-1]]
    current_cog = input_interp['cog'].iloc[-1]
    current_sog = input_interp['sog'].iloc[-1]

    # 실제 경로 (현재 위치를 시작점으로 포함하여 과거 경로와 연결)
    actual_df = ship_df[(ship_df['datetime'] > input_end) & (ship_df['datetime'] <= actual_end)]
    actual_coords = None
    if len(actual_df) >= 3:
        start_row = pd.DataFrame({
            'datetime': [input_end],
            'lat': [current_pos[0]], 'lon': [current_pos[1]],
            'sog': [current_sog], 'cog': [current_cog]
        })
        actual_with_start = pd.concat([start_row, actual_df]).sort_values('datetime').drop_duplicates('datetime')
        actual_interp = interpolate_1min(actual_with_start)
        if actual_interp is not None and len(actual_interp) >= DISPLAY_LEN:
            actual_coords = actual_interp.head(DISPLAY_LEN)[['lat', 'lon']].values

    # V13 예측 (입력: base_time까지 30개 → pred는 t+1부터)
    input_data = input_interp[['lat', 'lon', 'sog', 'cog']].values
    result = predictor.predict(input_data)
    pred_full = result['predicted_coords']

    # V13 보정
    if corrector is not None:
        last_position = result['last_position']
        last_cog = input_data[-1, 3]
        corrected_full = corrector.correct_path(
            pred_full.copy(), last_position, last_cog=last_cog,
            shiptype_cat=4, length_cat=0
        )
    else:
        corrected_full = pred_full.copy()

    # 현재 위치를 시작점으로 추가 (보정 후, 표시용 - 과거 경로와 연결)
    current_pos_arr = np.array([[current_pos[0], current_pos[1]]])
    v13_original = np.concatenate([current_pos_arr, pred_full[:DISPLAY_LEN-1]], axis=0)
    v13_corrected = np.concatenate([current_pos_arr, corrected_full[:DISPLAY_LEN-1]], axis=0)

    # 오차 계산 (끝점)
    li = DISPLAY_LEN - 1
    error_v13 = error_other = 0
    if actual_coords is not None and len(actual_coords) >= DISPLAY_LEN:
        # V13 보정 후 오차
        dlat = v13_corrected[li, 0] - actual_coords[li, 0]
        dlon = v13_corrected[li, 1] - actual_coords[li, 1]
        error_v13 = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)
        # 다른 플랫폼 오차
        if len(other_coords_display) >= DISPLAY_LEN:
            dlat = other_coords_display[li, 0] - actual_coords[li, 0]
            dlon = other_coords_display[li, 1] - actual_coords[li, 1]
            error_other = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

    # anomaly 정보
    anomaly_str = str(pred_row.get('anomaly_details', ''))

    # 과거 30분 SOG/COG 타임시리즈
    input_history = []
    for i in range(len(input_interp)):
        row_dt = input_interp['datetime'].iloc[i]
        input_history.append({
            'time': row_dt.strftime('%H:%M'),
            'sog': round(float(input_interp['sog'].iloc[i]), 1),
            'cog': round(float(input_interp['cog'].iloc[i]), 1),
            'lat': round(float(input_interp['lat'].iloc[i]), 5),
            'lon': round(float(input_interp['lon'].iloc[i]), 5),
        })

    # 보간 전 원본 AIS 데이터의 SOG/COG도 포함 (이상값 확인용)
    raw_input_df = ship_df[(ship_df['datetime'] >= input_start) & (ship_df['datetime'] <= input_end)].sort_values('datetime')
    raw_history = []
    for _, r in raw_input_df.iterrows():
        raw_history.append({
            'time': r['datetime'].strftime('%H:%M:%S'),
            'sog': round(float(r['sog']), 1),
            'cog': round(float(r['cog']), 1),
            'lat': round(float(r['lat']), 5),
            'lon': round(float(r['lon']), 5),
        })

    return {
        'mmsi': int(mmsi),
        'base_time': base_time,
        'current_pos': current_pos,
        'cog': current_cog,
        'sog': current_sog,
        'input_coords': input_interp[['lat', 'lon']].values,
        'input_history': input_history,
        'raw_history': raw_history,
        'v13_original': v13_original,
        'v13_corrected': v13_corrected,
        'other_coords': other_coords_display,
        'actual_coords': actual_coords,
        'error_v13': error_v13,
        'error_other': error_other,
        'anomaly': anomaly_str[:200],
    }


def create_compare_html(ships_data, output_path, title, base_time_str, depth_grids=None):
    if not ships_data:
        return False

    all_lats = [s['current_pos'][0] for s in ships_data]
    all_lons = [s['current_pos'][1] for s in ships_data]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    ships_json = []
    for ship in ships_data:
        ships_json.append({
            'mmsi': ship['mmsi'],
            'current_pos': [float(ship['current_pos'][0]), float(ship['current_pos'][1])],
            'cog': float(ship['cog']),
            'sog': float(ship['sog']),
            'input': [[float(c[0]), float(c[1])] for c in ship['input_coords']],
            'v13': [[float(c[0]), float(c[1])] for c in ship['v13_corrected']],
            'v13_orig': [[float(c[0]), float(c[1])] for c in ship['v13_original']],
            'other': [[float(c[0]), float(c[1])] for c in ship['other_coords']],
            'actual': [[float(c[0]), float(c[1])] for c in ship['actual_coords']] if ship['actual_coords'] is not None else [],
            'error_v13': float(ship['error_v13']),
            'error_other': float(ship['error_other']),
            'input_history': ship.get('input_history', []),
            'raw_history': ship.get('raw_history', []),
        })

    ships_json_str = json.dumps(ships_json)

    # 수심 격자 JSON
    depth_json_str = json.dumps(depth_grids) if depth_grids else "[]"

    errs_v13 = [s['error_v13'] for s in ships_data if s['error_v13'] > 0]
    errs_other = [s['error_other'] for s in ships_data if s['error_other'] > 0]
    avg_v13 = np.mean(errs_v13) if errs_v13 else 0
    avg_other = np.mean(errs_other) if errs_other else 0

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Malgun Gothic', sans-serif; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; padding: 15px 20px;
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
        }}
        .header h1 {{ font-size: 18px; font-weight: 500; }}
        .header-stats {{ display: flex; gap: 15px; font-size: 13px; }}
        .header-stats .stat {{ background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 4px; }}
        .stat-v13 {{ color: #e74c3c; font-weight: bold; }}
        .stat-other {{ color: #3498db; font-weight: bold; }}
        .depth-control {{ position: absolute; top: 85px; left: 10px; z-index: 1000; }}
        .depth-btn {{ padding: 8px 12px; background: #fff; border: 2px solid #666; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: 'Malgun Gothic', sans-serif; }}
        .depth-btn.on {{ background: #e74c3c; color: #fff; border-color: #c0392b; }}
        #map {{ width: 100%; height: calc(100vh - 120px); }}
        .info-panel {{
            position: absolute; top: 80px; right: 10px; background: white;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000; min-width: 380px; max-width: 420px; max-height: calc(100vh - 120px);
            overflow-y: auto; display: none;
        }}
        .info-panel.show {{ display: block; }}
        .info-panel h3 {{ margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #3498db; }}
        .info-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; font-size: 13px; }}
        .info-label {{ color: #666; }}
        .info-value {{ font-weight: bold; }}
        .close-btn {{ position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 20px; cursor: pointer; color: #999; }}
        .better {{ color: #27ae60; font-weight: bold; }}
        .worse {{ color: #e74c3c; font-weight: bold; }}
        .tab-btns {{ display: flex; gap: 5px; margin: 12px 0 8px 0; }}
        .tab-btn {{ padding: 5px 12px; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5; cursor: pointer; font-size: 12px; }}
        .tab-btn.active {{ background: #3498db; color: white; border-color: #3498db; }}
        .history-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
        .history-table th {{ background: #f0f0f0; padding: 4px 6px; text-align: center; border: 1px solid #ddd; position: sticky; top: 0; }}
        .history-table td {{ padding: 3px 6px; text-align: center; border: 1px solid #eee; }}
        .history-table tr:nth-child(even) {{ background: #fafafa; }}
        .history-table .sog-high {{ background: #ffe0e0; color: #c0392b; font-weight: bold; }}
        .history-table .sog-low {{ background: #e0e0ff; color: #2980b9; }}
        .history-wrap {{ max-height: 300px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .legend {{ background: white; padding: 12px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 11px; }}
        .legend-title {{ font-weight: bold; margin-bottom: 8px; font-size: 13px; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-line {{ width: 25px; height: 3px; margin-right: 8px; border-radius: 2px; }}
        .bottom-bar {{
            background: #2c3e50; color: white; padding: 8px 20px;
            display: flex; justify-content: space-between; align-items: center; font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="header-stats">
            <div class="stat">이상선박: <b>{len(ships_data)}</b>척</div>
            <div class="stat">V13: <span class="stat-v13">{avg_v13:.2f} km</span></div>
            <div class="stat">기존: <span class="stat-other">{avg_other:.2f} km</span></div>
        </div>
    </div>
    <div id="map"></div>
    <div class="info-panel" id="infoPanel">
        <button class="close-btn" onclick="closePanel()">&times;</button>
        <h3 id="panelTitle">선박 정보</h3>
        <div class="info-row"><span class="info-label">MMSI</span><span class="info-value" id="infoMmsi">-</span></div>
        <div class="info-row"><span class="info-label">현재 속력</span><span class="info-value" id="infoSog">-</span></div>
        <div class="info-row"><span class="info-label">V13 오차</span><span class="info-value" id="errorV13">-</span></div>
        <div class="info-row"><span class="info-label">기존 오차</span><span class="info-value" id="errorOther">-</span></div>
        <div class="info-row"><span class="info-label">차이</span><span class="info-value" id="errorDiff">-</span></div>
        <div class="tab-btns">
            <div class="tab-btn active" onclick="showTab('interp')">보간 데이터</div>
            <div class="tab-btn" onclick="showTab('raw')">원본 AIS</div>
        </div>
        <div class="history-wrap" id="historyWrap"></div>
    </div>
    <div class="bottom-bar">
        <div>이상선박 비교 | V13(빨강) vs 기존(파랑) vs 실제(검정점선)</div>
        <div>선박 클릭 시 상세 정보</div>
    </div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            subdomains: 'abcd'
        }}).addTo(map);
        var shipsData = {ships_json_str};
        var depthGrids = {depth_json_str};
        var shipMarkers = {{}};
        var currentPathLayers = null;
        var selectedMmsi = null;
        var depthLayer = null;
        var depthVisible = false;

        // 수심 격자 레이어 생성
        function createDepthLayer() {{
            var layer = L.layerGroup();
            var gridSize = 0.025;
            depthGrids.forEach(function(g) {{
                var bounds = [
                    [g.lat - gridSize/2, g.lon - gridSize/2],
                    [g.lat + gridSize/2, g.lon + gridSize/2]
                ];
                var color, fillOpacity;
                if (g.type === 'land') {{
                    color = '#8B4513';
                    fillOpacity = 0.7;
                }} else {{
                    color = '#FFD700';
                    fillOpacity = 0.5;
                }}
                var rect = L.rectangle(bounds, {{
                    color: color, weight: 1, fillColor: color, fillOpacity: fillOpacity
                }}).bindPopup('수심: ' + g.depth.toFixed(1) + 'm<br>유형: ' + (g.type === 'land' ? '육지' : '얕은물') + '<br>(' + g.lat.toFixed(3) + ', ' + g.lon.toFixed(3) + ')');
                layer.addLayer(rect);
            }});
            return layer;
        }}
        depthLayer = createDepthLayer();

        function createShipIcon(cog, sel) {{
            var sz = sel ? 32 : 24, cl = sel ? '#e74c3c' : '#3498db';
            var svg = '<svg width="'+sz+'" height="'+sz+'" viewBox="0 0 24 24"><g transform="rotate('+cog+' 12 12)"><polygon points="12,2 4,22 12,18 20,22" fill="'+cl+'" stroke="#fff" stroke-width="2"/></g></svg>';
            return L.divIcon({{ html: svg, className: 'ship-marker', iconSize: [sz,sz], iconAnchor: [sz/2,sz/2] }});
        }}

        shipsData.forEach(function(ship) {{
            // V13 예측 (빨강) - 항상 표시
            if (ship.v13.length > 0) {{
                L.polyline(ship.v13, {{ color: '#e74c3c', weight: 3, opacity: 0.8 }}).addTo(map);
                var li = ship.v13.length - 1;
                L.circleMarker(ship.v13[li], {{ radius: 4, color: '#fff', fillColor: '#e74c3c', fillOpacity: 1, weight: 1 }}).addTo(map);
            }}
            // 기존 플랫폼 예측 (파랑) - 항상 표시
            if (ship.other.length > 0) {{
                L.polyline(ship.other, {{ color: '#3498db', weight: 3, opacity: 0.8 }}).addTo(map);
                var li2 = ship.other.length - 1;
                L.circleMarker(ship.other[li2], {{ radius: 4, color: '#fff', fillColor: '#3498db', fillOpacity: 1, weight: 1 }}).addTo(map);
            }}
            // 실제 경로 (검정 점선) - 항상 표시
            if (ship.actual.length > 0) {{
                L.polyline(ship.actual, {{ color: '#000', weight: 2, opacity: 0.5, dashArray: '6,4' }}).addTo(map);
            }}
            // 선박 마커
            var marker = L.marker(ship.current_pos, {{ icon: createShipIcon(ship.cog, false) }});
            marker.on('click', function() {{ showDetail(ship); }});
            marker.addTo(map);
            shipMarkers[ship.mmsi] = marker;
        }});

        function showDetail(ship) {{
            if (currentPathLayers) currentPathLayers.forEach(l => map.removeLayer(l));
            if (selectedMmsi && shipMarkers[selectedMmsi]) {{
                var p = shipsData.find(s => s.mmsi === selectedMmsi);
                if (p) shipMarkers[selectedMmsi].setIcon(createShipIcon(p.cog, false));
            }}
            selectedMmsi = ship.mmsi;
            shipMarkers[ship.mmsi].setIcon(createShipIcon(ship.cog, true));
            currentPathLayers = [];
            // 보간 데이터 점 (초록 작은 점)
            if (ship.input_history && ship.input_history.length > 0) {{
                var il = L.polyline(ship.input, {{ color: '#2ecc71', weight: 3, opacity: 0.7 }}).addTo(map);
                currentPathLayers.push(il);
                for (var i = 0; i < ship.input_history.length; i++) {{
                    var h = ship.input_history[i];
                    var m = L.circleMarker([h.lat, h.lon], {{ radius: 3, color: '#27ae60', fillColor: '#2ecc71', fillOpacity: 0.9, weight: 1 }})
                        .bindPopup('<b>보간 ' + h.time + '</b><br>SOG: ' + h.sog + ' kn<br>COG: ' + h.cog + '°<br>(' + h.lat + ', ' + h.lon + ')').addTo(map);
                    currentPathLayers.push(m);
                }}
            }}
            // 원본 AIS 점 (주황 삼각형)
            if (ship.raw_history && ship.raw_history.length > 0) {{
                for (var i = 0; i < ship.raw_history.length; i++) {{
                    var r = ship.raw_history[i];
                    var m = L.circleMarker([r.lat, r.lon], {{ radius: 5, color: '#e67e22', fillColor: '#f39c12', fillOpacity: 0.9, weight: 2 }})
                        .bindPopup('<b>원본 ' + r.time + '</b><br>SOG: ' + r.sog + ' kn<br>COG: ' + r.cog + '°<br>(' + r.lat + ', ' + r.lon + ')').addTo(map);
                    currentPathLayers.push(m);
                }}
            }}
            // V13 보정 전 (회색 점선)
            if (ship.v13_orig.length > 0) {{
                var ol = L.polyline(ship.v13_orig, {{ color: '#888', weight: 3, opacity: 0.6, dashArray: '8,6' }}).addTo(map);
                currentPathLayers.push(ol);
            }}
            // 현재 위치
            var cm = L.circleMarker(ship.current_pos, {{ radius: 10, color: '#f39c12', fillColor: '#f1c40f', fillOpacity: 1, weight: 3 }}).bindPopup('MMSI: '+ship.mmsi).addTo(map);
            currentPathLayers.push(cm);
            // 패널
            document.getElementById('infoPanel').classList.add('show');
            document.getElementById('panelTitle').textContent = 'MMSI: ' + ship.mmsi;
            document.getElementById('infoMmsi').textContent = ship.mmsi;
            document.getElementById('infoSog').textContent = ship.sog.toFixed(1) + ' kn';
            var ev = document.getElementById('errorV13');
            ev.textContent = ship.error_v13.toFixed(2) + ' km';
            var eo = document.getElementById('errorOther');
            eo.textContent = ship.error_other.toFixed(2) + ' km';
            var diff = ship.error_other - ship.error_v13;
            var ed = document.getElementById('errorDiff');
            ed.textContent = (diff > 0 ? 'V13이 ' + diff.toFixed(2) + 'km 우수' : '기존이 ' + (-diff).toFixed(2) + 'km 우수');
            ed.className = 'info-value ' + (diff > 0 ? 'better' : 'worse');
            // 히스토리 테이블
            currentShip = ship;
            showTab('interp');
            var all = ship.input.concat(ship.v13).concat(ship.other).concat(ship.actual);
            if (all.length > 0) map.fitBounds(L.polyline(all).getBounds().pad(0.2));
        }}

        var currentShip = null;
        function showTab(tab) {{
            document.querySelectorAll('.tab-btn').forEach(function(b,i) {{
                b.className = 'tab-btn' + (i === (tab==='interp'?0:1) ? ' active' : '');
            }});
            if (!currentShip) return;
            var data = tab === 'interp' ? currentShip.input_history : currentShip.raw_history;
            var html = '<table class="history-table"><thead><tr><th>시간</th><th>SOG(kn)</th><th>COG(°)</th><th>위도</th><th>경도</th></tr></thead><tbody>';
            for (var i = 0; i < data.length; i++) {{
                var d = data[i];
                var sogClass = d.sog >= 30 ? 'sog-high' : (d.sog < 0.5 ? 'sog-low' : '');
                html += '<tr><td>' + d.time + '</td><td class="' + sogClass + '">' + d.sog + '</td><td>' + d.cog + '</td><td>' + d.lat + '</td><td>' + d.lon + '</td></tr>';
            }}
            html += '</tbody></table>';
            document.getElementById('historyWrap').innerHTML = html;
        }}

        function closePanel() {{
            document.getElementById('infoPanel').classList.remove('show');
            if (currentPathLayers) {{ currentPathLayers.forEach(l => map.removeLayer(l)); currentPathLayers = null; }}
            if (selectedMmsi && shipMarkers[selectedMmsi]) {{
                var s = shipsData.find(x => x.mmsi === selectedMmsi);
                if (s) shipMarkers[selectedMmsi].setIcon(createShipIcon(s.cog, false));
            }}
            selectedMmsi = null;
        }}

        // 수심 격자 토글
        var depthControl = L.control({{position: 'topleft'}});
        depthControl.onAdd = function() {{
            var div = L.DomUtil.create('div', 'depth-control');
            div.innerHTML = '<button id="depthToggle" class="depth-btn">격자 ON</button>';
            div.onclick = function(e) {{
                e.stopPropagation();
                depthVisible = !depthVisible;
                var btn = document.getElementById('depthToggle');
                if (depthVisible) {{
                    depthLayer.addTo(map);
                    btn.textContent = '격자 OFF';
                    btn.className = 'depth-btn on';
                }} else {{
                    map.removeLayer(depthLayer);
                    btn.textContent = '격자 ON';
                    btn.className = 'depth-btn';
                }}
            }};
            return div;
        }};
        depthControl.addTo(map);

        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function() {{
            var d = L.DomUtil.create('div', 'legend');
            d.innerHTML = '<div class="legend-title">범례</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#2ecc71;"></div>보간 경로 (초록 점)</div>' +
                '<div class="legend-item"><div style="width:10px;height:10px;border-radius:50%;background:#f39c12;border:2px solid #e67e22;margin-right:8px;"></div>원본 AIS (주황 점)</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#e74c3c;"></div>V13 예측</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#3498db;"></div>기존 플랫폼</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#888;border-top:3px dashed #888;height:0;"></div>V13 보정 전</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#000;border-top:3px dashed #000;height:0;"></div>실제 경로</div>' +
                '<div style="margin-top:6px;border-top:1px solid #ddd;padding-top:6px;">' +
                '<div class="legend-title">수심 격자</div>' +
                '<div class="legend-item"><div style="width:25px;height:12px;background:#8B4513;opacity:0.7;margin-right:8px;border-radius:2px;"></div>육지</div>' +
                '<div class="legend-item"><div style="width:25px;height:12px;background:#FFD700;opacity:0.5;margin-right:8px;border-radius:2px;"></div>얕은물 (&lt;10m)</div></div>';
            return d;
        }};
        legend.addTo(map);
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return True


def create_region_index(output_dir, results, region):
    html = f'''<!DOCTYPE html>
<html><head><title>V13 비교 {region}</title><meta charset="utf-8">
<style>
body {{ font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
.info {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }}
.card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-decoration: none; color: #333; transition: transform 0.2s; }}
.card:hover {{ transform: translateY(-3px); }}
.card .time {{ font-size: 22px; font-weight: bold; color: #e74c3c; }}
.card .stats {{ margin-top: 10px; font-size: 13px; color: #666; line-height: 1.6; }}
.v13 {{ color: #e74c3c; font-weight: bold; }}
.other {{ color: #3498db; font-weight: bold; }}
</style></head><body>
<h1>V13 vs 기존 비교 - {region}</h1>
<div class="info"><p><b>이상선박 비교</b> | <b>예측:</b> {DISPLAY_LEN}분 | <b>파일:</b> {len(results)}개 | <a href="../index.html">전체 목록</a></p></div>
<div class="grid">'''
    for r in results:
        html += f'''<a class="card" href="{r['filename']}"><div class="time">{r['time']}</div>
<div class="stats">선박: {r['ships']}척<br>V13: <span class="v13">{r['avg_v13']:.2f}km</span> | 기존: <span class="other">{r['avg_other']:.2f}km</span></div></a>'''
    html += '</div></body></html>'
    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)


def create_master_index(root_dir, all_results):
    html = f'''<!DOCTYPE html>
<html><head><title>V13 vs 기존 플랫폼 비교</title><meta charset="utf-8">
<style>
body {{ font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
.info {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }}
.card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-decoration: none; color: #333; transition: transform 0.2s; }}
.card:hover {{ transform: translateY(-3px); }}
.card .region {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
.card .stats {{ margin-top: 12px; font-size: 14px; color: #666; line-height: 1.8; }}
.v13 {{ color: #e74c3c; font-weight: bold; }}
.other {{ color: #3498db; font-weight: bold; }}
</style></head><body>
<h1>V13 vs 기존 플랫폼 - 이상선박 예측 비교</h1>
<div class="info"><p><b>이상선박(pred_route.csv) 대상</b> | <b>V13(빨강) vs 기존(파랑) vs 실제(검정)</b></p></div>
<div class="grid">'''
    for region, results in all_results.items():
        if not results:
            continue
        total = sum(r['ships'] for r in results)
        avg_v13 = np.mean([r['avg_v13'] for r in results if r['avg_v13'] > 0]) if results else 0
        avg_other = np.mean([r['avg_other'] for r in results if r['avg_other'] > 0]) if results else 0
        html += f'''<a class="card" href="{region}/index.html"><div class="region">{region}</div>
<div class="stats">HTML: {len(results)}개 | 총 예측: {total}건<br>V13: <span class="v13">{avg_v13:.2f}km</span> | 기존: <span class="other">{avg_other:.2f}km</span></div></a>'''
    html += '</div></body></html>'
    with open(os.path.join(root_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ais_path = r"K:\coding_project\NIA_선박항적예측프로그램\down20260305\ais.csv"
    pred_path = r"K:\coding_project\NIA_선박항적예측프로그램\down20260305\pred_route.csv"

    print("=" * 60)
    print("V13 vs 기존 플랫폼 비교 - 이상선박 (20분 예측)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # AIS 데이터 로드
    print("AIS 데이터 로드...")
    ais_df = pd.read_csv(ais_path)
    if 'date_time' in ais_df.columns and 'datetime' not in ais_df.columns:
        ais_df = ais_df.rename(columns={'date_time': 'datetime'})
    ais_df['datetime'] = pd.to_datetime(ais_df['datetime'])
    # 센티넬 값 제거
    ais_df = ais_df[(ais_df['lat'] > -90) & (ais_df['lat'] < 90) &
                    (ais_df['lon'] > -180) & (ais_df['lon'] < 180) &
                    (ais_df['sog'] < 102)].copy()
    print(f"AIS: {len(ais_df):,}행, {ais_df['mmsi'].nunique()}척")

    # pred_route 로드 및 파싱
    print("pred_route 로드...")
    pred_df = pd.read_csv(pred_path)
    pred_df['req_dt'] = pd.to_datetime(pred_df['data_req_time']).dt.floor('1min')
    # 센티넬 좌표 제외 (region_id=-1에 lat=91 등)
    valid_pred = []
    for _, row in pred_df.iterrows():
        coords = parse_js_coords(row['pred_route'])
        if coords and abs(coords[0]['lat']) < 90:
            valid_pred.append(row)
    pred_df = pd.DataFrame(valid_pred)
    print(f"유효 pred_route: {len(pred_df)}행, {pred_df['mmsi'].nunique()}척")

    # pred_route의 좌표로 지역 판별 → 지역별 그룹핑
    region_pred = {}
    for idx, row in pred_df.iterrows():
        coords = parse_js_coords(row['pred_route'])
        if not coords:
            continue
        lat0, lon0 = coords[0]['lat'], coords[0]['lon']
        region = find_region(lat0, lon0)
        if region is None:
            continue
        if region not in region_pred:
            region_pred[region] = []
        region_pred[region].append(row)

    print(f"\n지역별 pred_route 수:")
    for region in sorted(region_pred.keys()):
        print(f"  {region}: {len(region_pred[region])}건")

    # 지역별 처리
    root_output = os.path.join(base_dir, "예측결과_비교_이상선박")
    all_region_results = {}

    for region in sorted(region_pred.keys()):
        model_dir = os.path.join(base_dir, "models", "v12", region)
        if not os.path.exists(os.path.join(model_dir, "model_best.pth")):
            print(f"\n[SKIP] {region} - 모델 없음")
            continue

        output_dir = os.path.join(root_output, region)
        os.makedirs(output_dir, exist_ok=True)

        pred_rows = region_pred[region]
        bounds = ALL_REGION_BOUNDS[region]

        # 지역 내 AIS 데이터
        ais_region = ais_df[
            (ais_df['lat'] >= bounds['lat_min']) & (ais_df['lat'] <= bounds['lat_max']) &
            (ais_df['lon'] >= bounds['lon_min']) & (ais_df['lon'] <= bounds['lon_max'])
        ].copy()

        print(f"\n{'='*60}")
        print(f"[{region}] pred_route: {len(pred_rows)}건, AIS: {len(ais_region):,}행")

        # 예측기
        predictor = TrajectoryPredictorV12(region=region, model_dir=model_dir, device=device)

        # 보정기 + 수심 격자
        corrector = None
        depth_grids = []
        try:
            depth_checker = DepthChecker(region=region)
            track_grid = HistoricalTrackGrid(grid_resolution=0.0001)
            tg_file = os.path.join(model_dir, 'track_grid.pkl')
            if os.path.exists(tg_file):
                track_grid.load(tg_file)
            corrector = PathCorrector(depth_checker, track_grid)
            depth_grids = load_depth_grids_for_display(depth_checker, region)
        except Exception as e:
            print(f"  [경고] 보정기 실패: {e}")

        # 시간대별 그룹핑 (30분 간격)
        time_groups = {}
        for row in pred_rows:
            req = pd.to_datetime(row['data_req_time']).floor('30min')
            key = req.strftime('%Y%m%d_%H%M')
            if key not in time_groups:
                time_groups[key] = []
            time_groups[key].append(row)

        results = []
        for time_key in tqdm(sorted(time_groups.keys()), desc=f"{region}"):
            group = time_groups[time_key]
            # MMSI별 중복 제거 (같은 시간대에 동일 선박은 최신 1건만)
            mmsi_seen = {}
            for pred_row in group:
                mmsi = pred_row['mmsi']
                mmsi_seen[mmsi] = pred_row  # 같은 MMSI면 마지막 것으로 덮어쓰기
            ships_data = []
            seen_mmsi = set()
            for pred_row in mmsi_seen.values():
                ship_result = process_pred_route_entry(pred_row, ais_region, predictor, corrector)
                if ship_result is not None and ship_result['mmsi'] not in seen_mmsi:
                    ships_data.append(ship_result)
                    seen_mmsi.add(ship_result['mmsi'])

            if ships_data:
                filename = f"compare_{time_key}.html"
                output_path = os.path.join(output_dir, filename)
                time_label = time_key.split('_')[1]
                time_display = f"{time_label[:2]}:{time_label[2:]}"
                title = f"V13 비교 {region} - {time_key.replace('_', ' ')}"
                if create_compare_html(ships_data, output_path, title, time_key, depth_grids):
                    errs_v13 = [s['error_v13'] for s in ships_data if s['error_v13'] > 0]
                    errs_other = [s['error_other'] for s in ships_data if s['error_other'] > 0]
                    results.append({
                        'time': time_display,
                        'filename': filename,
                        'ships': len(ships_data),
                        'avg_v13': np.mean(errs_v13) if errs_v13 else 0,
                        'avg_other': np.mean(errs_other) if errs_other else 0,
                    })

        create_region_index(output_dir, results, region)
        all_region_results[region] = results
        print(f"  -> {region}: {len(results)}개 HTML")

    create_master_index(root_output, all_region_results)
    print(f"\n전체 인덱스: {root_output}/index.html")
    print("\n완료!")


if __name__ == "__main__":
    main()
