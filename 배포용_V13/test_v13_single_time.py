# -*- coding: utf-8 -*-
"""
V13 단일 시점 전체 항적 예측 시각화
특정 시점, 특정 지역의 모든 선박에 대해 V13 예측 수행 후 folium 지도로 표시
"""

import os
import numpy as np
import pandas as pd
import torch
import json
from datetime import timedelta

from trajectory_predictor_v13 import (
    TrajectoryPredictorV12, DepthChecker, HistoricalTrackGrid, PathCorrector
)

# ============ 설정 ============
REGION = '부산'
BASE_TIME = pd.Timestamp('2026-03-04 06:00')
DISPLAY_LEN = 20  # 20분 예측
AIS_PATH = r"K:\coding_project\NIA_선박항적예측프로그램\down20260305\ais.csv"

REGION_BOUNDS = {
    '부산': {'lat_min': 34.8, 'lat_max': 35.5, 'lon_min': 128.5, 'lon_max': 129.5},
}

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


def process_ship(mmsi, ship_df, base_time, predictor, corrector):
    """한 선박 처리"""
    input_start = base_time - timedelta(minutes=30)
    input_end = base_time
    actual_end = base_time + timedelta(minutes=DISPLAY_LEN)

    input_df = ship_df[(ship_df['datetime'] >= input_start) & (ship_df['datetime'] <= input_end)]
    input_df = input_df[input_df['sog'] < 30]
    if len(input_df) < 5:
        return None

    input_interp = interpolate_1min(input_df)
    if input_interp is None or len(input_interp) < 30:
        return None
    input_interp = input_interp.tail(30).reset_index(drop=True)

    # 현재 위치 (base_time = 입력 마지막 포인트)
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

    # 현재 위치를 시작점으로 추가 (보정 후, 표시용)
    current_pos_arr = np.array([[current_pos[0], current_pos[1]]])
    v13_original = np.concatenate([current_pos_arr, pred_full[:DISPLAY_LEN-1]], axis=0)
    v13_corrected = np.concatenate([current_pos_arr, corrected_full[:DISPLAY_LEN-1]], axis=0)

    # 오차 계산 (끝점: 둘 다 DISPLAY_LEN개이므로 인덱스 동일)
    error = 0
    if actual_coords is not None and len(actual_coords) >= DISPLAY_LEN:
        li = DISPLAY_LEN - 1
        dlat = v13_corrected[li, 0] - actual_coords[li, 0]
        dlon = v13_corrected[li, 1] - actual_coords[li, 1]
        error = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

    return {
        'mmsi': int(mmsi),
        'current_pos': current_pos,
        'cog': float(current_cog),
        'sog': float(current_sog),
        'input_coords': input_interp[['lat', 'lon']].values,
        'v13_original': v13_original,
        'v13_corrected': v13_corrected,
        'actual_coords': actual_coords,
        'error': error,
    }


def create_html(ships_data, output_path, region, base_time):
    all_lats = [s['current_pos'][0] for s in ships_data]
    all_lons = [s['current_pos'][1] for s in ships_data]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    ships_json = []
    for ship in ships_data:
        ships_json.append({
            'mmsi': ship['mmsi'],
            'pos': [float(ship['current_pos'][0]), float(ship['current_pos'][1])],
            'cog': ship['cog'],
            'sog': ship['sog'],
            'input': [[float(c[0]), float(c[1])] for c in ship['input_coords']],
            'v13': [[float(c[0]), float(c[1])] for c in ship['v13_corrected']],
            'v13_orig': [[float(c[0]), float(c[1])] for c in ship['v13_original']],
            'actual': [[float(c[0]), float(c[1])] for c in ship['actual_coords']] if ship['actual_coords'] is not None else [],
            'error': ship['error'],
        })

    ships_str = json.dumps(ships_json)
    errs = [s['error'] for s in ships_data if s['error'] > 0]
    avg_err = np.mean(errs) if errs else 0
    title = f"V13 전체 항적 - {region} {base_time.strftime('%Y-%m-%d %H:%M')}"

    html = f'''<!DOCTYPE html>
<html><head>
<title>{title}</title>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Malgun Gothic',sans-serif; }}
.header {{
    background:linear-gradient(135deg,#1a1a2e,#16213e); color:white;
    padding:12px 20px; display:flex; justify-content:space-between; align-items:center;
}}
.header h1 {{ font-size:17px; font-weight:500; }}
.header-stats {{ display:flex; gap:12px; font-size:13px; }}
.stat {{ background:rgba(255,255,255,0.1); padding:4px 10px; border-radius:4px; }}
#map {{ width:100%; height:calc(100vh - 90px); }}
.info-panel {{
    position:absolute; top:70px; right:10px; background:white;
    padding:15px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.2);
    z-index:1000; min-width:350px; max-height:calc(100vh-100px);
    overflow-y:auto; display:none;
}}
.info-panel.show {{ display:block; }}
.info-row {{ display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #eee; font-size:13px; }}
.info-label {{ color:#666; }}
.info-value {{ font-weight:bold; }}
.close-btn {{ position:absolute; top:8px; right:10px; background:none; border:none; font-size:18px; cursor:pointer; color:#999; }}
.legend {{
    background:white; padding:10px; border-radius:6px;
    box-shadow:0 2px 6px rgba(0,0,0,0.2); font-size:11px;
}}
.legend-title {{ font-weight:bold; margin-bottom:6px; }}
.legend-item {{ display:flex; align-items:center; margin:3px 0; }}
.legend-line {{ width:25px; height:3px; margin-right:8px; border-radius:2px; }}
.bottom-bar {{
    background:#2c3e50; color:white; padding:6px 20px;
    display:flex; justify-content:space-between; font-size:13px;
}}
</style>
</head><body>
<div class="header">
    <h1>{title}</h1>
    <div class="header-stats">
        <div class="stat">선박: <b>{len(ships_data)}</b>척</div>
        <div class="stat">평균 오차: <b style="color:#e74c3c">{avg_err:.2f} km</b></div>
        <div class="stat">실적 비교: <b>{len(errs)}</b>척</div>
    </div>
</div>
<div id="map"></div>
<div class="info-panel" id="infoPanel">
    <button class="close-btn" onclick="closePanel()">&times;</button>
    <h3 id="panelTitle" style="margin:0 0 10px;border-bottom:2px solid #3498db;padding-bottom:6px;">선박 정보</h3>
    <div class="info-row"><span class="info-label">MMSI</span><span class="info-value" id="infoMmsi">-</span></div>
    <div class="info-row"><span class="info-label">속력</span><span class="info-value" id="infoSog">-</span></div>
    <div class="info-row"><span class="info-label">침로</span><span class="info-value" id="infoCog">-</span></div>
    <div class="info-row"><span class="info-label">{DISPLAY_LEN}분 오차</span><span class="info-value" id="infoError">-</span></div>
</div>
<div class="bottom-bar">
    <div>V13 예측 | 초록=과거30분 | 빨강=예측{DISPLAY_LEN}분 | 검정점선=실제</div>
    <div>선박 클릭 시 상세</div>
</div>
<script>
var map = L.map('map').setView([{center_lat},{center_lon}], 12);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution:'&copy; OSM &copy; CARTO', subdomains:'abcd'
}}).addTo(map);

var ships = {ships_str};
var markers = {{}};
var detailLayers = null;
var selectedMmsi = null;

function shipIcon(cog, sel) {{
    var sz = sel ? 30 : 22, cl = sel ? '#e74c3c' : '#3498db';
    var svg = '<svg width="'+sz+'" height="'+sz+'" viewBox="0 0 24 24"><g transform="rotate('+cog+' 12 12)"><polygon points="12,2 4,22 12,18 20,22" fill="'+cl+'" stroke="#fff" stroke-width="2"/></g></svg>';
    return L.divIcon({{ html:svg, className:'', iconSize:[sz,sz], iconAnchor:[sz/2,sz/2] }});
}}

ships.forEach(function(s) {{
    var mk = L.marker(s.pos, {{icon:shipIcon(s.cog,false)}});
    mk.on('click', function() {{ showShip(s); }});
    mk.addTo(map);
    markers[s.mmsi] = mk;
}});

function showShip(s) {{
    if (detailLayers) detailLayers.forEach(function(l){{ map.removeLayer(l); }});
    if (selectedMmsi && markers[selectedMmsi]) {{
        var prev = ships.find(function(x){{ return x.mmsi===selectedMmsi; }});
        if (prev) markers[selectedMmsi].setIcon(shipIcon(prev.cog, false));
    }}
    selectedMmsi = s.mmsi;
    markers[s.mmsi].setIcon(shipIcon(s.cog, true));
    detailLayers = [];

    // 과거 경로 (초록)
    var il = L.polyline(s.input, {{color:'#2ecc71', weight:3, opacity:0.8}}).addTo(map);
    detailLayers.push(il);
    // 과거 시작점
    var sm = L.circleMarker(s.input[0], {{radius:4, color:'#27ae60', fillColor:'#2ecc71', fillOpacity:1, weight:1}}).addTo(map);
    detailLayers.push(sm);

    // V13 보정 예측 (빨강)
    var pl = L.polyline(s.v13, {{color:'#e74c3c', weight:3, opacity:0.9}}).addTo(map);
    detailLayers.push(pl);
    // 예측 끝점
    if (s.v13.length > 0) {{
        var em = L.circleMarker(s.v13[s.v13.length-1], {{radius:5, color:'#fff', fillColor:'#e74c3c', fillOpacity:1, weight:2}}).bindPopup('+{DISPLAY_LEN}분 예측').addTo(map);
        detailLayers.push(em);
    }}

    // V13 보정 전 (회색 점선)
    var ol = L.polyline(s.v13_orig, {{color:'#888', weight:2, opacity:0.5, dashArray:'6,4'}}).addTo(map);
    detailLayers.push(ol);

    // 실제 경로 (검정 점선)
    if (s.actual.length > 0) {{
        var al = L.polyline(s.actual, {{color:'#000', weight:2, opacity:0.6, dashArray:'6,4'}}).addTo(map);
        detailLayers.push(al);
        var ae = L.circleMarker(s.actual[s.actual.length-1], {{radius:5, color:'#fff', fillColor:'#000', fillOpacity:1, weight:2}}).bindPopup('실제 +{DISPLAY_LEN}분').addTo(map);
        detailLayers.push(ae);
    }}

    // 현재 위치 강조
    var cm = L.circleMarker(s.pos, {{radius:10, color:'#f39c12', fillColor:'#f1c40f', fillOpacity:1, weight:3}}).addTo(map);
    detailLayers.push(cm);

    // 패널
    document.getElementById('infoPanel').classList.add('show');
    document.getElementById('panelTitle').textContent = 'MMSI: ' + s.mmsi;
    document.getElementById('infoMmsi').textContent = s.mmsi;
    document.getElementById('infoSog').textContent = s.sog.toFixed(1) + ' kn';
    document.getElementById('infoCog').textContent = s.cog.toFixed(1) + '\u00B0';
    document.getElementById('infoError').textContent = s.error > 0 ? s.error.toFixed(2) + ' km' : '실적 없음';

    // 화면 맞춤
    var all = s.input.concat(s.v13).concat(s.actual);
    if (all.length > 0) map.fitBounds(L.polyline(all).getBounds().pad(0.3));
}}

function closePanel() {{
    document.getElementById('infoPanel').classList.remove('show');
    if (detailLayers) {{ detailLayers.forEach(function(l){{ map.removeLayer(l); }}); detailLayers = null; }}
    if (selectedMmsi && markers[selectedMmsi]) {{
        var s = ships.find(function(x){{ return x.mmsi===selectedMmsi; }});
        if (s) markers[selectedMmsi].setIcon(shipIcon(s.cog, false));
    }}
    selectedMmsi = null;
}}

var legend = L.control({{position:'bottomright'}});
legend.onAdd = function() {{
    var d = L.DomUtil.create('div','legend');
    d.innerHTML = '<div class="legend-title">범례</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#2ecc71"></div>과거 경로</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#e74c3c"></div>V13 예측</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#888;border-top:2px dashed #888;height:0"></div>V13 보정 전</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#000;border-top:2px dashed #000;height:0"></div>실제 경로</div>';
    return d;
}};
legend.addTo(map);
</script>
</body></html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"저장: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    region = REGION
    base_time = BASE_TIME
    bounds = REGION_BOUNDS[region]

    print(f"{'='*60}")
    print(f"V13 전체 항적 예측 - {region} {base_time}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # AIS 로드
    ais_df = pd.read_csv(AIS_PATH)
    if 'date_time' in ais_df.columns:
        ais_df = ais_df.rename(columns={'date_time': 'datetime'})
    ais_df['datetime'] = pd.to_datetime(ais_df['datetime'])
    ais_df = ais_df[(ais_df['lat'] > -90) & (ais_df['lat'] < 90) &
                    (ais_df['lon'] > -180) & (ais_df['lon'] < 180) &
                    (ais_df['sog'] < 102)].copy()

    # 지역 필터
    ais_region = ais_df[
        (ais_df['lat'] >= bounds['lat_min']) & (ais_df['lat'] <= bounds['lat_max']) &
        (ais_df['lon'] >= bounds['lon_min']) & (ais_df['lon'] <= bounds['lon_max'])
    ].copy()
    print(f"AIS ({region}): {len(ais_region):,}행, {ais_region['mmsi'].nunique()}척")

    # 예측기
    model_dir = os.path.join(base_dir, "models", "v12", region)
    predictor = TrajectoryPredictorV12(region=region, model_dir=model_dir, device=device)

    # 보정기
    corrector = None
    try:
        depth_checker = DepthChecker(region=region)
        track_grid = HistoricalTrackGrid(grid_resolution=0.0001)
        tg_file = os.path.join(model_dir, 'track_grid.pkl')
        if os.path.exists(tg_file):
            track_grid.load(tg_file)
        corrector = PathCorrector(depth_checker, track_grid)
    except Exception as e:
        print(f"[경고] 보정기 실패: {e}")

    # 각 선박 처리
    mmsi_list = ais_region['mmsi'].unique()
    print(f"\n선박 {len(mmsi_list)}척 처리 중...")

    ships_data = []
    for mmsi in mmsi_list:
        ship_df = ais_region[ais_region['mmsi'] == mmsi]
        result = process_ship(mmsi, ship_df, base_time, predictor, corrector)
        if result is not None:
            ships_data.append(result)

    print(f"\n처리 완료: {len(ships_data)}척 (실적 비교: {sum(1 for s in ships_data if s['error']>0)}척)")

    if ships_data:
        output_path = os.path.join(base_dir, f"prediction_v13_{region}_{base_time.strftime('%Y%m%d_%H%M')}.html")
        create_html(ships_data, output_path, region, base_time)

    print("완료!")


if __name__ == "__main__":
    main()
