# -*- coding: utf-8 -*-
"""
V13 이상선박 테스트 - ais.csv 이상선박만 필터링하여 20분 예측
==============================================================
- down20260305/ais.csv 에서 이상선박 MMSI 추출
- ais_20260206.csv 전국 데이터에서 해당 MMSI만 필터
- 11개 지역 모델로 예측 및 HTML 시각화
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import json
from tqdm import tqdm

from trajectory_predictor_v13 import (
    TrajectoryPredictorV12, DepthChecker, HistoricalTrackGrid, PathCorrector,
    REGION_BOUNDS, get_shiptype_category, get_length_category,
    SHIPTYPE_NAMES, LENGTH_NAMES
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


def process_time_slot(df, predictor, corrector, base_time, min_sog=2.0):
    input_start = base_time - timedelta(minutes=30)
    input_end = base_time
    actual_end = base_time + timedelta(minutes=DISPLAY_LEN)

    mask = (df['datetime'] >= input_start) & (df['datetime'] <= input_end) & (df['sog'] >= min_sog)
    ships_with_data = df[mask].groupby('mmsi').size()
    ships_with_data = ships_with_data[ships_with_data >= 5]

    ships_data = []
    for mmsi in ships_with_data.index:
        ship_df = df[df['mmsi'] == mmsi].copy()
        input_df = ship_df[(ship_df['datetime'] >= input_start) & (ship_df['datetime'] <= input_end)]
        input_df = input_df[input_df['sog'] >= min_sog]
        if len(input_df) < 5:
            continue

        input_interp = interpolate_1min(input_df)
        if input_interp is None or len(input_interp) < 30:
            continue
        input_interp = input_interp.tail(30).reset_index(drop=True)

        current_pos = [input_interp['lat'].iloc[-1], input_interp['lon'].iloc[-1]]
        current_cog = input_interp['cog'].iloc[-1]
        current_sog = input_interp['sog'].iloc[-1]

        shiptype_cat = 4
        length_cat = 0

        # 실제 데이터
        actual_df = ship_df[(ship_df['datetime'] > input_end) & (ship_df['datetime'] <= actual_end)]
        actual_coords = None
        if len(actual_df) >= 5:
            actual_interp = interpolate_1min(actual_df)
            if actual_interp is not None and len(actual_interp) >= DISPLAY_LEN:
                actual_interp = actual_interp.head(DISPLAY_LEN).reset_index(drop=True)
                actual_coords = actual_interp[['lat', 'lon']].values

        # 예측
        input_data = input_interp[['lat', 'lon', 'sog', 'cog']].values
        result = predictor.predict(input_data)
        pred_full = result['predicted_coords']
        original_coords = pred_full[:DISPLAY_LEN]

        # 보정
        if corrector is not None:
            last_position = result['last_position']
            last_cog = input_data[-1, 3]
            corrected_coords = corrector.correct_path(
                pred_full.copy(), last_position, last_cog=last_cog,
                shiptype_cat=shiptype_cat, length_cat=length_cat
            )[:DISPLAY_LEN]
        else:
            corrected_coords = original_coords.copy()

        # 오차 계산 (끝점)
        li = DISPLAY_LEN - 1
        error_original = error_corrected = 0
        if actual_coords is not None and len(actual_coords) >= DISPLAY_LEN:
            dlat = original_coords[li, 0] - actual_coords[li, 0]
            dlon = original_coords[li, 1] - actual_coords[li, 1]
            error_original = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)
            dlat = corrected_coords[li, 0] - actual_coords[li, 0]
            dlon = corrected_coords[li, 1] - actual_coords[li, 1]
            error_corrected = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

        correction_dist = np.sqrt(
            ((corrected_coords[li, 0] - original_coords[li, 0]) * 111)**2 +
            ((corrected_coords[li, 1] - original_coords[li, 1]) * 111 * np.cos(np.radians(35)))**2
        )

        ships_data.append({
            'mmsi': int(mmsi),
            'current_pos': current_pos,
            'cog': current_cog,
            'sog': current_sog,
            'shiptype_name': SHIPTYPE_NAMES[shiptype_cat],
            'length_name': LENGTH_NAMES[length_cat],
            'input_coords': input_interp[['lat', 'lon']].values,
            'original_coords': original_coords,
            'corrected_coords': corrected_coords,
            'actual_coords': actual_coords,
            'error_original': error_original,
            'error_corrected': error_corrected,
            'correction_dist': correction_dist,
        })

    return ships_data


def create_interactive_html(ships_data, output_path, title, base_time):
    if not ships_data:
        return False

    all_lats = [s['current_pos'][0] for s in ships_data]
    all_lons = [s['current_pos'][1] for s in ships_data]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    ships_json = []
    for ship in ships_data:
        ships_json.append({
            'mmsi': int(ship['mmsi']),
            'current_pos': [float(ship['current_pos'][0]), float(ship['current_pos'][1])],
            'cog': float(ship['cog']),
            'sog': float(ship['sog']),
            'shiptype': ship.get('shiptype_name', '기타선'),
            'length': ship.get('length_name', '0-40m'),
            'input': [[float(c[0]), float(c[1])] for c in ship['input_coords']],
            'original': [[float(c[0]), float(c[1])] for c in ship['original_coords']],
            'corrected': [[float(c[0]), float(c[1])] for c in ship['corrected_coords']],
            'actual': [[float(c[0]), float(c[1])] for c in ship['actual_coords']] if ship['actual_coords'] is not None else [],
            'error_original': float(ship.get('error_original', 0)),
            'error_corrected': float(ship.get('error_corrected', 0)),
            'correction_dist': float(ship.get('correction_dist', 0)),
        })

    ships_json_str = json.dumps(ships_json)
    input_start = (base_time - timedelta(minutes=30)).strftime('%H:%M')
    input_end = base_time.strftime('%H:%M')
    pred_end = (base_time + timedelta(minutes=DISPLAY_LEN)).strftime('%H:%M')

    ships_with_errors = [s for s in ships_data if s.get('error_corrected', 0) > 0]
    if ships_with_errors:
        avg_original = np.mean([s['error_original'] for s in ships_with_errors])
        avg_corrected = np.mean([s['error_corrected'] for s in ships_with_errors])
        avg_correction = np.mean([s['correction_dist'] for s in ships_with_errors])
    else:
        avg_original = avg_corrected = avg_correction = 0

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
            display: flex; justify-content: space-between; align-items: center;
        }}
        .header h1 {{ font-size: 20px; font-weight: 500; }}
        .header-stats {{ display: flex; gap: 20px; font-size: 13px; }}
        .header-stats .stat {{ background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 4px; }}
        .stat-improved {{ color: #2ecc71; }}
        #map {{ width: 100%; height: calc(100vh - 120px); }}
        .info-panel {{
            position: absolute; top: 80px; right: 10px; background: white;
            padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000; min-width: 300px; max-height: calc(100vh - 200px);
            overflow-y: auto; display: none;
        }}
        .info-panel.show {{ display: block; }}
        .info-panel h3 {{ margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #3498db; }}
        .info-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; font-size: 13px; }}
        .info-label {{ color: #666; }}
        .info-value {{ font-weight: bold; }}
        .close-btn {{ position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 20px; cursor: pointer; color: #999; }}
        .error-improved {{ color: #27ae60; font-weight: bold; }}
        .error-worse {{ color: #e74c3c; font-weight: bold; }}
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
            <div class="stat">보정 전: <b>{avg_original:.2f}</b>km</div>
            <div class="stat stat-improved">보정 후: <b>{avg_corrected:.2f}</b>km</div>
        </div>
    </div>
    <div id="map"></div>
    <div class="info-panel" id="infoPanel">
        <button class="close-btn" onclick="closePanel()">&times;</button>
        <h3 id="panelTitle">선박 정보</h3>
        <div class="info-row"><span class="info-label">MMSI</span><span class="info-value" id="infoMmsi">-</span></div>
        <div class="info-row"><span class="info-label">선종</span><span class="info-value" id="infoShiptype">-</span></div>
        <div class="info-row"><span class="info-label">속력</span><span class="info-value" id="infoSog">-</span></div>
        <div class="info-row"><span class="info-label">보정 전</span><span class="info-value" id="errorOriginal">-</span></div>
        <div class="info-row"><span class="info-label">보정 후</span><span class="info-value" id="errorCorrected">-</span></div>
        <div class="info-row"><span class="info-label">보정 거리</span><span class="info-value" id="correctionDist">-</span></div>
    </div>
    <div class="bottom-bar">
        <div>기준: {base_time.strftime('%Y-%m-%d %H:%M')} | 과거: {input_start}~{input_end} | 예측: {input_end}~{pred_end}</div>
        <div>이상선박만 필터링 | 선박 클릭 시 상세 정보</div>
    </div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);
        var shipsData = {ships_json_str};
        var shipMarkers = {{}};
        var currentPathLayers = null;
        var selectedMmsi = null;

        function createShipIcon(cog, sel) {{
            var sz = sel ? 32 : 24, cl = sel ? '#e74c3c' : '#3498db';
            var svg = '<svg width="'+sz+'" height="'+sz+'" viewBox="0 0 24 24"><g transform="rotate('+cog+' 12 12)"><polygon points="12,2 4,22 12,18 20,22" fill="'+cl+'" stroke="#fff" stroke-width="2"/></g></svg>';
            return L.divIcon({{ html: svg, className: 'ship-marker', iconSize: [sz,sz], iconAnchor: [sz/2,sz/2] }});
        }}

        shipsData.forEach(function(ship) {{
            // 예측선 (빨강) - 항상 표시
            if (ship.corrected.length > 0) {{
                L.polyline(ship.corrected, {{ color: '#e74c3c', weight: 3, opacity: 0.7 }})
                    .bindPopup('MMSI: '+ship.mmsi+'<br>SOG: '+ship.sog.toFixed(1)+' kn<br>{DISPLAY_LEN}분 오차: '+ship.error_corrected.toFixed(2)+' km').addTo(map);
                var li = ship.corrected.length - 1;
                L.circleMarker(ship.corrected[li], {{ radius: 4, color: '#fff', fillColor: '#e74c3c', fillOpacity: 1, weight: 1 }}).addTo(map);
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
            if (ship.input.length > 0) {{
                var il = L.polyline(ship.input, {{ color: '#2ecc71', weight: 5, opacity: 0.9 }}).addTo(map);
                currentPathLayers.push(il);
                var sm = L.circleMarker(ship.input[0], {{ radius: 6, color: '#27ae60', fillColor: '#2ecc71', fillOpacity: 1 }}).bindPopup('<b>30분 전</b>').addTo(map);
                currentPathLayers.push(sm);
            }}
            if (ship.original.length > 0) {{
                var ol = L.polyline(ship.original, {{ color: '#888', weight: 4, opacity: 0.8, dashArray: '8,6' }}).bindPopup('보정 전 오차: '+ship.error_original.toFixed(2)+' km').addTo(map);
                currentPathLayers.push(ol);
            }}
            var cm = L.circleMarker(ship.current_pos, {{ radius: 10, color: '#f39c12', fillColor: '#f1c40f', fillOpacity: 1, weight: 3 }}).bindPopup('MMSI: '+ship.mmsi).addTo(map);
            currentPathLayers.push(cm);
            document.getElementById('infoPanel').classList.add('show');
            document.getElementById('panelTitle').textContent = 'MMSI: ' + ship.mmsi;
            document.getElementById('infoMmsi').textContent = ship.mmsi;
            document.getElementById('infoShiptype').textContent = ship.shiptype;
            document.getElementById('infoSog').textContent = ship.sog.toFixed(1) + ' kn';
            document.getElementById('errorOriginal').textContent = ship.error_original.toFixed(2) + ' km';
            var ce = document.getElementById('errorCorrected');
            ce.textContent = ship.error_corrected.toFixed(2) + ' km';
            ce.className = 'info-value ' + (ship.error_corrected < ship.error_original ? 'error-improved' : 'error-worse');
            document.getElementById('correctionDist').textContent = ship.correction_dist.toFixed(2) + ' km';
            var all = ship.input.concat(ship.corrected).concat(ship.actual);
            if (all.length > 0) map.fitBounds(L.polyline(all).getBounds().pad(0.2));
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

        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function() {{
            var d = L.DomUtil.create('div', 'legend');
            d.innerHTML = '<div class="legend-title">범례</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#2ecc71;"></div>과거 30분</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#888;border-top:3px dashed #888;height:0;"></div>보정 전</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#e74c3c;"></div>보정 후 예측</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#000;border-top:3px dashed #000;height:0;"></div>실제 경로</div>';
            return d;
        }};
        legend.addTo(map);
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return True


def create_index_html(output_dir, results, base_date, region):
    html = f'''<!DOCTYPE html>
<html><head><title>V13 이상선박 {region} - {base_date.strftime('%Y-%m-%d')}</title><meta charset="utf-8">
<style>
body {{ font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
.info {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
.card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-decoration: none; color: #333; transition: transform 0.2s; }}
.card:hover {{ transform: translateY(-3px); }}
.card .time {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
.card .stats {{ margin-top: 10px; font-size: 13px; color: #666; }}
.improved {{ color: #27ae60; font-weight: bold; }}
</style></head><body>
<h1>V13 이상선박 {region} - {base_date.strftime('%Y-%m-%d')}</h1>
<div class="info"><p><b>이상선박만 필터링</b> | <b>예측:</b> {DISPLAY_LEN}분 | <b>파일:</b> {len(results)}개 | <a href="../index.html">전체 목록</a></p></div>
<div class="grid">'''
    for r in results:
        ts = r['time'].replace(':', '')
        imp = r['avg_original'] - r['avg_corrected']
        html += f'''<a class="card" href="prediction_{ts}.html"><div class="time">{r['time']}</div>
<div class="stats">이상선박: {r['ships']}척<br>오차: <span class="improved">{r['avg_corrected']:.2f} km</span> ({imp:+.2f})</div></a>'''
    html += '</div></body></html>'
    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)


def create_master_index(root_dir, all_results, base_date):
    html = f'''<!DOCTYPE html>
<html><head><title>V13 이상선박 전국 - {base_date.strftime('%Y-%m-%d')}</title><meta charset="utf-8">
<style>
body {{ font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
.info {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
.card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-decoration: none; color: #333; transition: transform 0.2s; }}
.card:hover {{ transform: translateY(-3px); }}
.card .region {{ font-size: 28px; font-weight: bold; color: #e74c3c; }}
.card .stats {{ margin-top: 12px; font-size: 14px; color: #666; line-height: 1.8; }}
</style></head><body>
<h1>V13 이상선박 전국 항적 예측 - {base_date.strftime('%Y-%m-%d')}</h1>
<div class="info"><p><b>이상선박(ais.csv)만 필터링</b> | <b>예측:</b> {DISPLAY_LEN}분 | <b>지역:</b> {len(all_results)}개</p></div>
<div class="grid">'''
    for region, results in all_results.items():
        if not results:
            continue
        total = sum(r['ships'] for r in results)
        avg = np.mean([r['avg_corrected'] for r in results if r['avg_corrected'] > 0]) if results else 0
        html += f'''<a class="card" href="{region}/index.html"><div class="region">{region}</div>
<div class="stats">HTML: {len(results)}개<br>총 예측: {total}건<br>평균 오차: {avg:.2f} km</div></a>'''
    html += '</div></body></html>'
    with open(os.path.join(root_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = r"K:\coding_project\NIA_선박항적예측프로그램\down20260305\ais_20260206.csv"
    outlier_path = r"K:\coding_project\NIA_선박항적예측프로그램\down20260305\ais.csv"
    min_sog = 2.0
    TIME_INTERVAL_MIN = 30

    print("=" * 60)
    print("V13 이상선박 테스트 - 오차 큰 선박만 필터링 (20분 예측)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 이상선박 MMSI 목록 로드
    print("이상선박 목록 로드 중...")
    outlier_df = pd.read_csv(outlier_path)
    outlier_mmsis = set(outlier_df['mmsi'].unique())
    print(f"이상선박 MMSI: {len(outlier_mmsis)}개")

    # 전국 데이터 로드
    print("전국 데이터 로드 중...")
    df_all = pd.read_csv(data_path)
    if 'date_time' in df_all.columns and 'datetime' not in df_all.columns:
        df_all = df_all.rename(columns={'date_time': 'datetime'})
    df_all['datetime'] = pd.to_datetime(df_all['datetime'])
    print(f"전체: {len(df_all):,}개, 선박: {df_all['mmsi'].nunique()}척")

    # 이상선박만 필터링
    df_all = df_all[df_all['mmsi'].isin(outlier_mmsis)].copy()
    print(f"이상선박 필터 후: {len(df_all):,}개, 선박: {df_all['mmsi'].nunique()}척")

    # 시간대
    base_date_val = df_all['datetime'].dt.date.min()
    base_date = datetime(base_date_val.year, base_date_val.month, base_date_val.day)
    time_slots = []
    for hour in range(24):
        for minute in range(0, 60, TIME_INTERVAL_MIN):
            t = base_date + timedelta(hours=hour, minutes=minute)
            if t >= base_date + timedelta(minutes=30):
                time_slots.append(t)

    all_region_results = {}

    for region, bounds in ALL_REGION_BOUNDS.items():
        model_dir = os.path.join(base_dir, "models", "v12", region)
        if not os.path.exists(os.path.join(model_dir, "model_best.pth")):
            print(f"\n[SKIP] {region} - 모델 없음")
            continue

        output_dir = os.path.join(base_dir, "예측결과_이상선박_20260206", region)
        os.makedirs(output_dir, exist_ok=True)

        df = df_all[
            (df_all['lat'] >= bounds['lat_min']) & (df_all['lat'] <= bounds['lat_max']) &
            (df_all['lon'] >= bounds['lon_min']) & (df_all['lon'] <= bounds['lon_max'])
        ].copy()

        if len(df) < 100:
            print(f"\n[SKIP] {region} - 데이터 부족 ({len(df)}건)")
            continue

        print(f"\n{'='*60}")
        print(f"[{region}] 이상선박 데이터: {len(df):,}개, 선박: {df['mmsi'].nunique()}척")

        # 예측기
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
            print(f"  [경고] 보정기 초기화 실패: {e}")

        results = []
        for base_time in tqdm(time_slots, desc=f"{region} HTML"):
            ships_data = process_time_slot(df, predictor, corrector, base_time, min_sog)
            if ships_data:
                time_str = base_time.strftime('%H%M')
                output_path = os.path.join(output_dir, f"prediction_{time_str}.html")
                title = f"V13 이상선박 {region} - {base_time.strftime('%Y-%m-%d %H:%M')}"
                if create_interactive_html(ships_data, output_path, title, base_time):
                    errs = [s for s in ships_data if s['error_corrected'] > 0]
                    avg_o = np.mean([s['error_original'] for s in errs]) if errs else 0
                    avg_c = np.mean([s['error_corrected'] for s in errs]) if errs else 0
                    results.append({'time': base_time.strftime('%H:%M'), 'ships': len(ships_data),
                                    'avg_original': avg_o, 'avg_corrected': avg_c})

        create_index_html(output_dir, results, base_date, region)
        all_region_results[region] = results
        print(f"  -> {region}: {len(results)}개 HTML")

    root_output = os.path.join(base_dir, "예측결과_이상선박_20260206")
    create_master_index(root_output, all_region_results, base_date)
    print(f"\n전체 인덱스: {root_output}/index.html")
    print("\n완료!")


if __name__ == "__main__":
    main()
