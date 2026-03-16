# -*- coding: utf-8 -*-
"""
V13 전국 전체 예측 HTML 생성
============================
- 11개 지역 × 48개 시간대(30분 간격) = 최대 528개 HTML
- V13 보정 전/후 비교 + 실제 경로 표시
- 수심 격자 토글 기능 포함
"""

import os
import numpy as np
import pandas as pd
import torch
import json
from datetime import timedelta
from tqdm import tqdm

from trajectory_predictor_v13 import (
    TrajectoryPredictorV12, DepthChecker, HistoricalTrackGrid, PathCorrector
)

# ============ 설정 ============
AIS_PATH = r"K:\coding_project\NIA_선박항적예측프로그램\down20260305\ais_20260206.csv"
DISPLAY_LEN = 20  # 20분 표시
OUTPUT_DIR_NAME = "예측결과_전국_20260206"

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

    current_pos = [input_interp['lat'].iloc[-1], input_interp['lon'].iloc[-1]]
    current_cog = input_interp['cog'].iloc[-1]
    current_sog = input_interp['sog'].iloc[-1]

    # 실제 경로 (현재 위치를 시작점으로 포함)
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

    # V13 예측
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

    # 오차 계산
    error_orig = error_corr = 0
    if actual_coords is not None and len(actual_coords) >= DISPLAY_LEN:
        li = DISPLAY_LEN - 1
        dlat = v13_original[li, 0] - actual_coords[li, 0]
        dlon = v13_original[li, 1] - actual_coords[li, 1]
        error_orig = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)
        dlat = v13_corrected[li, 0] - actual_coords[li, 0]
        dlon = v13_corrected[li, 1] - actual_coords[li, 1]
        error_corr = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

    # 보정 거리
    corr_dist = 0
    if len(v13_corrected) >= DISPLAY_LEN and len(v13_original) >= DISPLAY_LEN:
        dlat = v13_corrected[-1, 0] - v13_original[-1, 0]
        dlon = v13_corrected[-1, 1] - v13_original[-1, 1]
        corr_dist = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

    return {
        'mmsi': int(mmsi),
        'current_pos': current_pos,
        'cog': float(current_cog),
        'sog': float(current_sog),
        'input_coords': input_interp[['lat', 'lon']].values,
        'v13_original': v13_original,
        'v13_corrected': v13_corrected,
        'actual_coords': actual_coords,
        'error_orig': error_orig,
        'error_corr': error_corr,
        'corr_dist': corr_dist,
    }


def create_html(ships_data, output_path, title, base_time, region, depth_grids=None):
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
            'pos': [float(ship['current_pos'][0]), float(ship['current_pos'][1])],
            'cog': ship['cog'], 'sog': ship['sog'],
            'input': [[float(c[0]), float(c[1])] for c in ship['input_coords']],
            'v13': [[float(c[0]), float(c[1])] for c in ship['v13_corrected']],
            'v13_orig': [[float(c[0]), float(c[1])] for c in ship['v13_original']],
            'actual': [[float(c[0]), float(c[1])] for c in ship['actual_coords']] if ship['actual_coords'] is not None else [],
            'err_orig': ship['error_orig'],
            'err_corr': ship['error_corr'],
            'corr_dist': ship['corr_dist'],
        })

    ships_str = json.dumps(ships_json)
    depth_str = json.dumps(depth_grids) if depth_grids else "[]"

    errs_orig = [s['error_orig'] for s in ships_data if s['error_orig'] > 0]
    errs_corr = [s['error_corr'] for s in ships_data if s['error_corr'] > 0]
    avg_orig = np.mean(errs_orig) if errs_orig else 0
    avg_corr = np.mean(errs_corr) if errs_corr else 0

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
    padding:12px 20px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
}}
.header h1 {{ font-size:17px; font-weight:500; }}
.header-stats {{ display:flex; gap:12px; font-size:13px; }}
.stat {{ background:rgba(255,255,255,0.1); padding:4px 10px; border-radius:4px; }}
.depth-control {{ position:absolute; top:85px; left:10px; z-index:1000; }}
.depth-btn {{ padding:8px 12px; background:#fff; border:2px solid #666; border-radius:4px; cursor:pointer; font-size:12px; font-family:'Malgun Gothic',sans-serif; }}
.depth-btn.on {{ background:#e74c3c; color:#fff; border-color:#c0392b; }}
#map {{ width:100%; height:calc(100vh - 90px); }}
.info-panel {{
    position:absolute; top:70px; right:10px; background:white;
    padding:15px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.2);
    z-index:1000; min-width:340px; max-height:calc(100vh-100px);
    overflow-y:auto; display:none;
}}
.info-panel.show {{ display:block; }}
.info-row {{ display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #eee; font-size:13px; }}
.info-label {{ color:#666; }}
.info-value {{ font-weight:bold; }}
.close-btn {{ position:absolute; top:8px; right:10px; background:none; border:none; font-size:18px; cursor:pointer; color:#999; }}
.better {{ color:#27ae60; font-weight:bold; }}
.worse {{ color:#e74c3c; font-weight:bold; }}
.legend {{ background:white; padding:10px; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.2); font-size:11px; }}
.legend-title {{ font-weight:bold; margin-bottom:6px; }}
.legend-item {{ display:flex; align-items:center; margin:3px 0; }}
.legend-line {{ width:25px; height:3px; margin-right:8px; border-radius:2px; }}
.bottom-bar {{ background:#2c3e50; color:white; padding:6px 20px; display:flex; justify-content:space-between; font-size:13px; }}
</style>
</head><body>
<div class="header">
    <h1>{title}</h1>
    <div class="header-stats">
        <div class="stat">선박: <b>{len(ships_data)}</b>척</div>
        <div class="stat">보정전: <b>{avg_orig:.2f}km</b></div>
        <div class="stat" style="color:#2ecc71">보정후: <b>{avg_corr:.2f}km</b></div>
        <div class="stat">실적비교: <b>{len(errs_corr)}</b>척</div>
    </div>
</div>
<div id="map"></div>
<div class="info-panel" id="infoPanel">
    <button class="close-btn" onclick="closePanel()">&times;</button>
    <h3 id="panelTitle" style="margin:0 0 10px;border-bottom:2px solid #3498db;padding-bottom:6px;">선박 정보</h3>
    <div class="info-row"><span class="info-label">MMSI</span><span class="info-value" id="infoMmsi">-</span></div>
    <div class="info-row"><span class="info-label">속력</span><span class="info-value" id="infoSog">-</span></div>
    <div class="info-row"><span class="info-label">침로</span><span class="info-value" id="infoCog">-</span></div>
    <div style="margin-top:10px;padding-top:8px;border-top:2px solid #eee;">
        <div style="font-weight:bold;margin-bottom:6px;">{DISPLAY_LEN}분 예측 오차</div>
        <div class="info-row"><span class="info-label">보정 전 (회색)</span><span class="info-value" id="errOrig">-</span></div>
        <div class="info-row"><span class="info-label">보정 후 (빨강)</span><span class="info-value" id="errCorr">-</span></div>
        <div class="info-row"><span class="info-label">보정 거리</span><span class="info-value" id="corrDist">-</span></div>
    </div>
</div>
<div class="bottom-bar">
    <div>{base_time.strftime('%Y-%m-%d %H:%M')} | 초록=과거30분 | 빨강=보정후 | 회색점선=보정전 | 검정점선=실제</div>
    <div>선박 클릭 시 상세</div>
</div>
<script>
var map = L.map('map').setView([{center_lat},{center_lon}], 11);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution:'&copy; OSM &copy; CARTO', subdomains:'abcd'
}}).addTo(map);
var ships = {ships_str};
var depthGrids = {depth_str};
var markers = {{}};
var detailLayers = null;
var selectedMmsi = null;
var depthLayer = null;
var depthVisible = false;

function createDepthLayer() {{
    var layer = L.layerGroup();
    var gs = 0.025;
    depthGrids.forEach(function(g) {{
        var b = [[g.lat-gs/2,g.lon-gs/2],[g.lat+gs/2,g.lon+gs/2]];
        var c = g.type==='land' ? '#8B4513' : '#FFD700';
        var o = g.type==='land' ? 0.7 : 0.5;
        layer.addLayer(L.rectangle(b, {{color:c,weight:1,fillColor:c,fillOpacity:o}})
            .bindPopup('수심: '+g.depth.toFixed(1)+'m<br>유형: '+(g.type==='land'?'육지':'얕은물')));
    }});
    return layer;
}}
depthLayer = createDepthLayer();

var depthControl = L.control({{position:'topleft'}});
depthControl.onAdd = function() {{
    var div = L.DomUtil.create('div','depth-control');
    div.innerHTML = '<button id="depthToggle" class="depth-btn">격자 ON</button>';
    div.onclick = function(e) {{
        e.stopPropagation();
        depthVisible = !depthVisible;
        var btn = document.getElementById('depthToggle');
        if (depthVisible) {{ depthLayer.addTo(map); btn.textContent='격자 OFF'; btn.className='depth-btn on'; }}
        else {{ map.removeLayer(depthLayer); btn.textContent='격자 ON'; btn.className='depth-btn'; }}
    }};
    return div;
}};
depthControl.addTo(map);

function shipIcon(cog, sel) {{
    var sz = sel?30:22, cl = sel?'#e74c3c':'#3498db';
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
    // 과거 (초록)
    var il = L.polyline(s.input, {{color:'#2ecc71',weight:3,opacity:0.8}}).addTo(map); detailLayers.push(il);
    detailLayers.push(L.circleMarker(s.input[0], {{radius:4,color:'#27ae60',fillColor:'#2ecc71',fillOpacity:1,weight:1}}).addTo(map));
    // 보정후 (빨강)
    var pl = L.polyline(s.v13, {{color:'#e74c3c',weight:3,opacity:0.9}}).addTo(map); detailLayers.push(pl);
    if (s.v13.length>0) detailLayers.push(L.circleMarker(s.v13[s.v13.length-1], {{radius:5,color:'#fff',fillColor:'#e74c3c',fillOpacity:1,weight:2}}).bindPopup('+{DISPLAY_LEN}분 보정후').addTo(map));
    // 보정전 (회색점선)
    var ol = L.polyline(s.v13_orig, {{color:'#888',weight:2,opacity:0.5,dashArray:'6,4'}}).addTo(map); detailLayers.push(ol);
    if (s.v13_orig.length>0) detailLayers.push(L.circleMarker(s.v13_orig[s.v13_orig.length-1], {{radius:4,color:'#888',fillColor:'#aaa',fillOpacity:1,weight:1}}).bindPopup('+{DISPLAY_LEN}분 보정전').addTo(map));
    // 실제 (검정점선)
    if (s.actual.length>0) {{
        var al = L.polyline(s.actual, {{color:'#000',weight:2,opacity:0.6,dashArray:'6,4'}}).addTo(map); detailLayers.push(al);
        detailLayers.push(L.circleMarker(s.actual[s.actual.length-1], {{radius:5,color:'#fff',fillColor:'#000',fillOpacity:1,weight:2}}).bindPopup('실제 +{DISPLAY_LEN}분').addTo(map));
    }}
    // 현재 위치
    detailLayers.push(L.circleMarker(s.pos, {{radius:10,color:'#f39c12',fillColor:'#f1c40f',fillOpacity:1,weight:3}}).addTo(map));
    // 패널
    document.getElementById('infoPanel').classList.add('show');
    document.getElementById('panelTitle').textContent = 'MMSI: ' + s.mmsi;
    document.getElementById('infoMmsi').textContent = s.mmsi;
    document.getElementById('infoSog').textContent = s.sog.toFixed(1) + ' kn';
    document.getElementById('infoCog').textContent = s.cog.toFixed(1) + '\\u00B0';
    document.getElementById('errOrig').textContent = s.err_orig > 0 ? s.err_orig.toFixed(2)+' km' : '실적없음';
    document.getElementById('errCorr').textContent = s.err_corr > 0 ? s.err_corr.toFixed(2)+' km' : '실적없음';
    document.getElementById('corrDist').textContent = s.corr_dist.toFixed(3)+' km';
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
        '<div class="legend-item"><div class="legend-line" style="background:#2ecc71"></div>과거 30분</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#e74c3c"></div>V13 보정후</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#888;border-top:2px dashed #888;height:0"></div>V13 보정전</div>' +
        '<div class="legend-item"><div class="legend-line" style="background:#000;border-top:2px dashed #000;height:0"></div>실제 경로</div>' +
        '<div style="margin-top:6px;border-top:1px solid #ddd;padding-top:4px;">' +
        '<div class="legend-item"><div style="width:25px;height:12px;background:#8B4513;opacity:0.7;margin-right:8px;border-radius:2px;"></div>육지</div>' +
        '<div class="legend-item"><div style="width:25px;height:12px;background:#FFD700;opacity:0.5;margin-right:8px;border-radius:2px;"></div>얕은물</div></div>';
    return d;
}};
legend.addTo(map);
</script>
</body></html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return True


def create_region_index(output_dir, results, region):
    html = f'''<!DOCTYPE html>
<html><head><title>V13 {region}</title><meta charset="utf-8">
<style>
body {{ font-family:'Malgun Gothic',sans-serif; margin:0; padding:20px; background:#f5f5f5; }}
h1 {{ color:#2c3e50; border-bottom:3px solid #e74c3c; padding-bottom:10px; }}
.info {{ background:white; padding:15px; border-radius:8px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:15px; }}
.card {{ background:white; padding:15px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1); text-decoration:none; color:#333; transition:transform 0.2s; }}
.card:hover {{ transform:translateY(-3px); }}
.card .time {{ font-size:22px; font-weight:bold; color:#e74c3c; }}
.card .stats {{ margin-top:10px; font-size:13px; color:#666; line-height:1.6; }}
.improved {{ color:#27ae60; font-weight:bold; }}
</style></head><body>
<h1>V13 예측 - {region}</h1>
<div class="info"><p><b>{DISPLAY_LEN}분 예측</b> | <b>HTML:</b> {len(results)}개 | <a href="../index.html">전체 목록</a></p></div>
<div class="grid">'''
    for r in results:
        html += f'''<a class="card" href="{r['filename']}"><div class="time">{r['time']}</div>
<div class="stats">선박: {r['ships']}척<br>보정전: {r['avg_orig']:.2f}km<br><span class="improved">보정후: {r['avg_corr']:.2f}km</span></div></a>'''
    html += '</div></body></html>'
    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)


def create_master_index(root_dir, all_results):
    html = f'''<!DOCTYPE html>
<html><head><title>V13 전국 예측</title><meta charset="utf-8">
<style>
body {{ font-family:'Malgun Gothic',sans-serif; margin:0; padding:20px; background:#f5f5f5; }}
h1 {{ color:#2c3e50; border-bottom:3px solid #e74c3c; padding-bottom:10px; }}
.info {{ background:white; padding:15px; border-radius:8px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:20px; }}
.card {{ background:white; padding:20px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1); text-decoration:none; color:#333; transition:transform 0.2s; }}
.card:hover {{ transform:translateY(-3px); }}
.card .region {{ font-size:28px; font-weight:bold; color:#2c3e50; }}
.card .stats {{ margin-top:12px; font-size:14px; color:#666; line-height:1.8; }}
.improved {{ color:#27ae60; font-weight:bold; }}
</style></head><body>
<h1>V13 전국 예측 결과</h1>
<div class="info"><p><b>전국 11개 지역</b> | <b>{DISPLAY_LEN}분 예측</b> | V13 보정 전/후 비교</p></div>
<div class="grid">'''
    for region, results in all_results.items():
        if not results:
            continue
        total = sum(r['ships'] for r in results)
        avg_orig = np.mean([r['avg_orig'] for r in results if r['avg_orig'] > 0]) if results else 0
        avg_corr = np.mean([r['avg_corr'] for r in results if r['avg_corr'] > 0]) if results else 0
        html += f'''<a class="card" href="{region}/index.html"><div class="region">{region}</div>
<div class="stats">HTML: {len(results)}개 | 총 예측: {total}건<br>보정전: {avg_orig:.2f}km<br><span class="improved">보정후: {avg_corr:.2f}km</span></div></a>'''
    html += '</div></body></html>'
    with open(os.path.join(root_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("V13 전국 예측 HTML 생성 (20분 예측, 보정 전/후 비교)")
    print("=" * 60)
    print(f"Device: {device}")

    # AIS 데이터 로드
    print("AIS 데이터 로드...")
    ais_df = pd.read_csv(AIS_PATH)
    if 'date_time' in ais_df.columns:
        ais_df = ais_df.rename(columns={'date_time': 'datetime'})
    ais_df['datetime'] = pd.to_datetime(ais_df['datetime'])
    ais_df = ais_df[(ais_df['lat'] > -90) & (ais_df['lat'] < 90) &
                    (ais_df['lon'] > -180) & (ais_df['lon'] < 180) &
                    (ais_df['sog'] < 102)].copy()
    print(f"AIS: {len(ais_df):,}행, {ais_df['mmsi'].nunique()}척")

    # 날짜 확인
    date_str = ais_df['datetime'].dt.date.iloc[0].strftime('%Y-%m-%d')
    print(f"날짜: {date_str}")

    # 30분 간격 시간대 (48개)
    base_date = pd.to_datetime(date_str)
    time_slots = [base_date + timedelta(minutes=30 * i) for i in range(48)]

    root_output = os.path.join(base_dir, OUTPUT_DIR_NAME)
    all_region_results = {}

    for region in sorted(ALL_REGION_BOUNDS.keys()):
        model_dir = os.path.join(base_dir, "models", "v12", region)
        if not os.path.exists(os.path.join(model_dir, "model_best.pth")):
            print(f"\n[SKIP] {region} - 모델 없음")
            continue

        output_dir = os.path.join(root_output, region)
        os.makedirs(output_dir, exist_ok=True)

        bounds = ALL_REGION_BOUNDS[region]
        ais_region = ais_df[
            (ais_df['lat'] >= bounds['lat_min']) & (ais_df['lat'] <= bounds['lat_max']) &
            (ais_df['lon'] >= bounds['lon_min']) & (ais_df['lon'] <= bounds['lon_max'])
        ].copy()

        print(f"\n{'='*60}")
        print(f"[{region}] AIS: {len(ais_region):,}행, {ais_region['mmsi'].nunique()}척")

        # 예측기 + 보정기
        predictor = TrajectoryPredictorV12(region=region, model_dir=model_dir, device=device)
        corrector = None
        depth_grids = []
        try:
            depth_checker = DepthChecker(region=region)
            track_grid = HistoricalTrackGrid(grid_resolution=0.0001)
            tg_file = os.path.join(model_dir, 'track_grid.pkl')
            if os.path.exists(tg_file):
                track_grid.load(tg_file)
            corrector = PathCorrector(depth_checker, track_grid)
            grids = depth_checker.get_depth_grid_for_display(region=region)
            depth_grids = grids
            print(f"  수심 격자: {len(grids):,}개")
        except Exception as e:
            print(f"  [경고] 보정기 실패: {e}")

        mmsi_list = ais_region['mmsi'].unique()
        results = []

        for base_time in tqdm(time_slots, desc=f"{region}"):
            ships_data = []
            for mmsi in mmsi_list:
                ship_df = ais_region[ais_region['mmsi'] == mmsi]
                result = process_ship(mmsi, ship_df, base_time, predictor, corrector)
                if result is not None:
                    ships_data.append(result)

            if ships_data:
                time_label = base_time.strftime('%H%M')
                filename = f"prediction_{time_label}.html"
                output_path = os.path.join(output_dir, filename)
                title = f"V13 {region} - {base_time.strftime('%Y-%m-%d %H:%M')}"
                if create_html(ships_data, output_path, title, base_time, region, depth_grids):
                    errs_orig = [s['error_orig'] for s in ships_data if s['error_orig'] > 0]
                    errs_corr = [s['error_corr'] for s in ships_data if s['error_corr'] > 0]
                    results.append({
                        'time': f"{time_label[:2]}:{time_label[2:]}",
                        'filename': filename,
                        'ships': len(ships_data),
                        'avg_orig': np.mean(errs_orig) if errs_orig else 0,
                        'avg_corr': np.mean(errs_corr) if errs_corr else 0,
                    })

        create_region_index(output_dir, results, region)
        all_region_results[region] = results
        print(f"  -> {region}: {len(results)}개 HTML, {sum(r['ships'] for r in results)}건 예측")

    create_master_index(root_output, all_region_results)
    print(f"\n전체 인덱스: {root_output}/index.html")
    print("\n완료!")


if __name__ == "__main__":
    main()
