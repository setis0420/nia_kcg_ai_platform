# -*- coding: utf-8 -*-
"""
V13 목포 모델 - 30분 단위 전체 예측 HTML 생성
=============================================
- V12 기반 + 수심 제약 + 과거 항적 방향 보정
- 00:00 ~ 23:30 까지 30분 단위로 48개 HTML 생성
- 보정 전/후 경로 비교 표시
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from datetime import datetime, timedelta
import json
from tqdm import tqdm

# V13 보정기 import
from trajectory_predictor_v13 import (
    DepthChecker, HistoricalTrackGrid, PathCorrector, REGION_BOUNDS
)
from trajectory_predictor_v12 import (
    TrajectoryTransformerV12, get_shiptype_category, get_length_category,
    SHIPTYPE_NAMES, LENGTH_NAMES
)


# ============================================
# 수심 격자 로드 (표시용) - 450m 고해상도 NC 파일 사용
# ============================================

def load_depth_grids_for_display(region='목포'):
    """
    수심 격자 데이터 로드 (육지 + 얕은 물만)
    - 450m 고해상도 NC 파일 사용
    - DepthChecker의 get_depth_grid_for_display() 활용
    """
    # DepthChecker 초기화 (자동으로 NC 파일 탐색)
    depth_checker = DepthChecker(region=region)

    if not depth_checker.loaded:
        print(f"[경고] 수심 데이터 로드 실패")
        return []

    # 시각화용 그리드 데이터 반환
    grids = depth_checker.get_depth_grid_for_display(region=region)
    print(f"[V13] 수심 격자 (표시용): {len(grids):,}개 (육지+얕은물)")

    return grids


# ============================================
# 보간 함수
# ============================================

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


# ============================================
# V13 예측기 클래스 (수심/항적 보정 포함)
# ============================================

class PredictorV13WithCorrection:
    """V12 모델 + V13 보정 (수심 제약 + 과거 항적 방향)"""
    SEQ_LEN = 30
    PRED_LEN = 60

    def __init__(self, model_dir, region='목포', device='cuda'):
        self.device = device
        self.region = region
        self.model = None
        self.path_corrector = None

        self._load_model(model_dir)
        self._init_corrector()

    def _load_model(self, model_dir):
        """best 모델 로드"""
        model_path = os.path.join(model_dir, "model_best.pth")

        self.model = TrajectoryTransformerV12(
            input_dim=5,
            hidden_dim=192,
            num_heads=6,
            num_layers=4,
            pred_len=self.PRED_LEN,
            num_shiptype=5,
            num_length=9,
            meta_embed_dim=16
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[V13] 모델 로드 완료: {self.region} (device: {self.device})")

    def _init_corrector(self):
        """V13 보정기 초기화"""
        # 수심 체커 (NC 파일 자동 탐색)
        self.depth_checker = DepthChecker(region=self.region)

        # 과거 항적 그리드
        self.track_grid = HistoricalTrackGrid(grid_resolution=0.0001)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        track_grid_file = os.path.join(script_dir, 'models', self.region, 'track_grid.pkl')

        if os.path.exists(track_grid_file):
            self.track_grid.load(track_grid_file)
        else:
            print(f"[V13] 과거 항적 그리드 없음: {track_grid_file}")

        # 경로 보정기
        self.path_corrector = PathCorrector(self.depth_checker, self.track_grid)

    def _normalize_input(self, input_data):
        seq_len = input_data.shape[0]
        normalized = np.zeros((seq_len, 5), dtype=np.float32)

        normalized[:, 0] = (input_data[:, 0] - input_data[0, 0]) * 100
        normalized[:, 1] = (input_data[:, 1] - input_data[0, 1]) * 100
        normalized[:, 2] = input_data[:, 2] / 30.0

        cog_rad = np.radians(input_data[:, 3])
        normalized[:, 3] = np.sin(cog_rad)
        normalized[:, 4] = np.cos(cog_rad)

        return normalized

    def predict(self, input_data, shiptype_cat=4, length_cat=0, apply_correction=True):
        """
        예측 수행 (보정 포함)

        Returns:
            dict: {
                'original': 보정 전 예측 좌표,
                'corrected': 보정 후 예측 좌표,
                'correction_applied': bool
            }
        """
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)

        last_position = input_data[-1, :2].copy()
        last_cog = input_data[-1, 3]

        normalized = self._normalize_input(input_data)
        input_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

        shiptype_tensor = torch.tensor([shiptype_cat], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor([length_cat], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_delta = self.model(input_tensor, shiptype_tensor, length_tensor)

        pred_delta_np = pred_delta.cpu().numpy()[0]
        pred_delta_real = pred_delta_np / 100.0

        original_coords = np.zeros((self.PRED_LEN, 2), dtype=np.float32)
        original_coords[:, 0] = last_position[0] + pred_delta_real[:, 0]
        original_coords[:, 1] = last_position[1] + pred_delta_real[:, 1]

        # 보정 적용 (선종/길이별 밀도 그리드 참조)
        if apply_correction and self.path_corrector is not None:
            corrected_coords = self.path_corrector.correct_path(
                original_coords,
                last_position,
                last_cog=last_cog,
                shiptype_cat=shiptype_cat,
                length_cat=length_cat
            )
        else:
            corrected_coords = original_coords.copy()

        return {
            'original': original_coords,
            'corrected': corrected_coords,
            'correction_applied': apply_correction,
            'last_position': last_position
        }


# ============================================
# HTML 생성
# ============================================

def create_interactive_html(ships_data, output_path, title, base_time, depth_grids=None):
    """V13 보정 전/후 비교 HTML 생성 (수심 격자 포함)"""

    if not ships_data:
        return False

    # 중심점 계산
    all_lats = [s['current_pos'][0] for s in ships_data]
    all_lons = [s['current_pos'][1] for s in ships_data]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # 수심 격자 JSON (육지와 얕은 곳만)
    depth_json_str = "[]"
    if depth_grids is not None:
        depth_json_str = json.dumps(depth_grids)

    # 선박 데이터를 JSON으로 변환
    ships_json = []
    for ship in ships_data:
        ship_obj = {
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
        }
        ships_json.append(ship_obj)

    ships_json_str = json.dumps(ships_json)

    # 시간 정보
    input_start = (base_time - timedelta(minutes=30)).strftime('%H:%M')
    input_end = base_time.strftime('%H:%M')
    pred_end = (base_time + timedelta(minutes=60)).strftime('%H:%M')

    # 통계
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
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ font-size: 20px; font-weight: 500; }}
        .header-stats {{
            display: flex;
            gap: 20px;
            font-size: 13px;
        }}
        .header-stats .stat {{
            background: rgba(255,255,255,0.1);
            padding: 5px 12px;
            border-radius: 4px;
        }}
        .stat-improved {{ color: #2ecc71; }}

        #map {{ width: 100%; height: calc(100vh - 120px); }}

        .info-panel {{
            position: absolute;
            top: 80px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            min-width: 320px;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
            display: none;
        }}
        .info-panel.show {{ display: block; }}
        .info-panel h3 {{
            margin: 0 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
            font-size: 16px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #eee;
        }}
        .info-label {{ color: #666; font-size: 13px; }}
        .info-value {{ font-weight: bold; font-size: 13px; }}
        .close-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #999;
        }}

        .comparison-section {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 2px solid #eee;
        }}
        .comparison-section h4 {{
            font-size: 14px;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .error-improved {{ color: #27ae60; font-weight: bold; }}
        .error-worse {{ color: #e74c3c; font-weight: bold; }}

        .legend {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            font-size: 11px;
        }}
        .legend-title {{ font-weight: bold; margin-bottom: 8px; font-size: 13px; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-line {{ width: 25px; height: 3px; margin-right: 8px; border-radius: 2px; }}

        .bottom-bar {{
            background: #2c3e50;
            color: white;
            padding: 8px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>V13 선박 항적 예측 - 목포 (수심/항적 보정)</h1>
        <div class="header-stats">
            <div class="stat">선박: <b>{len(ships_data)}</b>척</div>
            <div class="stat">보정 전: <b>{avg_original:.2f}</b>km</div>
            <div class="stat stat-improved">보정 후: <b>{avg_corrected:.2f}</b>km</div>
            <div class="stat">평균 보정: <b>{avg_correction:.2f}</b>km</div>
        </div>
    </div>

    <div id="map"></div>

    <div class="info-panel" id="infoPanel">
        <button class="close-btn" onclick="closePanel()">&times;</button>
        <h3 id="panelTitle">선박 정보</h3>
        <div class="info-row">
            <span class="info-label">MMSI</span>
            <span class="info-value" id="infoMmsi">-</span>
        </div>
        <div class="info-row">
            <span class="info-label">선종</span>
            <span class="info-value" id="infoShiptype">-</span>
        </div>
        <div class="info-row">
            <span class="info-label">길이</span>
            <span class="info-value" id="infoLength">-</span>
        </div>
        <div class="info-row">
            <span class="info-label">속력 (SOG)</span>
            <span class="info-value" id="infoSog">-</span>
        </div>
        <div class="comparison-section">
            <h4>60분 예측 오차 비교</h4>
            <div class="info-row">
                <span class="info-label">보정 전 (회색)</span>
                <span class="info-value" id="errorOriginal">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">보정 후 (빨강)</span>
                <span class="info-value" id="errorCorrected">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">보정 거리</span>
                <span class="info-value" id="correctionDist">-</span>
            </div>
        </div>
    </div>

    <div class="bottom-bar">
        <div class="time-info">
            <span><b>기준 시각:</b> {base_time.strftime('%Y-%m-%d %H:%M')}</span>
            <span><b>과거:</b> {input_start}~{input_end}</span>
            <span><b>예측:</b> {input_end}~{pred_end}</span>
        </div>
        <div>선박 클릭 시 보정 전/후 경로 비교</div>
    </div>

    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
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
            var gridSize = 0.025;  // 격자 크기 (도)

            depthGrids.forEach(function(g) {{
                var bounds = [
                    [g.lat - gridSize/2, g.lon - gridSize/2],
                    [g.lat + gridSize/2, g.lon + gridSize/2]
                ];

                var color, fillOpacity;
                if (g.type === 'land') {{
                    color = '#8B4513';  // 갈색 (육지)
                    fillOpacity = 0.7;
                }} else {{
                    color = '#FFD700';  // 노랑 (얕은 물)
                    fillOpacity = 0.5;
                }}

                var rect = L.rectangle(bounds, {{
                    color: color,
                    weight: 1,
                    fillColor: color,
                    fillOpacity: fillOpacity
                }}).bindPopup('수심: ' + g.depth.toFixed(1) + 'm<br>(' + g.lat.toFixed(3) + ', ' + g.lon.toFixed(3) + ')');

                layer.addLayer(rect);
            }});

            return layer;
        }}

        depthLayer = createDepthLayer();

        // 수심 격자 토글 버튼
        var depthControl = L.control({{position: 'topright'}});
        depthControl.onAdd = function() {{
            var div = L.DomUtil.create('div', 'depth-toggle');
            div.innerHTML = '<button id="depthToggle" style="padding:8px 12px;background:#fff;border:2px solid #666;border-radius:4px;cursor:pointer;font-size:12px;">수심격자 표시</button>';
            div.onclick = function(e) {{
                e.stopPropagation();
                depthVisible = !depthVisible;
                if (depthVisible) {{
                    depthLayer.addTo(map);
                    document.getElementById('depthToggle').textContent = '수심격자 숨기기';
                    document.getElementById('depthToggle').style.background = '#e74c3c';
                    document.getElementById('depthToggle').style.color = '#fff';
                }} else {{
                    map.removeLayer(depthLayer);
                    document.getElementById('depthToggle').textContent = '수심격자 표시';
                    document.getElementById('depthToggle').style.background = '#fff';
                    document.getElementById('depthToggle').style.color = '#000';
                }}
            }};
            return div;
        }};
        depthControl.addTo(map);

        function createShipIcon(cog, isSelected) {{
            var size = isSelected ? 32 : 24;
            var color = isSelected ? '#e74c3c' : '#3498db';
            var svg = '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24">' +
                '<g transform="rotate(' + cog + ' 12 12)">' +
                '<polygon points="12,2 4,22 12,18 20,22" fill="' + color + '" stroke="#fff" stroke-width="2"/>' +
                '</g></svg>';
            return L.divIcon({{
                html: svg,
                className: 'ship-marker',
                iconSize: [size, size],
                iconAnchor: [size/2, size/2]
            }});
        }}

        shipsData.forEach(function(ship) {{
            var marker = L.marker(ship.current_pos, {{
                icon: createShipIcon(ship.cog, false)
            }});
            marker.on('click', function() {{ showShipPath(ship); }});
            marker.addTo(map);
            shipMarkers[ship.mmsi] = marker;
        }});

        function showShipPath(ship) {{
            if (currentPathLayers) {{
                currentPathLayers.forEach(l => map.removeLayer(l));
            }}
            if (selectedMmsi && shipMarkers[selectedMmsi]) {{
                var prev = shipsData.find(s => s.mmsi === selectedMmsi);
                if (prev) shipMarkers[selectedMmsi].setIcon(createShipIcon(prev.cog, false));
            }}

            selectedMmsi = ship.mmsi;
            shipMarkers[ship.mmsi].setIcon(createShipIcon(ship.cog, true));
            currentPathLayers = [];

            // 과거 경로 (초록)
            if (ship.input.length > 0) {{
                var inputLine = L.polyline(ship.input, {{ color: '#2ecc71', weight: 4, opacity: 0.9 }}).addTo(map);
                currentPathLayers.push(inputLine);
                var startMarker = L.circleMarker(ship.input[0], {{
                    radius: 6, color: '#27ae60', fillColor: '#2ecc71', fillOpacity: 1
                }}).bindPopup('<b>30분 전</b>').addTo(map);
                currentPathLayers.push(startMarker);
            }}

            // 보정 전 예측 (회색 점선)
            if (ship.original.length > 0) {{
                var origLine = L.polyline(ship.original, {{
                    color: '#888888', weight: 3, opacity: 0.6, dashArray: '8,6'
                }}).bindPopup('<b>보정 전</b><br>60분 오차: ' + ship.error_original.toFixed(2) + ' km').addTo(map);
                currentPathLayers.push(origLine);
            }}

            // 보정 후 예측 (빨강)
            if (ship.corrected.length > 0) {{
                var corrLine = L.polyline(ship.corrected, {{
                    color: '#e74c3c', weight: 4, opacity: 0.9
                }}).bindPopup('<b>보정 후</b><br>60분 오차: ' + ship.error_corrected.toFixed(2) + ' km').addTo(map);
                currentPathLayers.push(corrLine);

                // 10분 단위 마커
                [9, 19, 29, 39, 49, 59].forEach(function(idx) {{
                    if (ship.corrected.length > idx) {{
                        var m = L.circleMarker(ship.corrected[idx], {{
                            radius: 5, color: '#fff', fillColor: '#e74c3c', fillOpacity: 1, weight: 2
                        }}).bindPopup((idx+1) + '분 예측').addTo(map);
                        currentPathLayers.push(m);
                    }}
                }});
            }}

            // 실제 경로 (검정)
            if (ship.actual.length > 0) {{
                var actualLine = L.polyline(ship.actual, {{
                    color: '#000000', weight: 4, opacity: 0.9, dashArray: '10,6'
                }}).bindPopup('<b>실제 경로</b>').addTo(map);
                currentPathLayers.push(actualLine);

                if (ship.actual.length >= 60) {{
                    var actualEnd = L.circleMarker(ship.actual[59], {{
                        radius: 7, color: '#fff', fillColor: '#000', fillOpacity: 1, weight: 2
                    }}).bindPopup('<b>실제 +60분</b>').addTo(map);
                    currentPathLayers.push(actualEnd);
                }}
            }}

            // 현재 위치
            var currMarker = L.circleMarker(ship.current_pos, {{
                radius: 10, color: '#f39c12', fillColor: '#f1c40f', fillOpacity: 1, weight: 3
            }}).bindPopup('<b>현재 위치</b>').addTo(map);
            currentPathLayers.push(currMarker);

            updateInfoPanel(ship);

            var allPoints = ship.input.concat(ship.corrected).concat(ship.actual);
            if (allPoints.length > 0) map.fitBounds(L.polyline(allPoints).getBounds().pad(0.2));
        }}

        function updateInfoPanel(ship) {{
            document.getElementById('infoPanel').classList.add('show');
            document.getElementById('panelTitle').textContent = 'MMSI: ' + ship.mmsi;
            document.getElementById('infoMmsi').textContent = ship.mmsi;
            document.getElementById('infoShiptype').textContent = ship.shiptype;
            document.getElementById('infoLength').textContent = ship.length;
            document.getElementById('infoSog').textContent = ship.sog.toFixed(1) + ' knots';

            var origEl = document.getElementById('errorOriginal');
            origEl.textContent = ship.error_original.toFixed(2) + ' km';

            var corrEl = document.getElementById('errorCorrected');
            corrEl.textContent = ship.error_corrected.toFixed(2) + ' km';
            corrEl.className = 'info-value ' + (ship.error_corrected < ship.error_original ? 'error-improved' : 'error-worse');

            document.getElementById('correctionDist').textContent = ship.correction_dist.toFixed(2) + ' km';
        }}

        function closePanel() {{
            document.getElementById('infoPanel').classList.remove('show');
            if (currentPathLayers) {{
                currentPathLayers.forEach(l => map.removeLayer(l));
                currentPathLayers = null;
            }}
            if (selectedMmsi && shipMarkers[selectedMmsi]) {{
                var ship = shipsData.find(s => s.mmsi === selectedMmsi);
                if (ship) shipMarkers[selectedMmsi].setIcon(createShipIcon(ship.cog, false));
            }}
            selectedMmsi = null;
        }}

        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function() {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<div class="legend-title">경로 범례</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#2ecc71;"></div>과거 30분</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#888;border-top:3px dashed #888;height:0;"></div>보정 전 예측</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#e74c3c;"></div>보정 후 예측</div>' +
                '<div class="legend-item"><div class="legend-line" style="background:#000;border-top:3px dashed #000;height:0;"></div>실제 경로</div>';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return True


# ============================================
# 특정 시간대 선박 처리
# ============================================

def process_time_slot(df, predictor, base_time, min_sog=5.0):
    """특정 시간대의 선박 데이터 처리 (V13 보정 적용)"""

    input_start = base_time - timedelta(minutes=30)
    input_end = base_time
    actual_end = base_time + timedelta(minutes=60)

    mask_input = (df['datetime'] >= input_start) & (df['datetime'] <= input_end) & (df['sog'] >= min_sog)
    ships_with_data = df[mask_input].groupby('mmsi').size()
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

        # 선종/길이
        shiptype_cat = 4
        length_cat = 0

        if 'shiptype' in ship_df.columns:
            val = ship_df['shiptype'].iloc[0]
            if pd.notna(val):
                shiptype_cat = get_shiptype_category(val)

        if 'length' in ship_df.columns:
            val = ship_df['length'].iloc[0]
            if pd.notna(val):
                length_cat = get_length_category(val)

        shiptype_name = SHIPTYPE_NAMES[shiptype_cat]
        length_name = LENGTH_NAMES[length_cat]

        # 실제 데이터
        actual_df = ship_df[(ship_df['datetime'] > input_end) & (ship_df['datetime'] <= actual_end)]
        actual_coords = None
        if len(actual_df) >= 5:
            actual_interp = interpolate_1min(actual_df)
            if actual_interp is not None and len(actual_interp) >= 60:
                actual_interp = actual_interp.head(60).reset_index(drop=True)
                actual_coords = actual_interp[['lat', 'lon']].values

        # V13 예측 (보정 포함)
        input_data = input_interp[['lat', 'lon', 'sog', 'cog']].values
        result = predictor.predict(input_data, shiptype_cat=shiptype_cat, length_cat=length_cat)

        original_coords = result['original']
        corrected_coords = result['corrected']

        # 오차 계산
        error_original = error_corrected = 0
        if actual_coords is not None and len(actual_coords) >= 60:
            # 보정 전 오차
            dlat = original_coords[59, 0] - actual_coords[59, 0]
            dlon = original_coords[59, 1] - actual_coords[59, 1]
            error_original = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

            # 보정 후 오차
            dlat = corrected_coords[59, 0] - actual_coords[59, 0]
            dlon = corrected_coords[59, 1] - actual_coords[59, 1]
            error_corrected = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)

        # 보정 거리
        correction_dist = np.sqrt(
            ((corrected_coords[59, 0] - original_coords[59, 0]) * 111)**2 +
            ((corrected_coords[59, 1] - original_coords[59, 1]) * 111 * np.cos(np.radians(35)))**2
        )

        ships_data.append({
            'mmsi': int(mmsi),
            'current_pos': current_pos,
            'cog': current_cog,
            'sog': current_sog,
            'shiptype_name': shiptype_name,
            'length_name': length_name,
            'input_coords': input_interp[['lat', 'lon']].values,
            'original_coords': original_coords,
            'corrected_coords': corrected_coords,
            'actual_coords': actual_coords,
            'error_original': error_original,
            'error_corrected': error_corrected,
            'correction_dist': correction_dist,
        })

    return ships_data


# ============================================
# 메인 실행
# ============================================

def main():
    base_dir = r"K:\coding_project\NIA_선박항적예측프로그램\배포용_V12"
    model_dir = os.path.join(base_dir, "models", "목포")
    data_path = r"K:\coding_project\NIA_선박항적예측프로그램\aisSample.parquet"

    output_dir = os.path.join(base_dir, "예측결과_v13_목포")
    os.makedirs(output_dir, exist_ok=True)

    min_sog = 2.0
    TIME_INTERVAL_MIN = 10  # 10분 간격

    # 목포 지역 범위
    MOKPO_BOUNDS = {'lat_min': 33.5, 'lat_max': 35.7, 'lon_min': 125.2, 'lon_max': 126.8}

    print("=" * 60)
    print("V13 목포 - 수심/항적 보정 적용 예측")
    print("=" * 60)
    print(f"모델 폴더: {model_dir}")
    print(f"출력 폴더: {output_dir}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # V13 예측기 초기화
    predictor = PredictorV13WithCorrection(model_dir, region='목포', device=device)

    # 수심 격자 데이터 로드 (육지 + 얕은 물만)
    depth_grids = load_depth_grids_for_display()
    print(f"수심 격자 (표시용): {len(depth_grids)}개")

    # 데이터 로드 (parquet)
    print("데이터 로드 중...")
    df_all = pd.read_parquet(data_path)
    df_all['datetime'] = pd.to_datetime(df_all['datetime'])

    # 목포 지역 필터링
    df_all = df_all[(df_all['lat'] >= MOKPO_BOUNDS['lat_min']) & (df_all['lat'] <= MOKPO_BOUNDS['lat_max']) &
                    (df_all['lon'] >= MOKPO_BOUNDS['lon_min']) & (df_all['lon'] <= MOKPO_BOUNDS['lon_max'])]

    # 날짜별 처리
    df_all['date'] = df_all['datetime'].dt.date
    unique_dates = sorted(df_all['date'].unique())
    print(f"목포 지역 전체 데이터: {len(df_all):,}개")
    print(f"처리할 날짜: {len(unique_dates)}개 - {unique_dates}")

    # 모든 날짜에 대해 10분 간격 시간대 생성
    all_time_slots = []
    for target_date in unique_dates:
        base_date = datetime(target_date.year, target_date.month, target_date.day)
        for hour in range(24):
            for minute in range(0, 60, TIME_INTERVAL_MIN):
                if hour == 0 and minute == 0:
                    continue
                all_time_slots.append(base_date + timedelta(hours=hour, minutes=minute))

    # 현재 날짜 데이터로 df 설정 (첫 번째 날짜)
    df = df_all[df_all['date'] == unique_dates[-1]]  # 가장 많은 데이터가 있는 마지막 날짜
    time_slots = [t for t in all_time_slots if t.date() == unique_dates[-1]]

    print(f"처리 대상: {len(df):,}개, 선박 수: {df['mmsi'].nunique()}")

    print(f"생성할 시간대: {len(time_slots)}개\n")

    results = []
    for base_time in tqdm(time_slots, desc="HTML 생성"):
        ships_data = process_time_slot(df, predictor, base_time, min_sog)

        if ships_data:
            time_str = base_time.strftime('%H%M')
            output_path = os.path.join(output_dir, f"prediction_{time_str}.html")
            title = f"V13 목포 예측 - {base_time.strftime('%H:%M')}"

            success = create_interactive_html(ships_data, output_path, title, base_time, depth_grids)

            if success:
                ships_with_errors = [s for s in ships_data if s['error_corrected'] > 0]
                if ships_with_errors:
                    avg_orig = np.mean([s['error_original'] for s in ships_with_errors])
                    avg_corr = np.mean([s['error_corrected'] for s in ships_with_errors])
                else:
                    avg_orig = avg_corr = 0

                results.append({
                    'time': base_time.strftime('%H:%M'),
                    'ships': len(ships_data),
                    'avg_original': avg_orig,
                    'avg_corrected': avg_corr
                })

    # 결과 요약
    print("\n" + "=" * 60)
    print("생성 완료")
    print("=" * 60)
    print(f"생성된 파일: {len(results)}개")

    if results:
        print(f"\n{'시간':<10} {'선박':>6} {'보정전':>10} {'보정후':>10} {'개선':>8}")
        print("-" * 50)
        for r in results[:10]:
            improve = r['avg_original'] - r['avg_corrected']
            print(f"{r['time']:<10} {r['ships']:>6} {r['avg_original']:>10.2f} {r['avg_corrected']:>10.2f} {improve:>+8.2f}")

    # 인덱스 생성
    create_index_html(output_dir, results, base_date)
    print("\n완료!")


def create_index_html(output_dir, results, base_date):
    """인덱스 HTML 생성"""
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>V13 목포 예측 - 시간별 목록</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        .info {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                 text-decoration: none; color: #333; transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 15px rgba(0,0,0,0.15); }}
        .card .time {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
        .card .stats {{ margin-top: 10px; font-size: 13px; color: #666; }}
        .improved {{ color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>V13 목포 항적 예측 (수심/항적 보정)</h1>
    <div class="info">
        <p><b>날짜:</b> {base_date.strftime('%Y-%m-%d')}</p>
        <p><b>모델:</b> V13 (V12 + 수심 제약 + 과거 항적 방향 보정)</p>
        <p><b>총 파일:</b> {len(results)}개</p>
    </div>
    <div class="grid">'''

    for r in results:
        time_str = r['time'].replace(':', '')
        improve = r['avg_original'] - r['avg_corrected']
        html += f'''
        <a class="card" href="prediction_{time_str}.html">
            <div class="time">{r['time']}</div>
            <div class="stats">
                선박: {r['ships']}척<br>
                보정 전: {r['avg_original']:.2f} km<br>
                보정 후: <span class="improved">{r['avg_corrected']:.2f} km</span><br>
                개선: <span class="improved">{improve:+.2f} km</span>
            </div>
        </a>'''

    html += '''
    </div>
</body>
</html>'''

    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"인덱스 생성: {output_dir}/index.html")


if __name__ == "__main__":
    main()
