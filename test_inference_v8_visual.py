# -*- coding: utf-8 -*-
"""
V8 Direct Coordinate Transformer 추론 및 Folium 시각화
======================================================
- 선박별 2일치 항적 로드
- 30분 간격으로 예측 실행
- Folium 지도에 아이콘 클릭 시 예측 항적 표시
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import glob
import warnings
import math
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"
MODEL_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\global_model_v8_direct\울산"
SEQ_LEN = 50
PRED_LEN = 20
PREDICTION_INTERVAL = 30  # 30분 간격으로 예측


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_heads=8,
                 num_layers=4, dropout=0.1, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        encoded = self.transformer(x)
        out = self.output_mlp(encoded[:, -1, :])
        return out.view(B, self.pred_len, 2)


def haversine(lat1, lon1, lat2, lon2):
    """두 점 사이 거리 (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize_input(inp):
    """입력 데이터 정규화 (5차원)"""
    seq_len = inp.shape[0]
    inp_norm = np.zeros((seq_len, 5), dtype=np.float32)

    inp_norm[:, 0] = (inp[:, 0] - inp[0, 0]) * 100
    inp_norm[:, 1] = (inp[:, 1] - inp[0, 1]) * 100
    inp_norm[:, 2] = inp[:, 2] / 30.0

    cog_rad = np.radians(inp[:, 3])
    inp_norm[:, 3] = np.sin(cog_rad)
    inp_norm[:, 4] = np.cos(cog_rad)

    return inp_norm


def load_model(device):
    """모델 로드"""
    print(f"모델 로드 중: {MODEL_DIR}")

    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    model = TrajectoryTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        pred_len=config['pred_len']
    ).to(device)

    best_model_path = os.path.join(MODEL_DIR, 'model_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"  Best 모델 로드: model_best.pth")
    else:
        # 가장 최신 epoch 모델 찾기
        epoch_models = sorted(glob.glob(os.path.join(MODEL_DIR, 'model_epoch*.pth')))
        if epoch_models:
            model.load_state_dict(torch.load(epoch_models[-1], map_location=device))
            print(f"  최신 모델 로드: {os.path.basename(epoch_models[-1])}")
        else:
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model.pth'), map_location=device))
            print(f"  모델 로드: model.pth")

    model.eval()
    return model, config


def load_ship_trajectory(num_chunks=2):
    """
    선박별 연속 항적 로드 (약 2일치)
    Returns: dict[ship_id] = list of (input, target, meta) sequences
    """
    print(f"\n선박별 항적 데이터 로드 중...")

    chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, "sequences_chunk_*.pkl")))

    if not chunk_files:
        print("  [ERROR] 청크 파일 없음")
        return {}

    # 마지막 청크들에서 데이터 로드
    all_sequences = []
    for chunk_file in chunk_files[-num_chunks:]:
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
            all_sequences.extend(chunk)

    print(f"  로드된 시퀀스: {len(all_sequences):,}개")

    # 선박별로 그룹화 (meta[0] = shiptype, 위치 기반으로 그룹화)
    # 실제로는 연속된 시퀀스를 찾아야 하므로, 위치 연속성 기반으로 그룹화
    ship_trajectories = {}

    # 시작 위치 기반으로 클러스터링
    for i, seq in enumerate(all_sequences):
        inp = seq['input']
        start_lat, start_lon = inp[0, 0], inp[0, 1]

        # 간단한 그리드 기반 ship_id 생성 (실제로는 MMSI 사용해야 함)
        grid_lat = int(start_lat * 100) / 100
        grid_lon = int(start_lon * 100) / 100
        ship_id = f"ship_{grid_lat:.2f}_{grid_lon:.2f}"

        if ship_id not in ship_trajectories:
            ship_trajectories[ship_id] = []
        ship_trajectories[ship_id].append(seq)

    # 시퀀스가 많은 상위 선박만 선택
    ship_counts = {k: len(v) for k, v in ship_trajectories.items()}
    top_ships = sorted(ship_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    result = {}
    for ship_id, count in top_ships:
        if count >= 10:  # 최소 10개 시퀀스 이상
            result[ship_id] = ship_trajectories[ship_id]
            print(f"  {ship_id}: {count}개 시퀀스")

    return result


def run_predictions(model, sequences, device, interval=PREDICTION_INTERVAL):
    """
    시퀀스 리스트에서 interval 간격으로 예측 실행
    Returns: list of prediction results
    """
    predictions = []

    # interval 간격으로 샘플링
    step = max(1, interval // 1)  # 1분 간격 데이터 가정
    selected_indices = list(range(0, len(sequences), step))

    for idx in selected_indices:
        if idx >= len(sequences):
            break

        seq = sequences[idx]
        inp = seq['input'].astype(np.float32)
        target = seq['target'].astype(np.float32)

        # 현재 정보
        current_sog = inp[-1, 2]
        current_cog = inp[-1, 3]
        current_lat = inp[-1, 0]
        current_lon = inp[-1, 1]

        # 입력 정규화
        inp_norm = normalize_input(inp)
        inp_tensor = torch.from_numpy(inp_norm).unsqueeze(0).to(device)

        # 예측
        with torch.no_grad():
            pred_delta = model(inp_tensor)

        pred_delta_np = pred_delta.cpu().numpy()[0] / 100
        last_pos = inp[-1, :2]

        # 예측 좌표
        predicted_points = []
        for t in range(PRED_LEN):
            pred_lat = last_pos[0] + pred_delta_np[t, 0]
            pred_lon = last_pos[1] + pred_delta_np[t, 1]
            predicted_points.append((pred_lat, pred_lon))

        # 실제 좌표
        actual_points = [(target[t, 0], target[t, 1]) for t in range(PRED_LEN)]

        # 히스토리 좌표
        history_points = [(inp[t, 0], inp[t, 1]) for t in range(SEQ_LEN)]

        # 오차 계산
        errors = []
        for pred, act in zip(predicted_points, actual_points):
            err = haversine(pred[0], pred[1], act[0], act[1])
            errors.append(err)

        predictions.append({
            'idx': idx,
            'current_lat': current_lat,
            'current_lon': current_lon,
            'current_sog': current_sog,
            'current_cog': current_cog,
            'history': history_points,
            'predicted': predicted_points,
            'actual': actual_points,
            'errors': errors,
            'avg_error': np.mean(errors),
            'error_5min': errors[4] if len(errors) > 4 else None,
            'error_10min': errors[9] if len(errors) > 9 else None,
            'error_20min': errors[19] if len(errors) > 19 else None,
        })

    return predictions


def create_ship_html(ship_id, predictions, output_path):
    """
    개별 선박 HTML 파일 생성
    """
    if not predictions:
        return

    # 중심점 계산
    all_lats = [p['current_lat'] for p in predictions]
    all_lons = [p['current_lon'] for p in predictions]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # 지도 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    # 전체 히스토리 라인 (연한 파란색)
    all_history = []
    for pred in predictions:
        all_history.extend(pred['history'])

    if all_history:
        folium.PolyLine(
            locations=all_history,
            color='blue',
            weight=2,
            opacity=0.4,
            tooltip='전체 항적'
        ).add_to(m)

    # 각 예측 지점
    for i, pred in enumerate(predictions):
        # 예측 경로 (빨간색 점선)
        folium.PolyLine(
            locations=pred['predicted'],
            color='red',
            weight=3,
            opacity=0.8,
            dash_array='8',
            tooltip=f'예측 #{i+1} (오차: {pred["avg_error"]:.2f}km)'
        ).add_to(m)

        # 실제 경로 (녹색 실선)
        folium.PolyLine(
            locations=pred['actual'],
            color='green',
            weight=3,
            opacity=0.8,
            tooltip=f'실제 #{i+1}'
        ).add_to(m)

        # 현재 위치 마커
        popup_html = f"""
        <div style="width:300px; font-family: Arial;">
            <h4 style="margin:5px 0;">예측 #{i+1}</h4>
            <hr style="margin:5px 0;">
            <p><b>속력:</b> {pred['current_sog']:.1f} knots</p>
            <p><b>방향:</b> {pred['current_cog']:.1f}°</p>
            <hr style="margin:5px 0;">
            <p><b>평균 오차:</b> {pred['avg_error']:.2f} km</p>
            <p><b>5분 후:</b> {pred['error_5min']:.2f} km</p>
            <p><b>10분 후:</b> {pred['error_10min']:.2f} km</p>
            <p><b>20분 후:</b> {pred['error_20min']:.2f} km</p>
        </div>
        """
        folium.Marker(
            location=[pred['current_lat'], pred['current_lon']],
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color='blue', icon='ship', prefix='fa'),
            tooltip=f'#{i+1} SOG:{pred["current_sog"]:.1f}kn'
        ).add_to(m)

        # 예측 종료점
        folium.CircleMarker(
            location=pred['predicted'][-1],
            radius=6,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.9,
            tooltip=f'예측 종료 #{i+1}'
        ).add_to(m)

        # 실제 종료점
        folium.CircleMarker(
            location=pred['actual'][-1],
            radius=6,
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.9,
            tooltip=f'실제 종료 #{i+1}'
        ).add_to(m)

    # 평균 오차 계산
    avg_errors = [p['avg_error'] for p in predictions]
    avg_5min = np.mean([p['error_5min'] for p in predictions])
    avg_10min = np.mean([p['error_10min'] for p in predictions])
    avg_20min = np.mean([p['error_20min'] for p in predictions])

    # 범례
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial;">
        <h4 style="margin: 0 0 10px 0;">{ship_id}</h4>
        <p style="margin: 5px 0; font-size: 12px;">예측 수: {len(predictions)}</p>
        <hr style="margin: 10px 0;">
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: blue; opacity: 0.5; margin-right: 10px;"></span>
            히스토리
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: red; margin-right: 10px; border-style: dashed;"></span>
            예측 경로
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: green; margin-right: 10px;"></span>
            실제 경로
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 3px 0; font-size: 11px;"><b>평균 오차:</b> {np.mean(avg_errors):.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>5분 후:</b> {avg_5min:.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>10분 후:</b> {avg_10min:.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>20분 후:</b> {avg_20min:.2f} km</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # 저장
    m.save(output_path)
    print(f"    저장: {output_path}")


def create_folium_map(ship_predictions, output_path):
    """
    Folium 지도 생성
    - 선박별 항적 표시
    - 마커 클릭 시 예측 경로 표시
    """
    print(f"\nFolium 지도 생성 중...")

    # 전체 좌표로 중심점 계산
    all_lats = []
    all_lons = []
    for ship_id, predictions in ship_predictions.items():
        for pred in predictions:
            all_lats.append(pred['current_lat'])
            all_lons.append(pred['current_lon'])

    if not all_lats:
        print("  [ERROR] 데이터 없음")
        return

    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # 지도 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

    # 선박별 색상
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'pink']

    for ship_idx, (ship_id, predictions) in enumerate(ship_predictions.items()):
        color = colors[ship_idx % len(colors)]

        # 선박 그룹 생성
        ship_group = folium.FeatureGroup(name=f"{ship_id} ({len(predictions)}개 예측)")

        # 전체 항적 라인 (연한 색)
        all_history = []
        for pred in predictions:
            all_history.extend(pred['history'])

        if all_history:
            folium.PolyLine(
                locations=all_history,
                color=color,
                weight=2,
                opacity=0.3
            ).add_to(ship_group)

        # 각 예측 지점에 마커 추가
        for i, pred in enumerate(predictions):
            # 팝업 내용 생성
            popup_html = f"""
            <div style="width:350px; font-family: Arial;">
                <h4 style="margin:5px 0;">{ship_id} - 예측 #{i+1}</h4>
                <hr style="margin:5px 0;">
                <p><b>현재 속력:</b> {pred['current_sog']:.1f} knots</p>
                <p><b>현재 방향:</b> {pred['current_cog']:.1f}°</p>
                <p><b>평균 오차:</b> {pred['avg_error']:.2f} km</p>
                <p><b>5분 후 오차:</b> {pred['error_5min']:.2f} km</p>
                <p><b>10분 후 오차:</b> {pred['error_10min']:.2f} km</p>
                <p><b>20분 후 오차:</b> {pred['error_20min']:.2f} km</p>
            </div>
            """

            # 예측/실제 경로를 포함한 JavaScript
            history_coords = str([list(p) for p in pred['history']])
            predicted_coords = str([list(p) for p in pred['predicted']])
            actual_coords = str([list(p) for p in pred['actual']])

            # 마커에 클릭 이벤트 추가
            marker_id = f"marker_{ship_idx}_{i}"

            # 커스텀 아이콘 (방향 표시)
            icon = folium.Icon(
                color=color,
                icon='ship',
                prefix='fa'
            )

            marker = folium.Marker(
                location=[pred['current_lat'], pred['current_lon']],
                popup=folium.Popup(popup_html, max_width=400),
                icon=icon,
                tooltip=f"SOG: {pred['current_sog']:.1f}kn, 오차: {pred['avg_error']:.2f}km"
            )
            marker.add_to(ship_group)

            # 예측 경로 (빨간색 점선)
            folium.PolyLine(
                locations=pred['predicted'],
                color='red',
                weight=3,
                opacity=0.7,
                dash_array='10'
            ).add_to(ship_group)

            # 실제 경로 (녹색 실선)
            folium.PolyLine(
                locations=pred['actual'],
                color='green',
                weight=3,
                opacity=0.7
            ).add_to(ship_group)

            # 예측 종료점 마커
            folium.CircleMarker(
                location=pred['predicted'][-1],
                radius=6,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.8,
                tooltip=f"예측 종료점 (20분 후)"
            ).add_to(ship_group)

            # 실제 종료점 마커
            folium.CircleMarker(
                location=pred['actual'][-1],
                radius=6,
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.8,
                tooltip=f"실제 종료점 (20분 후)"
            ).add_to(ship_group)

        ship_group.add_to(m)

    # 범례 추가
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial;">
        <h4 style="margin: 0 0 10px 0;">V8 Direct Transformer 예측</h4>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: red; margin-right: 10px; border-style: dashed;"></span>
            예측 경로
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: green; margin-right: 10px;"></span>
            실제 경로
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: gray; opacity: 0.5; margin-right: 10px;"></span>
            히스토리
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 12px;">입력: 50분 / 예측: 20분</p>
        <p style="margin: 5px 0; font-size: 12px;">마커 클릭 시 상세 정보</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)

    # 저장
    m.save(output_path)
    print(f"  지도 저장: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V8 Direct Transformer Visual Inference')
    parser.add_argument('--num_chunks', type=int, default=2, help='로드할 청크 수 (약 2일치)')
    parser.add_argument('--interval', type=int, default=30, help='예측 간격 (분)')
    parser.add_argument('--output', type=str, default='prediction_v8_map.html', help='출력 파일명')
    parser.add_argument('--num_ships', type=int, default=5, help='개별 HTML 생성할 선박 수')
    args = parser.parse_args()

    print("=" * 60)
    print("V8 Direct Coordinate Transformer - Folium 시각화")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 모델 로드
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] 모델 디렉토리 없음: {MODEL_DIR}")
        return

    model, config = load_model(device)

    # 선박별 항적 로드
    ship_trajectories = load_ship_trajectory(num_chunks=args.num_chunks)

    if not ship_trajectories:
        print("[ERROR] 선박 데이터 없음")
        return

    # 선박별 예측 실행
    print(f"\n선박별 예측 실행 중 (간격: {args.interval}분)...")
    ship_predictions = {}

    for ship_id, sequences in ship_trajectories.items():
        print(f"\n  {ship_id} 예측 중...")
        predictions = run_predictions(model, sequences, device, interval=args.interval)
        ship_predictions[ship_id] = predictions

        if predictions:
            avg_errors = [p['avg_error'] for p in predictions]
            print(f"    예측 수: {len(predictions)}, 평균 오차: {np.mean(avg_errors):.2f} km")

    # Folium 지도 생성 (전체)
    create_folium_map(ship_predictions, args.output)

    # 선박별 개별 HTML 생성
    print(f"\n선박별 개별 HTML 생성 중 (상위 {args.num_ships}개)...")
    ship_list = list(ship_predictions.items())[:args.num_ships]
    for i, (ship_id, predictions) in enumerate(ship_list):
        if predictions:
            output_file = f"prediction_v8_ship_{i+1}.html"
            create_ship_html(ship_id, predictions, output_file)

    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("전체 결과 요약")
    print("=" * 60)

    all_errors = []
    for ship_id, predictions in ship_predictions.items():
        for pred in predictions:
            all_errors.append(pred['errors'])

    if all_errors:
        all_errors = np.array(all_errors)
        print(f"총 예측 수: {len(all_errors)}")
        print(f"평균 오차: {np.mean(all_errors):.2f} km")
        print(f"5분 후 평균: {np.mean(all_errors[:, 4]):.2f} km")
        print(f"10분 후 평균: {np.mean(all_errors[:, 9]):.2f} km")
        print(f"20분 후 평균: {np.mean(all_errors[:, 19]):.2f} km")


if __name__ == "__main__":
    main()
