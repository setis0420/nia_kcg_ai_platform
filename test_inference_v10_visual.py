# -*- coding: utf-8 -*-
"""
V10 Inertia-Aware Transformer 추론 및 Folium 시각화
====================================================
- 울산 100분 예측 모델 (Inertia Loss 적용)
- 단일 선박 연속 항적 시각화
- 30분 간격으로 예측 실행
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

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산_100min"
MODEL_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\global_model_v10_inertia"
SEQ_LEN = 50
PRED_LEN = 100
PREDICTION_INTERVAL = 30


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
    """V10 모델 - V9와 동일한 구조 (Loss만 다름)"""
    def __init__(self, input_dim=5, hidden_dim=256, num_heads=8,
                 num_layers=6, dropout=0.1, pred_len=PRED_LEN):
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
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
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

    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] 모델 디렉토리 없음: {MODEL_DIR}")
        print("학습을 먼저 실행하세요: python run_train_v10_inertia.py")
        return None, None

    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    print(f"  Config: {config}")

    model = TrajectoryTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        pred_len=config['pred_len']
    ).to(device)

    best_model_path = os.path.join(MODEL_DIR, 'model_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"  Best 모델 로드: model_best.pth")
    else:
        epoch_models = sorted(glob.glob(os.path.join(MODEL_DIR, 'model_epoch*.pth')))
        if epoch_models:
            model.load_state_dict(torch.load(epoch_models[-1], map_location=device, weights_only=True))
            print(f"  최신 모델 로드: {os.path.basename(epoch_models[-1])}")
        else:
            raise FileNotFoundError("모델 파일을 찾을 수 없습니다")

    model.eval()
    return model, config


def find_continuous_trajectory(sequences, min_length=200):
    """연속된 단일 항적 찾기"""
    print(f"\n연속 항적 검색 중...")

    continuous_segments = []
    current_segment = [sequences[0]]

    for i in range(1, len(sequences)):
        prev_seq = sequences[i-1]
        curr_seq = sequences[i]

        prev_end = prev_seq['target'][-1, :2]
        curr_start = curr_seq['input'][0, :2]

        dist = np.sqrt(np.sum((prev_end - curr_start)**2))

        if dist < 0.01:
            current_segment.append(curr_seq)
        else:
            if len(current_segment) >= min_length:
                continuous_segments.append(current_segment)
            current_segment = [curr_seq]

    if len(current_segment) >= min_length:
        continuous_segments.append(current_segment)

    print(f"  발견된 연속 항적: {len(continuous_segments)}개")
    for i, seg in enumerate(continuous_segments[:5]):
        start_pos = seg[0]['input'][0, :2]
        end_pos = seg[-1]['target'][-1, :2]
        print(f"    [{i}] 길이: {len(seg)}, 시작: ({start_pos[0]:.4f}, {start_pos[1]:.4f}), 끝: ({end_pos[0]:.4f}, {end_pos[1]:.4f})")

    return continuous_segments


def load_single_ship_trajectory(num_chunks=2, ship_idx=0):
    """단일 선박의 연속 항적 로드"""
    print(f"\n단일 선박 항적 데이터 로드 중... (from {DATA_DIR})")

    chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, "sequences_chunk_*.pkl")))

    if not chunk_files:
        print("  [ERROR] 청크 파일 없음")
        return []

    all_sequences = []
    for chunk_file in chunk_files[-num_chunks:]:
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
            all_sequences.extend(chunk)

    print(f"  로드된 시퀀스: {len(all_sequences):,}개")
    print(f"  입력 shape: {all_sequences[0]['input'].shape}")
    print(f"  타겟 shape: {all_sequences[0]['target'].shape}")

    continuous_segments = find_continuous_trajectory(all_sequences, min_length=100)

    if not continuous_segments:
        print("  [ERROR] 연속 항적 없음")
        return []

    selected_idx = min(ship_idx, len(continuous_segments) - 1)
    selected_segment = continuous_segments[selected_idx]

    print(f"\n  선택된 항적 [{selected_idx}]: {len(selected_segment)}개 시퀀스")

    return selected_segment


def run_predictions(model, sequences, device, interval=PREDICTION_INTERVAL):
    """시퀀스에서 interval 간격으로 예측 실행"""
    predictions = []
    step = max(1, interval)
    selected_indices = list(range(0, len(sequences), step))

    for idx in selected_indices:
        if idx >= len(sequences):
            break

        seq = sequences[idx]
        inp = seq['input'].astype(np.float32)
        target = seq['target'].astype(np.float32)

        current_sog = inp[-1, 2]
        current_cog = inp[-1, 3]
        current_lat = inp[-1, 0]
        current_lon = inp[-1, 1]

        inp_norm = normalize_input(inp)
        inp_tensor = torch.from_numpy(inp_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_delta = model(inp_tensor)

        pred_delta_np = pred_delta.cpu().numpy()[0] / 100
        last_pos = inp[-1, :2]

        predicted_points = []
        for t in range(PRED_LEN):
            pred_lat = last_pos[0] + pred_delta_np[t, 0]
            pred_lon = last_pos[1] + pred_delta_np[t, 1]
            predicted_points.append((pred_lat, pred_lon))

        actual_points = [(target[t, 0], target[t, 1]) for t in range(min(PRED_LEN, len(target)))]
        history_points = [(inp[t, 0], inp[t, 1]) for t in range(SEQ_LEN)]

        errors = []
        for i, (pred, act) in enumerate(zip(predicted_points, actual_points)):
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
            'error_10min': errors[9] if len(errors) > 9 else None,
            'error_30min': errors[29] if len(errors) > 29 else None,
            'error_60min': errors[59] if len(errors) > 59 else None,
            'error_100min': errors[99] if len(errors) > 99 else None,
        })

    return predictions


def create_ship_html(ship_id, predictions, output_path):
    """개별 선박 HTML 파일 생성"""
    if not predictions:
        return

    all_lats = [p['current_lat'] for p in predictions]
    all_lons = [p['current_lon'] for p in predictions]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

    # 전체 히스토리
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

    # 각 예측
    for i, pred in enumerate(predictions):
        # 예측 경로 (빨간색)
        folium.PolyLine(
            locations=pred['predicted'],
            color='red',
            weight=3,
            opacity=0.8,
            dash_array='8',
            tooltip=f'예측 #{i+1} (평균 오차: {pred["avg_error"]:.2f}km)'
        ).add_to(m)

        # 실제 경로 (녹색)
        folium.PolyLine(
            locations=pred['actual'],
            color='green',
            weight=3,
            opacity=0.8,
            tooltip=f'실제 #{i+1}'
        ).add_to(m)

        # 현재 위치 마커
        error_10 = pred['error_10min'] if pred['error_10min'] else 0
        error_30 = pred['error_30min'] if pred['error_30min'] else 0
        error_60 = pred['error_60min'] if pred['error_60min'] else 0
        error_100 = pred['error_100min'] if pred['error_100min'] else 0

        popup_html = f"""
        <div style="width:300px; font-family: Arial;">
            <h4 style="margin:5px 0;">V10 예측 #{i+1} (100분)</h4>
            <p style="color: #666; margin: 3px 0; font-size: 11px;">Inertia-Aware Model</p>
            <hr style="margin:5px 0;">
            <p><b>속력:</b> {pred['current_sog']:.1f} knots</p>
            <p><b>방향:</b> {pred['current_cog']:.1f}</p>
            <hr style="margin:5px 0;">
            <p><b>평균 오차:</b> {pred['avg_error']:.2f} km</p>
            <p><b>10분 후:</b> {error_10:.2f} km</p>
            <p><b>30분 후:</b> {error_30:.2f} km</p>
            <p><b>60분 후:</b> {error_60:.2f} km</p>
            <p><b>100분 후:</b> {error_100:.2f} km</p>
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
            radius=8,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.9,
            tooltip=f'예측 종료 #{i+1} (100분)'
        ).add_to(m)

        # 실제 종료점
        folium.CircleMarker(
            location=pred['actual'][-1],
            radius=8,
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.9,
            tooltip=f'실제 종료 #{i+1}'
        ).add_to(m)

    # 평균 오차 계산
    avg_errors = [p['avg_error'] for p in predictions]
    avg_10min = np.mean([p['error_10min'] for p in predictions if p['error_10min']])
    avg_30min = np.mean([p['error_30min'] for p in predictions if p['error_30min']])
    avg_60min = np.mean([p['error_60min'] for p in predictions if p['error_60min']])
    avg_100min = np.mean([p['error_100min'] for p in predictions if p['error_100min']])

    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial;">
        <h4 style="margin: 0 0 5px 0;">V10 Inertia-Aware 100분 예측</h4>
        <p style="margin: 5px 0; font-size: 12px; color: #666;">{ship_id}</p>
        <p style="margin: 5px 0; font-size: 12px;">예측 수: {len(predictions)}</p>
        <hr style="margin: 10px 0;">
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: blue; opacity: 0.5; margin-right: 10px;"></span>
            히스토리 (50분)
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: red; margin-right: 10px;"></span>
            예측 경로 (100분)
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 30px; height: 3px; background: green; margin-right: 10px;"></span>
            실제 경로
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 3px 0; font-size: 11px;"><b>평균 오차:</b> {np.mean(avg_errors):.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>10분 후:</b> {avg_10min:.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>30분 후:</b> {avg_30min:.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>60분 후:</b> {avg_60min:.2f} km</p>
        <p style="margin: 3px 0; font-size: 11px;"><b>100분 후:</b> {avg_100min:.2f} km</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"    저장: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V10 Inertia-Aware 100분 예측 시각화 - 단일 선박')
    parser.add_argument('--num_chunks', type=int, default=2, help='로드할 청크 수')
    parser.add_argument('--interval', type=int, default=30, help='예측 간격 (분)')
    parser.add_argument('--ship_idx', type=int, default=0, help='선박 인덱스 (0부터)')
    args = parser.parse_args()

    print("=" * 60)
    print("V10 Inertia-Aware Transformer - 단일 선박 100분 예측")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"데이터: {DATA_DIR}")
    print(f"모델: {MODEL_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] 데이터 디렉토리 없음: {DATA_DIR}")
        return

    model, config = load_model(device)
    if model is None:
        return

    print(f"\n{'='*40}")
    print(f"선박 #{args.ship_idx + 1} 처리 중...")
    print(f"{'='*40}")

    # 단일 선박의 연속 항적 로드
    sequences = load_single_ship_trajectory(num_chunks=args.num_chunks, ship_idx=args.ship_idx)

    if not sequences:
        print(f"  [ERROR] 연속 항적 없음")
        return

    # 예측 실행
    print(f"\n예측 실행 중 (간격: {args.interval}분)...")
    predictions = run_predictions(model, sequences, device, interval=args.interval)

    if not predictions:
        print(f"  [ERROR] 예측 없음")
        return

    # 결과 출력
    avg_errors = [p['avg_error'] for p in predictions]
    print(f"\n  예측 수: {len(predictions)}")
    print(f"  평균 오차: {np.mean(avg_errors):.2f} km")

    # HTML 생성
    ship_id = f"ship_{args.ship_idx + 1}"
    output_file = f"prediction_v10_single_{args.ship_idx + 1}.html"
    create_ship_html(ship_id, predictions, output_file)

    # 오차 통계
    all_errors = np.array([p['errors'] for p in predictions])
    print(f"\n  === V10 오차 통계 ===")
    print(f"  10분 후: {np.mean(all_errors[:, 9]):.2f} km")
    print(f"  30분 후: {np.mean(all_errors[:, 29]):.2f} km")
    print(f"  60분 후: {np.mean(all_errors[:, 59]):.2f} km")
    print(f"  100분 후: {np.mean(all_errors[:, 99]):.2f} km")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
