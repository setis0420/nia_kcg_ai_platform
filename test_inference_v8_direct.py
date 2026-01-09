# -*- coding: utf-8 -*-
"""
V8 Direct Coordinate Transformer 추론 및 시각화
================================================
- 좌표 직접 예측 (Regression)
- 격자 없이 정밀한 좌표 예측
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import glob
import warnings
import math
from scipy.interpolate import splprep, splev

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"
MODEL_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\global_model_v8_direct\울산"
SEQ_LEN = 50
PRED_LEN = 20


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
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
    """V8 Direct Coordinate Transformer"""
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


def smooth_path(points, num_smooth=50):
    """경로를 스플라인으로 스무딩"""
    if len(points) < 4:
        return points

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    try:
        tck, u = splprep([lats, lons], s=0.0001, k=min(3, len(points)-1))
        u_new = np.linspace(0, 1, num_smooth)
        smooth_lats, smooth_lons = splev(u_new, tck)
        return list(zip(smooth_lats, smooth_lons))
    except:
        return points


def normalize_input(inp):
    """입력 데이터 정규화 (5차원)"""
    seq_len = inp.shape[0]
    inp_norm = np.zeros((seq_len, 5), dtype=np.float32)

    # lat, lon: 상대 좌표 (첫 번째 위치 기준)
    inp_norm[:, 0] = (inp[:, 0] - inp[0, 0]) * 100
    inp_norm[:, 1] = (inp[:, 1] - inp[0, 1]) * 100

    # sog: 정규화 (0-30 knots)
    inp_norm[:, 2] = inp[:, 2] / 30.0

    # cog: sin/cos 변환
    cog_rad = np.radians(inp[:, 3])
    inp_norm[:, 3] = np.sin(cog_rad)
    inp_norm[:, 4] = np.cos(cog_rad)

    return inp_norm


def generate_html(sample_idx, history_points, actual_points, predicted_points,
                  predicted_smooth, errors, output_path,
                  current_sog=0, current_cog=0, avg_sog=0):
    """HTML 시각화 생성"""

    all_lats = [p[0] for p in history_points + actual_points + predicted_points]
    all_lons = [p[1] for p in history_points + actual_points + predicted_points]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # 오차 텍스트
    error_text = f"평균: {np.mean(errors):.2f}km"
    if len(errors) >= 5:
        error_text += f", 5분: {errors[4]:.2f}km"
    if len(errors) >= 10:
        error_text += f", 10분: {errors[9]:.2f}km"
    if len(errors) >= 20:
        error_text += f", 20분: {errors[19]:.2f}km"

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>V8 Direct Transformer 예측 - Sample {sample_idx}</title>
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
            max-width: 400px;
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
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3 style="margin-top:0;">V8 Direct Transformer 예측</h3>
        <p><strong>Sample:</strong> {sample_idx}</p>
        <p><strong>모델:</strong> Direct Coordinate Regression</p>
        <p><strong>입력:</strong> {SEQ_LEN}분 / <strong>예측:</strong> {PRED_LEN}분</p>
        <hr>
        <p><strong>현재 속력:</strong> {current_sog:.1f} knots (평균: {avg_sog:.1f})</p>
        <p><strong>현재 방향:</strong> {current_cog:.1f}°</p>
        <hr>
        <p><strong>예측 오차:</strong> {error_text}</p>
    </div>
    <div class="legend">
        <h4 style="margin-top:0;">범례</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: blue;"></div>
            <span>히스토리 (입력 {SEQ_LEN}분)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: green;"></div>
            <span>실제 경로 ({PRED_LEN}분)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: red;"></div>
            <span>예측 경로</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: orange; border-style: dashed;"></div>
            <span>예측 (스무딩)</span>
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

        // 예측 경로 (빨간색)
        var predictedPoints = {[list(p) for p in predicted_points]};
        L.polyline(predictedPoints, {{
            color: 'red',
            weight: 3,
            opacity: 0.9
        }}).addTo(map);

        // 예측 스무딩 경로 (주황색 점선)
        var predictedSmooth = {[list(p) for p in predicted_smooth]};
        L.polyline(predictedSmooth, {{
            color: 'orange',
            weight: 2,
            opacity: 0.6,
            dashArray: '5, 5'
        }}).addTo(map);

        // 마커들
        if (historyPoints.length > 0) {{
            L.circleMarker(historyPoints[0], {{
                radius: 8, color: 'blue', fillColor: 'white', fillOpacity: 1
            }}).addTo(map).bindPopup('히스토리 시작');

            L.circleMarker(historyPoints[historyPoints.length - 1], {{
                radius: 8, color: 'blue', fillColor: 'blue', fillOpacity: 1
            }}).addTo(map).bindPopup('예측 시작점');
        }}

        if (predictedPoints.length > 0) {{
            L.circleMarker(predictedPoints[predictedPoints.length - 1], {{
                radius: 8, color: 'red', fillColor: 'red', fillOpacity: 1
            }}).addTo(map).bindPopup('예측 종료 ({PRED_LEN}분)');
        }}

        if (actualPoints.length > 0) {{
            L.circleMarker(actualPoints[actualPoints.length - 1], {{
                radius: 8, color: 'green', fillColor: 'green', fillOpacity: 1
            }}).addTo(map).bindPopup('실제 종료');
        }}

        // 지도 범위 맞추기
        var allPoints = historyPoints.concat(actualPoints).concat(predictedPoints);
        if (allPoints.length > 0) {{
            map.fitBounds(allPoints);
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  HTML 저장: {output_path}")


def load_model(device):
    """모델 로드"""
    print(f"모델 로드 중: {MODEL_DIR}")

    # config 로드
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Best Val Loss: {config.get('best_val_loss', 'N/A'):.6f}")

    # 모델 로드
    model = TrajectoryTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        pred_len=config['pred_len']
    ).to(device)

    # Best 모델 우선, 없으면 일반 모델
    best_model_path = os.path.join(MODEL_DIR, 'model_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"  Best 모델 로드: model_best.pth")
    else:
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model.pth'), map_location=device))
        print(f"  모델 로드: model.pth")
    model.eval()

    print(f"  모델 로드 완료!")
    return model, config


def load_test_samples(num_samples=5):
    """테스트용 샘플 로드"""
    print(f"\n테스트 샘플 로드 중...")

    chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, "sequences_chunk_*.pkl")))
    if not chunk_files:
        print("  [ERROR] 청크 파일 없음")
        return []

    # 마지막 청크 사용
    with open(chunk_files[-1], 'rb') as f:
        chunk = pickle.load(f)

    # 랜덤 샘플 선택
    np.random.seed(42)
    indices = np.random.choice(len(chunk), min(num_samples, len(chunk)), replace=False)
    samples = [chunk[i] for i in indices]

    print(f"  {len(samples)}개 샘플 로드됨")
    return samples


def run_inference(model, sample, sample_idx, device):
    """단일 샘플에 대해 추론 실행"""
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx} 추론")
    print(f"{'='*60}")

    inp = sample['input']  # (50, 4): lat, lon, sog, cog
    target = sample['target']  # (20, 2): lat, lon

    # 현재 속력/방향 정보
    current_sog = inp[-1, 2]
    current_cog = inp[-1, 3]
    avg_sog = np.mean(inp[:, 2])

    print(f"  현재 속력(SOG): {current_sog:.1f} knots")
    print(f"  평균 속력: {avg_sog:.1f} knots")
    print(f"  현재 방향(COG): {current_cog:.1f}°")

    # 히스토리 포인트
    history_points = [(inp[i, 0], inp[i, 1]) for i in range(SEQ_LEN)]

    # 실제 미래 포인트
    actual_points = [(target[i, 0], target[i, 1]) for i in range(PRED_LEN)]

    # 입력 정규화
    inp_norm = normalize_input(inp)
    inp_tensor = torch.from_numpy(inp_norm).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        pred_delta = model(inp_tensor)  # (1, PRED_LEN, 2)

    # 예측 좌표 복원
    pred_delta_np = pred_delta.cpu().numpy()[0] / 100  # 스케일 복원
    last_pos = inp[-1, :2]

    predicted_points = []
    for t in range(PRED_LEN):
        pred_lat = last_pos[0] + pred_delta_np[t, 0]
        pred_lon = last_pos[1] + pred_delta_np[t, 1]
        predicted_points.append((pred_lat, pred_lon))

    # 스무딩
    predicted_smooth = smooth_path(predicted_points)

    # 오차 계산
    errors = []
    for pred, act in zip(predicted_points, actual_points):
        err = haversine(pred[0], pred[1], act[0], act[1])
        errors.append(err)

    print(f"  예측 오차:")
    print(f"    평균: {np.mean(errors):.2f} km")
    print(f"    5분 후: {errors[4]:.2f} km")
    print(f"    10분 후: {errors[9]:.2f} km")
    print(f"    15분 후: {errors[14]:.2f} km")
    print(f"    20분 후: {errors[19]:.2f} km")

    # HTML 생성
    output_path = f"prediction_v8_direct_sample{sample_idx}.html"
    generate_html(sample_idx, history_points, actual_points, predicted_points,
                  predicted_smooth, errors, output_path,
                  current_sog=current_sog, current_cog=current_cog, avg_sog=avg_sog)

    return errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V8 Direct Transformer Inference')
    parser.add_argument('--num_samples', type=int, default=5, help='테스트 샘플 수')
    args = parser.parse_args()

    print("=" * 60)
    print("V8 Direct Coordinate Transformer 추론")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 모델 로드
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] 모델 디렉토리 없음: {MODEL_DIR}")
        return

    model, config = load_model(device)

    # 테스트 샘플 로드
    samples = load_test_samples(num_samples=args.num_samples)
    if not samples:
        return

    # 추론 실행
    all_errors = []
    for i, sample in enumerate(samples):
        errors = run_inference(model, sample, i, device)
        all_errors.append(errors)

    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("전체 결과 요약")
    print("=" * 60)

    all_errors = np.array(all_errors)
    print(f"샘플 수: {len(all_errors)}")
    print(f"평균 오차: {np.mean(all_errors):.2f} km")
    print(f"5분 후 평균: {np.mean(all_errors[:, 4]):.2f} km")
    print(f"10분 후 평균: {np.mean(all_errors[:, 9]):.2f} km")
    print(f"15분 후 평균: {np.mean(all_errors[:, 14]):.2f} km")
    print(f"20분 후 평균: {np.mean(all_errors[:, 19]):.2f} km")


if __name__ == "__main__":
    main()
