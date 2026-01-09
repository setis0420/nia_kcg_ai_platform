# -*- coding: utf-8 -*-
"""
V7 울산 모델 추론 및 시각화
===========================
- 전처리된 pkl 데이터 사용
- H3 + SOG + COG 멀티 임베딩
- 20 스텝 다중 예측
"""

import os
import numpy as np
import torch
import torch.nn as nn
import h3
import pickle
import glob
import warnings
from scipy.interpolate import splprep, splev

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"
MODEL_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\global_model_v7_multi\울산"
H3_RESOLUTION = 9
SEQ_LEN = 50
PRED_LEN = 20
NUM_SOG_BINS = 21
NUM_COG_BINS = 24


def sog_to_bin(sog):
    """SOG를 구간 인덱스로 변환"""
    if sog < 0:
        return 0
    bin_idx = int(sog)
    return min(bin_idx, NUM_SOG_BINS - 1)


def cog_to_bin(cog):
    """COG를 15도 단위 구간 인덱스로 변환"""
    cog = cog % 360
    return int(cog / 15) % NUM_COG_BINS


class H3MultiEmbedModel(nn.Module):
    """H3 + SOG + COG 멀티 임베딩 Transformer 모델"""
    def __init__(self, num_cells, num_sog_bins=NUM_SOG_BINS, num_cog_bins=NUM_COG_BINS,
                 embed_dim=128, num_heads=8, num_layers=4, dropout=0.1, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.num_cells = num_cells

        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.sog_embed = nn.Embedding(num_sog_bins, embed_dim)
        self.cog_embed = nn.Embedding(num_cog_bins, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_cells * pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_ids, sog_bins, cog_bins):
        B, T = cell_ids.shape
        pos = torch.arange(T, device=cell_ids.device).unsqueeze(0).expand(B, T)
        x = self.cell_embed(cell_ids) + self.sog_embed(sog_bins) + self.cog_embed(cog_bins) + self.pos_embed(pos)
        x = self.dropout(x)
        out = self.transformer(x)
        logits = self.fc(out[:, -1, :])
        return logits.view(B, self.pred_len, self.num_cells)


def haversine(lat1, lon1, lat2, lon2):
    """두 점 사이 거리 (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def smooth_path(points, num_smooth=100):
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


def generate_html(sample_idx, history_points, actual_points, predicted_points,
                  predicted_smooth, h3_cells_predicted, errors, output_path,
                  current_sog=0, current_cog=0, avg_sog=0):
    """HTML 시각화 생성"""

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

    # 오차 텍스트
    error_text = f"평균: {np.mean(errors):.2f}km"
    if len(errors) >= 10:
        error_text += f", 10분: {errors[9]:.2f}km"
    if len(errors) >= 20:
        error_text += f", 20분: {errors[19]:.2f}km"

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>V7 울산 모델 예측 - Sample {sample_idx}</title>
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
        .legend-hex {{ width: 20px; height: 20px; margin-right: 10px; background: rgba(255,100,100,0.2); border: 2px solid rgba(255,100,100,0.5); }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3 style="margin-top:0;">V7 울산 모델 예측</h3>
        <p><strong>Sample:</strong> {sample_idx}</p>
        <p><strong>모델:</strong> H3 Multi-Embed Transformer</p>
        <p><strong>H3 Resolution:</strong> {H3_RESOLUTION} (~700m)</p>
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
            <span>예측 경로 (스무딩)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: orange; border-style: dashed;"></div>
            <span>예측 (원본 H3 셀)</span>
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

        // 예측 H3 Hexagon
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

            L.circleMarker(historyPoints[historyPoints.length - 1], {{
                radius: 8, color: 'blue', fillColor: 'blue', fillOpacity: 1
            }}).addTo(map).bindPopup('히스토리 끝 / 예측 시작');
        }}

        if (predictedSmooth.length > 0) {{
            L.circleMarker(predictedSmooth[predictedSmooth.length - 1], {{
                radius: 8, color: 'red', fillColor: 'red', fillOpacity: 1
            }}).addTo(map).bindPopup('예측 종료 ({PRED_LEN}분)');
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


def load_model(device):
    """울산 모델 로드"""
    print(f"모델 로드 중: {MODEL_DIR}")

    # config 로드
    with open(os.path.join(MODEL_DIR, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    num_cells = config['num_cells']
    print(f"  H3 셀 수: {num_cells:,}")
    print(f"  Best Val Acc: {config.get('best_val_acc', 'N/A')}")

    # 셀 매핑 로드
    with open(os.path.join(MODEL_DIR, 'cell_mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)
    cell_to_idx = mapping['cell_to_idx']
    idx_to_cell = mapping['idx_to_cell']

    # 모델 로드
    model = H3MultiEmbedModel(num_cells=num_cells, pred_len=PRED_LEN).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model.pth'), map_location=device))
    model.eval()

    print(f"  모델 로드 완료!")
    return model, cell_to_idx, idx_to_cell, config


def load_test_samples(num_samples=5):
    """테스트용 샘플 로드"""
    print(f"\n테스트 샘플 로드 중...")

    # 마지막 청크에서 샘플 추출
    chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, "sequences_chunk_*.pkl")))
    if not chunk_files:
        print("  [ERROR] 청크 파일 없음")
        return []

    # 마지막 청크 사용 (학습에 덜 사용됐을 가능성)
    with open(chunk_files[-1], 'rb') as f:
        chunk = pickle.load(f)

    # 랜덤 샘플 선택
    np.random.seed(42)
    indices = np.random.choice(len(chunk), min(num_samples, len(chunk)), replace=False)
    samples = [chunk[i] for i in indices]

    print(f"  {len(samples)}개 샘플 로드됨")
    return samples


def run_inference(model, cell_to_idx, idx_to_cell, sample, sample_idx, device):
    """단일 샘플에 대해 추론 실행"""
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx} 추론")
    print(f"{'='*60}")

    inp = sample['input']  # (50, 4): lat, lon, sog, cog
    target = sample['target']  # (20, 2): lat, lon

    # 현재 속력/방향 정보 (마지막 입력 시점)
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

    # 입력 시퀀스를 H3 셀, SOG bin, COG bin으로 변환
    cell_indices = []
    sog_bins = []
    cog_bins = []

    for i in range(SEQ_LEN):
        lat, lon, sog, cog = inp[i]
        cell = h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION)
        cell_idx = cell_to_idx.get(cell, 0)
        cell_indices.append(cell_idx)
        sog_bins.append(sog_to_bin(sog))
        cog_bins.append(cog_to_bin(cog))

    # 추론
    with torch.no_grad():
        cell_x = torch.tensor([cell_indices], dtype=torch.long, device=device)
        sog_x = torch.tensor([sog_bins], dtype=torch.long, device=device)
        cog_x = torch.tensor([cog_bins], dtype=torch.long, device=device)

        logits = model(cell_x, sog_x, cog_x)  # (1, PRED_LEN, num_cells)

    # 예측 결과 변환
    predicted_points = []
    predicted_cells = []

    for t in range(PRED_LEN):
        pred_idx = logits[0, t, :].argmax().item()
        pred_cell = idx_to_cell.get(pred_idx, idx_to_cell.get(1))

        if pred_cell == '<UNK>':
            # UNK인 경우 이전 예측 또는 마지막 히스토리 사용
            if predicted_cells:
                pred_cell = predicted_cells[-1]
            else:
                pred_cell = h3.latlng_to_cell(float(inp[-1, 0]), float(inp[-1, 1]), H3_RESOLUTION)

        pred_lat, pred_lon = h3.cell_to_latlng(pred_cell)
        predicted_points.append((pred_lat, pred_lon))
        predicted_cells.append(pred_cell)

    # 스무딩
    predicted_smooth = smooth_path(predicted_points, num_smooth=50)

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
    output_path = f"prediction_v7_ulsan_sample{sample_idx}.html"
    generate_html(sample_idx, history_points, actual_points, predicted_points,
                  predicted_smooth, predicted_cells, errors, output_path,
                  current_sog=current_sog, current_cog=current_cog, avg_sog=avg_sog)

    return errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V7 Ulsan Model Inference')
    parser.add_argument('--num_samples', type=int, default=5, help='테스트 샘플 수')
    args = parser.parse_args()

    print("=" * 60)
    print("V7 울산 모델 추론 및 시각화")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 모델 로드
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] 모델 디렉토리 없음: {MODEL_DIR}")
        return

    model, cell_to_idx, idx_to_cell, config = load_model(device)

    # 테스트 샘플 로드
    samples = load_test_samples(num_samples=args.num_samples)
    if not samples:
        return

    # 추론 실행
    all_errors = []
    for i, sample in enumerate(samples):
        errors = run_inference(model, cell_to_idx, idx_to_cell, sample, i, device)
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
