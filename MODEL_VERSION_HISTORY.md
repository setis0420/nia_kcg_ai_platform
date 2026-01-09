# 선박 항적 예측 모델 버전 히스토리

## 버전 변경 요약

| 버전 | 격자 시스템 | 입력 특성 | 필터링 | 모델 구조 | 학습 방식 |
|------|------------|----------|--------|----------|----------|
| V4 | 정사각형 (0.002도, ~222m) | 격자 ID + 속도 벡터 | - | Seq2Seq | 전역 모델 |
| V5 | 정사각형 (0.002도, ~222m) | 격자 ID만 | SOG >= 8노트 | Transformer | MMSI별 개별 |
| V6 | H3 Hexagon (Res 9, ~700m) | H3 셀 ID만 | SOG >= 8노트 | Transformer | MMSI별 개별 |
| V7 | H3 Hexagon (Res 9, ~700m) | H3 + SOG + COG 임베딩 | SOG >= 8노트 + 인접 셀 필터 | Transformer | MMSI별 개별 |

---

## V4: 격자 기반 Seq2Seq 모델

### 특징
- **격자 시스템**: 정사각형 격자 (0.002도, 약 222m)
- **입력 특성**: 격자 ID + 속도 벡터 (Vx, Vy)
- **모델 구조**: Seq2Seq with Attention
- **학습 방식**: 전체 데이터 통합 학습 (전역 모델)

### 파라미터
```python
GRID_SIZE = 0.002  # 약 222m
SEQ_LEN = 50       # 입력 시퀀스 길이
```

### 파일
- `trajectory_inference_v4.py`
- `prepared_data_v4/`

---

## V5: LLM 스타일 격자 예측 모델

### 변경점 (V4 → V5)
- 속도/침로 벡터 완전 삭제 (격자 ID만 사용)
- 속력 8노트 이상만 필터링
- MMSI별 개별 모델 학습 및 저장
- Transformer Encoder 구조로 변경

### 특징
- **격자 시스템**: 정사각형 격자 (0.002도, 약 222m)
- **입력 특성**: 격자 ID 시퀀스만 (LLM 스타일)
- **모델 구조**: Transformer Encoder
- **학습 방식**: MMSI별 개별 모델

### 파라미터
```python
GRID_SIZE = 0.002   # 약 222m
MIN_SOG = 8.0       # 8노트 이상 필터
SEQ_LEN = 50        # 입력 시퀀스 길이
embed_dim = 64
num_heads = 4
num_layers = 3
dropout = 0.1
batch_size = 64
lr = 1e-3
patience = 15       # Early stopping
```

### 파일
- `run_train_v5.py`
- `run_train_v5_single.py`
- `run_train_v5_topk.py`
- `test_inference_v5_visual.py`
- `global_model_v5/`

---

## V6: H3 Hexagonal Grid 모델

### 변경점 (V5 → V6)
- 정사각형 격자 → H3 육각형 격자
- H3 Resolution 9 사용 (~700m 직경)
- 육각형 격자로 더 자연스러운 이동 표현

### 특징
- **격자 시스템**: H3 Hexagonal Grid (Resolution 9, ~700m)
- **입력 특성**: H3 셀 ID 시퀀스
- **모델 구조**: Transformer Encoder
- **학습 방식**: MMSI별 개별 모델

### 파라미터
```python
H3_RESOLUTION = 9   # ~700m 직경
MIN_SOG = 8.0       # 8노트 이상 필터
SEQ_LEN = 50        # 입력 시퀀스 길이
TOP_K = 10          # 상위 10개 MMSI 학습
MAX_FILES_PER_MMSI = 500
embed_dim = 64
num_heads = 4
num_layers = 3
dropout = 0.1
batch_size = 64
lr = 1e-3
patience = 15       # Early stopping
```

### 학습 결과 (V6)
| MMSI | Val Acc |
|------|---------|
| 441277000 | 49.8% |
| 440626000 | 67.4% |
| 440147330 | 75.3% |
| 440154030 | 77.4% |
| 440108410 | 79.6% |
| 440058060 | 68.0% |
| 440130300 | 75.9% |
| 440155010 | 75.1% |
| 440315060 | 78.6% |
| 440051760 | 69.5% |

### 파일
- `run_train_v6_h3.py`
- `test_inference_v6_visual.py`
- `global_model_v6_h3/`

---

## V7: 멀티 임베딩 모델 (H3 + SOG + COG)

### 변경점 (V6 → V7)
- H3 셀 임베딩 + SOG 구간 임베딩 + COG 구간 임베딩 합산
- SOG: 1노트 단위 구간 (8-9, 9-10, ..., 21+ = 14개)
- COG: 15도 단위 구간 (0-15, 15-30, ..., 345-360 = 24개)
- 인접하지 않은 셀 전이 필터링 (학습 데이터 정제)
- 경로 스무딩 (B-spline)

### 특징
- **격자 시스템**: H3 Hexagonal Grid (Resolution 9, ~700m)
- **입력 특성**: H3 셀 ID + SOG 구간 + COG 구간 (멀티 임베딩)
- **모델 구조**: Transformer Encoder (멀티 임베딩 합산)
- **학습 방식**: MMSI별 개별 모델 + 인접 셀 전이만 학습

### 파라미터
```python
H3_RESOLUTION = 9     # ~700m 직경
MIN_SOG = 8.0         # 8노트 이상 필터
MAX_SOG = 21.0        # 21노트 이상은 같은 구간
SEQ_LEN = 50          # 입력 시퀀스 길이
TOP_K = 10            # 상위 10개 MMSI 학습
MAX_FILES_PER_MMSI = 500

# SOG 구간: 8-9, 9-10, ..., 20-21, 21+ = 14개 (0~13)
NUM_SOG_BINS = 14

# COG 구간: 0-15, 15-30, ..., 345-360 = 24개 (0~23)
NUM_COG_BINS = 24

# 모델 파라미터
embed_dim = 64
num_heads = 4
num_layers = 3
dropout = 0.1
batch_size = 64
lr = 1e-3
patience = 15         # Early stopping
```

### 데이터 필터링 (V7 신규)
1. **속력 필터**: SOG >= 8노트
2. **인접 셀 전이 필터**: `h3.grid_disk(prev_cell, 1)` 사용
   - 연속된 포인트가 인접 셀이 아니면 세그먼트 분리
   - 세그먼트 경계를 넘는 시퀀스 학습 제외
   - 데이터 제거율: 45~80%

### 추론 시 인접 셀 제약
```python
def select_best_adjacent_cell(current_cell, logits, idx_to_cell, cell_to_idx):
    # 1차: 인접 셀 중 학습된 셀에서 최고 확률 선택
    # 2차: 인접 셀 중 학습된 셀이 없으면 모델 최고 확률 셀 방향으로 인접 이동
```

### 학습 결과 (V7)
| MMSI | V6 Acc | V7 Acc | 개선 |
|------|--------|--------|------|
| 441277000 | 49.8% | **90.5%** | +40.7%p |
| 440626000 | 67.4% | **94.1%** | +26.7%p |
| 440147330 | 75.3% | **92.1%** | +16.8%p |
| 440154030 | 77.4% | **94.4%** | +17.0%p |
| 440108410 | 79.6% | **95.8%** | +16.2%p |
| 440058060 | 68.0% | **96.6%** | +28.6%p |
| 440130300 | 75.9% | **96.2%** | +20.3%p |
| 440155010 | 75.1% | **94.1%** | +19.0%p |
| 440315060 | 78.6% | **93.7%** | +15.1%p |
| 440051760 | 69.5% | **94.8%** | +25.3%p |

**평균 정확도: 70% → 94% 향상**

### 추론 결과 (30분 예측)
| MMSI | 평균 오차 | 10분 후 | 20분 후 | 30분 후 |
|------|----------|---------|---------|---------|
| 440051760 | 0.15 km | 0.11 km | 0.18 km | 0.26 km |
| 440058060 | 2.29 km | 1.28 km | 2.95 km | 5.14 km |
| 440315060 | 3.89 km | 2.24 km | 5.10 km | 8.07 km |
| 440154030 | 4.46 km | 2.56 km | 5.87 km | 9.13 km |
| 440155010 | 4.48 km | 2.81 km | 5.91 km | 9.02 km |

### 파일
- `run_train_v7_multi.py`
- `test_inference_v7_visual.py`
- `global_model_v7_multi/`

---

## 모델 구조 비교

### V5/V6 모델 (단일 임베딩)
```python
class H3SequenceModel(nn.Module):
    def __init__(self, num_cells, embed_dim=64, num_heads=4, num_layers=3):
        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)
        self.transformer = nn.TransformerEncoder(...)
        self.fc = nn.Linear(embed_dim, num_cells)

    def forward(self, x):
        x_emb = self.cell_embed(x) + self.pos_embed(pos)
        out = self.transformer(x_emb)
        return self.fc(out[:, -1, :])
```

### V7 모델 (멀티 임베딩)
```python
class H3MultiEmbedModel(nn.Module):
    def __init__(self, num_cells, num_sog_bins=14, num_cog_bins=24, embed_dim=64):
        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.sog_embed = nn.Embedding(num_sog_bins, embed_dim)
        self.cog_embed = nn.Embedding(num_cog_bins, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)
        self.transformer = nn.TransformerEncoder(...)
        self.fc = nn.Linear(embed_dim, num_cells)

    def forward(self, cell_ids, sog_bins, cog_bins):
        # 멀티 임베딩 합산
        x = self.cell_embed(cell_ids) + self.sog_embed(sog_bins) + \
            self.cog_embed(cog_bins) + self.pos_embed(pos)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])
```

---

## 핵심 개선 포인트

1. **V4 → V5**: 속도 벡터 제거, LLM 스타일 학습, MMSI별 개별 모델
2. **V5 → V6**: 정사각형 → H3 육각형 격자 (자연스러운 이동 표현)
3. **V6 → V7**:
   - SOG/COG 임베딩 추가 (멀티모달)
   - 인접 셀 전이 필터링 (노이즈 데이터 제거)
   - 추론 시 인접 셀 제약 (물리적으로 가능한 이동만)
   - 경로 스무딩

---

## 저장 폴더 구조

```
global_model_v7_multi/
├── {MMSI}/
│   ├── model.pth          # 모델 가중치
│   ├── config.npz         # 설정 (seq_len, h3_resolution, num_cells 등)
│   └── cell_mapping.pkl   # H3 셀 ↔ 인덱스 매핑
```

---

## V8: Direct Coordinate Transformer

### 변경점 (V7 → V8)
- **격자 분류 → 좌표 직접 예측** (Regression)
- H3 셀 매핑 불필요 → 모델 크기 대폭 감소
- 연속 좌표 예측으로 정밀도 향상
- COG를 sin/cos로 분리하여 각도 연속성 확보

### 특징
- **입력**: (lat_rel, lon_rel, sog_norm, cog_sin, cog_cos) × 50 스텝
- **출력**: (Δlat, Δlon) × 20 스텝 (상대 변위)
- **모델 구조**: Transformer Encoder + MLP Head
- **손실 함수**: SmoothL1Loss (Huber Loss)

### 모델 구조
```python
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_heads=8,
                 num_layers=4, dropout=0.1, pred_len=20):
        # 입력 프로젝션
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 출력 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len * 2)  # (Δlat, Δlon) × pred_len
        )
```

### 입력 정규화
```python
def normalize_input(inp):
    # lat, lon: 첫 번째 위치 기준 상대 좌표 × 100
    inp_norm[:, 0] = (inp[:, 0] - inp[0, 0]) * 100
    inp_norm[:, 1] = (inp[:, 1] - inp[0, 1]) * 100

    # sog: 0-30 knots 정규화
    inp_norm[:, 2] = inp[:, 2] / 30.0

    # cog: sin/cos 변환 (각도 연속성)
    cog_rad = np.radians(inp[:, 3])
    inp_norm[:, 3] = np.sin(cog_rad)
    inp_norm[:, 4] = np.cos(cog_rad)
```

### 파라미터
```python
SEQ_LEN = 50          # 입력 시퀀스 길이
PRED_LEN = 20         # 예측 시퀀스 길이

# 모델 파라미터
input_dim = 5         # lat_rel, lon_rel, sog_norm, cog_sin, cog_cos
hidden_dim = 128
num_heads = 8
num_layers = 4
dropout = 0.1

# 학습 파라미터
batch_size = 256
lr = 1e-3
weight_decay = 0.01
loss = SmoothL1Loss   # Huber Loss
scheduler = CosineAnnealingLR
patience = 10         # Early stopping
```

### V7 vs V8 비교

| 항목 | V7 (격자 분류) | V8 (좌표 직접) |
|------|---------------|----------------|
| 출력 타입 | H3 셀 분류 | (Δlat, Δlon) 회귀 |
| 정밀도 | ~700m (셀 크기) | 연속 좌표 |
| 모델 크기 | ~380MB | ~수 MB |
| 손실 함수 | CrossEntropyLoss | SmoothL1Loss |
| 셀 매핑 | 필요 (cell_mapping.pkl) | 불필요 |
| 추론 속도 | 보통 | 빠름 |

### 파일
- `run_train_v8_direct.py`
- `test_inference_v8_direct.py`
- `global_model_v8_direct/`

### 학습 실행
```bash
# 전체 데이터 학습
"C:\Users\user\anaconda3\envs\kki_gpu2\python.exe" run_train_v8_direct.py

# 테스트 (일부 청크만)
"C:\Users\user\anaconda3\envs\kki_gpu2\python.exe" run_train_v8_direct.py --max_chunks 5 --epochs 10

# 옵션
--max_chunks N    # 최대 청크 수 (None=전체)
--epochs N        # 에포크 수 (기본: 50)
--batch_size N    # 배치 크기 (기본: 256)
--lr FLOAT        # 학습률 (기본: 0.001)
```

### 추론 실행
```bash
"C:\Users\user\anaconda3\envs\kki_gpu2\python.exe" test_inference_v8_direct.py --num_samples 10
```

### 저장 폴더 구조
```
global_model_v8_direct/
├── 울산/
│   ├── model.pth      # 모델 가중치 (~수 MB)
│   └── config.pkl     # 설정
```

---

## 모델 선택 가이드

| 상황 | 추천 모델 |
|------|----------|
| 정밀한 좌표 예측 필요 | **V8 Direct** |
| 해역별 패턴 분석 | V7 H3 격자 |
| 빠른 추론 속도 | **V8 Direct** |
| 모델 크기 제한 | **V8 Direct** |
| 불확실성 표현 필요 | V7 (Top-K 확률) |
