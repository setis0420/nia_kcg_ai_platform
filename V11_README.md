# V11 선박 항적 예측 모델

## 개요

V11은 과거 30분의 선박 항적 데이터를 학습하여 미래 60분의 항적을 예측하는 Transformer 기반 딥러닝 모델입니다.

### V10 대비 변경사항

| 항목 | V10 | V11 |
|------|-----|-----|
| 입력 시퀀스 | 50분 | **30분** |
| 예측 시퀀스 | 100분 | **60분** |
| 최소 데이터 요구량 | 50개 | **30개** |
| hidden_dim | 256 | **192** |
| num_layers | 6 | **4** |
| num_heads | 8 | **6** |

## 파일 구조

```
├── preprocess_all_regions_v11.py  # 전처리 코드
├── run_train_v11_region.py        # 학습 코드
├── 학습데이터/                    # 원본 데이터
│   ├── 울산/
│   ├── 부산/
│   └── ...
├── 학습데이터_전처리/v11/         # 전처리된 데이터
│   ├── 울산/
│   │   ├── sequences_chunk_000.pkl
│   │   └── stats.pkl
│   └── ...
└── models/v11/                    # 학습된 모델
    ├── 울산/
    │   ├── model_best.pth
    │   └── config.pkl
    └── ...
```

## 사용법

### 1. 전처리

```bash
# 사용 가능한 지역 목록 확인
python preprocess_all_regions_v11.py --list

# 전체 지역 전처리
python preprocess_all_regions_v11.py

# 특정 지역만 전처리
python preprocess_all_regions_v11.py --regions 울산 부산

# 워커 수 조정
python preprocess_all_regions_v11.py --region_workers 2
```

#### 전처리 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--regions` | 전체 | 처리할 지역 목록 |
| `--seq_len` | 30 | 입력 시퀀스 길이 (분) |
| `--pred_len` | 60 | 예측 시퀀스 길이 (분) |
| `--step` | 15 | 슬라이딩 윈도우 스텝 |
| `--min_points` | 30 | 보간 후 최소 포인트 수 |
| `--region_workers` | 4 | 동시 처리 지역 수 |

### 2. 학습

```bash
# 사용 가능한 지역 목록 확인
python run_train_v11_region.py --list

# 특정 지역 학습
python run_train_v11_region.py --region 울산

# 에포크 수 조정
python run_train_v11_region.py --region 울산 --epochs 100

# 배치 크기 조정
python run_train_v11_region.py --region 울산 --batch_size 256
```

#### 학습 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--region` | - | 학습할 지역 이름 |
| `--epochs` | 50 | 에포크 수 |
| `--batch_size` | 128 | 배치 크기 |
| `--lr` | 5e-4 | 학습률 |
| `--inertia_weight` | 0.05 | Inertia Loss 가중치 |
| `--smoothness_weight` | 0.01 | Smoothness Loss 가중치 |
| `--max_chunks` | None | 최대 청크 수 (테스트용) |

## 데이터 형식

### 입력 데이터 (원본)

각 parquet/csv 파일에 필요한 컬럼:
- `datetime`: 시간 (필수)
- `lat`: 위도 (필수)
- `lon`: 경도 (필수)
- `sog`: 속력 knots (필수)
- `cog`: 침로 degrees (필수)
- `shiptype`: 선종 코드 (선택)
- `length`: 선박 길이 m (선택)

### 전처리 과정

1. **1분 간격 보간**
   - 불규칙한 시간 간격 → 균일한 1분 간격
   - lat, lon, sog: 선형 보간
   - cog: sin/cos 변환 후 보간 (0°↔360° 경계 문제 해결)

2. **필터링**
   - SOG: 0~30 knots 범위
   - 보간 후 최소 30개 포인트

3. **시퀀스 생성**
   - 입력: (30, 4) - [lat, lon, sog, cog] × 30분
   - 타겟: (60, 2) - [lat, lon] × 60분
   - 슬라이딩 윈도우: 15분 스텝

### 학습 시 정규화

```python
# 입력 정규화 (5차원)
inp_norm[:, 0] = (lat - lat[0]) * 100   # 상대 위도
inp_norm[:, 1] = (lon - lon[0]) * 100   # 상대 경도
inp_norm[:, 2] = sog / 30.0             # 정규화 SOG
inp_norm[:, 3] = sin(cog)               # COG sin
inp_norm[:, 4] = cos(cog)               # COG cos

# 타겟: 마지막 위치 기준 델타 × 100
delta_target = (target - last_pos) * 100
```

## 모델 아키텍처

```
TrajectoryTransformerV11
├── input_proj: Linear(5 → 192)
├── pos_encoding: Sinusoidal PE
├── transformer: TransformerEncoder
│   └── 4 layers × (192 dim, 6 heads, GELU)
└── output_mlp: 192 → 384 → 384 → 192 → 120
```

### Loss 함수

1. **Coordinate Loss (SmoothL1)**: 좌표 예측 오차
2. **Inertia Loss**: 관성 연속성 (첫 예측이 마지막 속도와 유사)
3. **Smoothness Loss**: 가속도 최소화

```
Total Loss = coord_loss + 0.05 × inertia_loss + 0.01 × smoothness_loss
```

## 출력 예시

### 전처리 결과

```
============================================================
V11 학습 데이터 전처리 - 30분→60분 예측
============================================================
입력: K:\...\학습데이터
출력: K:\...\학습데이터_전처리\v11
설정: 30분 입력 → 60분 예측

지역       총 파일      처리됨         시퀀스
----------------------------------------------------
울산        12,345       8,901        456,789
부산        10,234       7,654        345,678
----------------------------------------------------
합계        22,579      16,555        802,467
```

### 학습 결과

```
============================================================
V11 Inertia-Aware Transformer - 울산 60분 예측
============================================================
모델 파라미터: 2,345,678

Epoch  50 | Train: 0.012345 | Val: 0.015678

============================================================
최종 평가
============================================================
평균 오차: 1.23 km
10분 후: 0.45 km
30분 후: 0.89 km
60분 후: 1.78 km
```

## 요구사항

```
python >= 3.8
torch >= 1.12
pandas >= 1.4
numpy >= 1.21
tqdm
```
