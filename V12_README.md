# V12 선박 항적 예측 모델

## 개요

V12는 과거 30분의 선박 항적 데이터와 **선종/길이 정보**를 함께 학습하여 미래 60분의 항적을 예측하는 Transformer 기반 딥러닝 모델입니다.

### V11 대비 변경사항

| 항목 | V11 | V12 |
|------|-----|-----|
| 입력 특성 | 5차원 (lat, lon, sog, cog_sin, cog_cos) | **5차원 + 선종/길이 임베딩** |
| 선종 카테고리 | 미사용 | **5개 (화물선, 여객선, 유조선, 예부선, 기타선)** |
| 길이 카테고리 | 미사용 | **9개 (40m 단위)** |
| 메타 임베딩 | 없음 | **16차원 × 2 = 32차원** |

## 선종/길이 카테고리

### 선종 (5개)

| 카테고리 | 코드 범위 | 설명 |
|----------|-----------|------|
| 0 | 70-79 | 화물선 |
| 1 | 60-69 | 여객선 |
| 2 | 80-89 | 유조선 |
| 3 | 31-32 | 예부선 |
| 4 | 기타 | 기타선 |

### 길이 (9개, 40m 단위)

| 카테고리 | 범위 |
|----------|------|
| 0 | 0-40m |
| 1 | 40-80m |
| 2 | 80-120m |
| 3 | 120-160m |
| 4 | 160-200m |
| 5 | 200-240m |
| 6 | 240-280m |
| 7 | 280-320m |
| 8 | 320m+ |

## 파일 구조

```
├── preprocess_all_regions_v12.py  # 전처리 코드
├── run_train_v12_region.py        # 학습 코드
├── 학습데이터/                    # 원본 데이터
│   ├── 울산/
│   │   └── {mmsi}_{선종}_{길이}_{폭}_{시간}_{구역}.parquet
│   └── ...
├── 학습데이터_전처리/v12/         # 전처리된 데이터
│   ├── 울산/
│   │   ├── sequences_chunk_000.pkl
│   │   └── stats.pkl
│   └── ...
└── models/v12/                    # 학습된 모델
    ├── 울산/
    │   ├── model_best.pth
    │   ├── model_epoch001.pth
    │   └── config.pkl
    └── ...
```

## 파일명 형식

원본 데이터 파일명: `{mmsi}_{선종}_{길이}_{폭}_{시간}_{구역}.parquet`

예시: `205751000_80_180_28_20180620082849_ulsan.parquet`
- mmsi: 205751000
- 선종: 80 (유조선)
- 길이: 180m
- 폭: 28m
- 시간: 20180620082849
- 구역: ulsan

## 사용법

### 1. 전처리

```bash
# 사용 가능한 지역 목록 확인
python preprocess_all_regions_v12.py --list

# 전체 지역 전처리
python preprocess_all_regions_v12.py

# 특정 지역만 전처리
python preprocess_all_regions_v12.py --regions 울산 부산

# 워커 수 조정
python preprocess_all_regions_v12.py --regions 울산 --workers 4
```

#### 전처리 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--regions` | 전체 | 처리할 지역 목록 |
| `--seq_len` | 30 | 입력 시퀀스 길이 (분) |
| `--pred_len` | 60 | 예측 시퀀스 길이 (분) |
| `--step` | 15 | 슬라이딩 윈도우 스텝 |
| `--min_points` | 30 | 보간 후 최소 포인트 수 |
| `--workers` | 4 | 병렬 처리 워커 수 |

### 2. 학습

```bash
# 사용 가능한 지역 목록 확인
python run_train_v12_region.py --list

# 특정 지역 학습
python run_train_v12_region.py --region 울산

# 에포크 수 조정
python run_train_v12_region.py --region 울산 --epochs 100

# 배치 크기 조정
python run_train_v12_region.py --region 울산 --batch_size 256
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

### 전처리된 시퀀스 구조

```python
{
    'input': (30, 4),      # [lat, lon, sog, cog] × 30분
    'target': (60, 2),     # [lat, lon] × 60분
    'meta': [shiptype_cat, length_cat],  # 메타 정보
    'shiptype_cat': 0-4,   # 선종 카테고리
    'length_cat': 0-8      # 길이 카테고리
}
```

### 학습 시 정규화

```python
# 입력 정규화 (5차원)
inp_norm[:, 0] = (lat - lat[0]) * 100   # 상대 위도
inp_norm[:, 1] = (lon - lon[0]) * 100   # 상대 경도
inp_norm[:, 2] = sog / 30.0             # 정규화 SOG
inp_norm[:, 3] = sin(cog)               # COG sin
inp_norm[:, 4] = cos(cog)               # COG cos

# 메타 정보 (임베딩)
shiptype_embed = Embedding(5, 16)       # 선종 → 16차원
length_embed = Embedding(9, 16)         # 길이 → 16차원

# 타겟: 마지막 위치 기준 델타 × 100
delta_target = (target - last_pos) * 100
```

## 모델 아키텍처

```
TrajectoryTransformerV12
├── shiptype_embed: Embedding(5 → 16)
├── length_embed: Embedding(9 → 16)
├── input_proj: Linear(37 → 192)
│   └── 입력 5 + 선종 16 + 길이 16 = 37
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
V12 학습 데이터 전처리 - 선종/길이 포함
============================================================
입력: K:\...\학습데이터
출력: K:\...\학습데이터_전처리\v12
설정: 30분 입력 → 60분 예측
선종 카테고리: ['화물선', '여객선', '유조선', '예부선', '기타선']
길이 카테고리: ['0-40m', '40-80m', ..., '320m+']

지역       총 파일      처리됨         시퀀스
----------------------------------------------------
울산        12,345       8,901        456,789

  선종 분포:
    화물선: 5,234개
    여객선: 1,234개
    유조선: 987개
    예부선: 456개
    기타선: 990개

  길이 분포:
    0-40m: 2,345개
    40-80m: 3,456개
    ...
```

### 학습 결과

```
============================================================
V12 Inertia-Aware Transformer - 울산 60분 예측
============================================================
Device: cuda
선종 카테고리: 5개 ['화물선', '여객선', '유조선', '예부선', '기타선']
길이 카테고리: 9개
모델 파라미터: 2,456,789

Epoch  50 | Train: 0.012345 | Val: 0.015678

============================================================
최종 평가
============================================================
평균 오차: 1.15 km
10분 후: 0.42 km
30분 후: 0.85 km
60분 후: 1.65 km

----------------------------------------
선종별 60분 예측 오차
----------------------------------------
  화물선: 1.45 km (12,345개)
  여객선: 1.23 km (5,678개)
  유조선: 1.78 km (3,456개)
  예부선: 0.98 km (1,234개)
  기타선: 1.56 km (2,345개)

----------------------------------------
길이별 60분 예측 오차
----------------------------------------
  0-40m: 0.89 km (1,234개)
  40-80m: 1.12 km (2,345개)
  80-120m: 1.34 km (3,456개)
  ...
```

## 요구사항

```
python >= 3.8
torch >= 1.12
pandas >= 1.4
numpy >= 1.21
tqdm
```

## V11 → V12 마이그레이션

V12 모델은 선종/길이 정보를 추가로 필요로 합니다. 추론 시 선종/길이를 제공해야 합니다:

```python
# V11 (기존)
pred = model(input_tensor)

# V12 (새로운)
pred = model(input_tensor, shiptype_tensor, length_tensor)
```

추론 코드 예시:
```python
# 선종/길이를 모르는 경우 기본값 사용
shiptype = torch.tensor([4], device=device)  # 기타선
length = torch.tensor([0], device=device)    # 0-40m

pred = model(input_tensor, shiptype, length)
```
