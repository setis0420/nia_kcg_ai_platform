# V10 Inertia-Aware Transformer - 선박 항적 예측

## 개요

V10 모델은 선박의 관성(Inertia)을 고려한 Transformer 기반 100분 항적 예측 모델입니다.

- **입력**: 과거 50분 항적 (lat, lon, sog, cog)
- **출력**: 미래 100분 예측 (lat, lon)
- **특징**: Inertia Loss + Smoothness Loss로 물리적으로 자연스러운 예측

## 파일 구조

```
NIA_선박항적예측프로그램/
├── preprocess_all_regions_v10.py   # 전처리 (병렬)
├── run_train_v10_region.py         # 학습 (지역별)
├── 학습데이터/                      # 원본 데이터
│   ├── 부산/
│   ├── 울산/
│   └── ...
├── 학습데이터_전처리/v10/           # 전처리된 데이터
│   ├── 부산/
│   ├── 울산/
│   └── ...
└── models/v10/                      # 학습된 모델
    ├── 부산/
    ├── 울산/
    └── ...
```

## 사용법

### 1. 전처리

```bash
# 모든 지역 전처리 (병렬)
python preprocess_all_regions_v10.py

# 특정 지역만 전처리
python preprocess_all_regions_v10.py --regions 부산 울산

# 지역 목록 확인
python preprocess_all_regions_v10.py --list

# 옵션
python preprocess_all_regions_v10.py --region_workers 4  # 동시 처리 지역 수
```

### 2. 학습

```bash
# 특정 지역 학습
python run_train_v10_region.py --region 부산

# 옵션 추가
python run_train_v10_region.py --region 울산 --epochs 100 --batch_size 256

# 사용 가능한 지역 확인
python run_train_v10_region.py --list

# 전체 옵션
python run_train_v10_region.py --region 부산 \
    --epochs 50 \
    --batch_size 128 \
    --lr 5e-4 \
    --inertia_weight 0.05 \
    --smoothness_weight 0.01
```

## 모델 아키텍처

```
입력 (50, 4)                      출력 (100, 2)
[lat, lon, sog, cog]              [Δlat, Δlon]
       ↓                                ↑
   정규화 (5차원)                    Reshape
       ↓                                ↑
 Input Projection              Output MLP (4층)
       ↓                                ↑
Positional Encoding                    ↑
       ↓                                ↑
 Transformer Encoder ──────────────────┘
   (6 layers)
```

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| Hidden Dim | 256 |
| Num Heads | 8 |
| Num Layers | 6 |
| Dropout | 0.1 |
| Learning Rate | 5e-4 |
| Batch Size | 128 |

## Loss 함수

총 Loss = L_coord + 0.05 × L_inertia + 0.01 × L_smooth

| Loss | 설명 | 계산 방식 |
|------|------|----------|
| **L_coord** | 좌표 예측 손실 | SmoothL1Loss |
| **L_inertia** | 관성 손실 (속도 연속성) | MSE |
| **L_smooth** | 부드러움 손실 (가속도 제한) | 제곱 평균 |

## 저장 파일

학습 완료 후 `models/v10/지역명/` 폴더에 저장:

```
models/v10/부산/
├── model_epoch001.pth   # 매 에포크 모델
├── model_epoch002.pth
├── ...
├── model_best.pth       # 최고 성능 모델
└── config.pkl           # 설정 정보
```

## 성능 (울산 기준)

| 예측 시간 | 평균 오차 |
|----------|----------|
| 10분 | ~0.3 km |
| 30분 | ~0.8 km |
| 60분 | ~1.5 km |
| 100분 | ~2.5 km |

## 요구사항

```
torch>=1.9.0
pandas
numpy
tqdm
pickle
```
