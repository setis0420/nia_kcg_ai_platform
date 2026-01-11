# V9 Direct Coordinate Transformer - 100분 예측 모델

## 개요
- **모델명**: V9 Direct Coordinate Transformer
- **예측 시간**: 100분 (기존 V8의 20분에서 확장)
- **지역**: 울산
- **아키텍처**: Transformer Encoder + MLP

## 모델 구조

| 항목 | V8 (20분) | V9 (100분) |
|------|-----------|------------|
| 입력 | 50분 × 5차원 | 50분 × 5차원 |
| 출력 | 20분 × 2차원 | 100분 × 2차원 |
| hidden_dim | 128 | 256 |
| num_layers | 4 | 6 |
| num_heads | 8 | 8 |
| 파라미터 수 | ~1M | ~3M |

### 입력 피처 (5차원)
1. `lat_rel`: 상대 위도 (첫 위치 기준) × 100
2. `lon_rel`: 상대 경도 (첫 위치 기준) × 100
3. `sog_norm`: 속력 정규화 (0-30 knots)
4. `cog_sin`: 방향 sin 변환
5. `cog_cos`: 방향 cos 변환

### 출력
- (Δlat, Δlon) × 100: 마지막 입력 위치 기준 상대 변위

## 전처리

### 기존 문제점
- 기존 전처리(`pred_len=20`)에서 5개 시퀀스를 이어붙이려 했으나
- Sliding window(step=10) 방식으로 인해 시퀀스 간 불연속 발생
- `seq[i].target[-1]` → `seq[i+1].input[0]` 거리: ~5km (비연속!)

### 해결: 새로운 전처리
- `preprocess_ulsan_v3_100min.py` 생성
- `pred_len=100`으로 직접 100분 타겟 생성
- 출력 폴더: `학습데이터_전처리완료/울산_100min/`

### 전처리 결과
- 총 시퀀스: **18,236,248개**
- 입력 shape: (50, 4) - lat, lon, sog, cog
- 타겟 shape: (100, 2) - lat, lon
- 입력끝 → 타겟시작 거리: ~0.1km (연속성 확인)

## 학습 결과

### 학습 설정
- Epochs: 50 (early stopping)
- Batch size: 128
- Learning rate: 5e-4
- Optimizer: AdamW (weight_decay=0.01)
- Loss: SmoothL1Loss (Huber)
- Scheduler: CosineAnnealingLR

### 성능 (Validation)

| 예측 시간 | 평균 오차 |
|-----------|-----------|
| 10분 후 | 0.06 km |
| 30분 후 | 0.19 km |
| 60분 후 | 0.46 km |
| 100분 후 | 0.97 km |
| 전체 평균 | 0.41 km |

## 파일 구조

```
NIA_선박항적예측프로그램/
├── preprocess_ulsan_v3_100min.py    # 100분 전처리 스크립트
├── run_train_v9_ulsan_100_v2.py     # V9 학습 코드
├── test_inference_v9_visual_v2.py   # V9 추론 및 시각화
├── 학습데이터_전처리완료/
│   └── 울산_100min/                  # 100분 전처리 데이터
│       ├── sequences_chunk_000.pkl
│       ├── ...
│       ├── sequences_chunk_060.pkl
│       └── stats.pkl
└── global_model_v9_ulsan_100_v2/    # 학습된 모델
    ├── config.pkl
    ├── model_best.pth
    └── model_epoch*.pth
```

## 사용법

### 1. 전처리 (이미 완료)
```bash
python preprocess_ulsan_v3_100min.py
```

### 2. 학습
```bash
python run_train_v9_ulsan_100_v2.py --epochs 50 --batch_size 128
```

### 3. 추론 및 시각화
```bash
# 단일 선박 연속 항적 시각화 (30분 간격)
python test_inference_v9_visual_v2.py --num_chunks 2 --num_ships 3 --interval 30
```

### 출력 파일
- `prediction_v9_single_1.html`: 선박 #1 연속 항적 예측
- `prediction_v9_single_2.html`: 선박 #2 연속 항적 예측
- `prediction_v9_single_3.html`: 선박 #3 연속 항적 예측

## 시각화 설명

### HTML 맵 요소
- **파란선**: 입력 히스토리 (50분)
- **빨간 점선**: 예측 경로 (100분)
- **녹색 실선**: 실제 경로
- **파란 마커**: 예측 시작점 (클릭 시 상세 정보)
- **빨간 원**: 예측 종료점
- **녹색 원**: 실제 종료점

## 버전 히스토리

| 버전 | 예측시간 | 특징 |
|------|----------|------|
| V7 | 20분 | H3 격자 분류 |
| V8 | 20분 | 좌표 직접 예측 (Regression) |
| **V9** | **100분** | 좌표 직접 예측, 대형 모델 |
