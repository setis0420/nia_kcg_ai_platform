# V10 Inertia-Aware Transformer - 100분 예측 모델

## 개요
- **모델명**: V10 Inertia-Aware Transformer
- **예측 시간**: 100분
- **지역**: 울산
- **아키텍처**: Transformer Encoder + MLP (V9와 동일)
- **핵심 개선**: 선박 관성(Inertia) 반영 Loss 함수

## V9 대비 개선점

### 문제점 (V9)
- 예측 시작 시 급격한 침로(COG)/속력(SOG) 변화 발생
- 실제 선박은 관성으로 인해 방향/속도 변화에 시간이 걸림
- 물리적으로 비현실적인 예측 궤적

### 해결책 (V10)
- **Inertia Loss**: 입력 마지막 속도와 예측 첫 속도의 일관성 강제
- **Smoothness Loss**: 예측 초반 가속도 제한
- 추론 시간 영향 없음 (학습 시에만 적용)

## 모델 구조

| 항목 | V9 | V10 |
|------|-----|-----|
| 입력 | 50분 × 5차원 | 50분 × 5차원 |
| 출력 | 100분 × 2차원 | 100분 × 2차원 |
| hidden_dim | 256 | 256 |
| num_layers | 6 | 6 |
| num_heads | 8 | 8 |
| **Loss** | SmoothL1 | SmoothL1 + Inertia + Smoothness |

## Loss 함수

### 1. 기본 좌표 Loss (Coordinate Loss)
```python
loss_coord = SmoothL1Loss(pred_delta, target_delta)
```

### 2. Inertia Loss (관성 손실)
```python
def compute_inertia_loss(pred_delta, last_velocity):
    # 예측 첫 속도 = 첫 번째 위치 (0에서 시작)
    first_pred_velocity = pred_delta[:, 0, :]

    # 예측 두 번째 속도
    second_pred_velocity = pred_delta[:, 1, :] - pred_delta[:, 0, :]

    # 입력 마지막 속도와 예측 첫 속도 일관성
    loss1 = MSE(first_pred_velocity, last_velocity)

    # 예측 첫/두 번째 속도 일관성 (급격한 변화 방지)
    loss2 = MSE(first_pred_velocity, second_pred_velocity)

    return loss1 + 0.5 * loss2
```

### 3. Smoothness Loss (부드러움 손실)
```python
def compute_smoothness_loss(pred_delta, num_steps=10):
    # 속도 계산 (연속 점 사이 차이)
    velocities = pred_delta[:, 1:num_steps+1] - pred_delta[:, :num_steps]

    # 가속도 계산 (속도 변화)
    accelerations = velocities[:, 1:] - velocities[:, :-1]

    # 가속도 크기 최소화
    return mean(accelerations ** 2)
```

### 4. 총 손실
```python
total_loss = loss_coord + 0.05 * loss_inertia + 0.01 * loss_smooth
```

## 가중치 설정

| 가중치 | 초기값 | 수정값 | 비고 |
|--------|--------|--------|------|
| inertia_weight | 0.3 | **0.05** | 너무 높으면 모델이 평균으로 수렴 |
| smoothness_weight | 0.1 | **0.01** | 좌표 예측이 우선되어야 함 |

**주의**: 가중치가 너무 높으면 모델이 모든 입력에 대해 동일한 (평균적인) 출력을 생성합니다.

## 학습 설정

```bash
python run_train_v10_inertia.py --epochs 50 --batch_size 128 --lr 5e-4
```

| 설정 | 값 |
|------|-----|
| Epochs | 50 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience=15 |

## 파일 구조

```
NIA_선박항적예측프로그램/
├── run_train_v10_inertia.py      # V10 학습 코드
├── test_inference_v10_visual.py  # V10 추론 및 시각화
├── MODEL_V10_INERTIA.md          # V10 문서 (이 파일)
├── 학습데이터_전처리완료/
│   └── 울산_100min/              # 100분 전처리 데이터 (V9와 공유)
└── global_model_v10_inertia/     # 학습된 모델
    ├── config.pkl
    ├── model_best.pth
    └── model_epoch*.pth
```

## 사용법

### 1. 학습
```bash
# 기본 실행
python run_train_v10_inertia.py

# 가중치 조정
python run_train_v10_inertia.py --inertia_weight 0.05 --smoothness_weight 0.01

# 테스트 (청크 2개만)
python run_train_v10_inertia.py --max_chunks 2 --epochs 10
```

### 2. 추론 및 시각화
```bash
# 단일 선박 시각화
python test_inference_v10_visual.py --ship_idx 0 --interval 30

# 다른 선박
python test_inference_v10_visual.py --ship_idx 1 --interval 30
```

## 출력 파일
- `prediction_v10_single_1.html`: 선박 #1 연속 항적 예측

## 기대 효과

1. **물리적 현실성**: 예측 시작 시 급격한 방향/속도 변화 감소
2. **부드러운 궤적**: 초반 예측 구간의 자연스러운 곡선
3. **동일한 추론 속도**: 아키텍처 변경 없음

## 버전 히스토리

| 버전 | 예측시간 | 특징 |
|------|----------|------|
| V8 | 20분 | 좌표 직접 예측 (Regression) |
| V9 | 100분 | 좌표 직접 예측, 대형 모델 |
| **V10** | **100분** | **Inertia Loss로 관성 반영** |

## 트러블슈팅

### 모델이 모든 입력에 동일한 예측을 출력하는 경우
- **원인**: Inertia/Smoothness Loss 가중치가 너무 높음
- **해결**: 가중치를 낮춤 (inertia: 0.3→0.05, smoothness: 0.1→0.01)

### Loss가 V9보다 10배 이상 높은 경우
- **원인**: 가중치 불균형
- **해결**: 좌표 Loss가 주요 학습 신호가 되도록 가중치 조정
