# 선박 항적 예측 프로그램 개발 로그

## 프로젝트 개요
- **목적**: 선박의 과거 항적 데이터를 기반으로 미래 항적을 예측하는 LSTM 모델 학습 및 추론
- **데이터**: 여수 해역 선박 AIS 데이터
- **모델**: LSTM 기반 시계열 예측 모델

---

## 주요 파일 구조

```
NIA_선박항적예측프로그램/
├── run_train.py           # 모델 학습 실행 파일 (CLI)
├── train_global_model.py  # 학습 핵심 로직
├── trajectory_inference.py # 추론 클래스
├── train.bat              # Windows 배치 파일 (인코딩 문제로 직접 사용 어려움)
├── area_transition_results/ # 구간별 전이 정보 CSV
└── global_model/          # 학습된 모델 저장 폴더
    ├── lstm_global.pth    # 모델 가중치
    └── scaler_global.npz  # 정규화 파라미터 + 메타정보
```

---

## 학습 실행 방법

### PowerShell에서 실행 (권장)
```powershell
& "C:\Users\user\anaconda3\envs\kki_gpu2\python.exe" run_train.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" --transition_folder "area_transition_results" --start_area "남쪽 진입" --end_area "여수정박지B" --epochs 200 --batch_size 256 --lr 0.001 --seq_len 50 --step_size 3 --patience 30 --warmup_epochs 30 --smooth_lambda 0.05 --heading_lambda 0.02 --turn_boost 3.0 --val_ratio 0.2 --device cuda --save_dir global_model
```

### 주요 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--data_folder` | (필수) | 항적 CSV 파일 폴더 |
| `--transition_folder` | (필수) | 전이 정보 CSV 폴더 |
| `--start_area` | (필수) | 시작 구역 이름 |
| `--end_area` | (필수) | 도착 구역 이름 |
| `--epochs` | 300 | 최대 학습 에폭 |
| `--batch_size` | 256 | 배치 크기 |
| `--lr` | 0.001 | 학습률 |
| `--seq_len` | 50 | 입력 시퀀스 길이 (분) |
| `--step_size` | 3 | 시간 간격 (분) - **추론 시 dt_minutes로 사용됨** |
| `--patience` | 20 | Early stopping patience |
| `--warmup_epochs` | 30 | Warmup 에폭 수 |
| `--smooth_lambda` | 0.05 | Smoothness 정규화 계수 |
| `--heading_lambda` | 0.02 | 침로 smoothness (작을수록 변침 허용) |
| `--turn_boost` | 2.0 | 변침 구간 가중치 (클수록 변침 학습 강화) |
| `--device` | cuda | 학습 장치 (cuda/cpu) |

---

## 해결한 문제들

### 1. 추론 시 이동 거리가 절반으로 줄어드는 문제
**원인**: 학습 시 `step_size=3` (3분 간격)으로 학습했는데, 추론 시 `dt_minutes=1`로 Dead Reckoning

**해결**:
- `scaler_global.npz`에 `step` 값 저장
- `TrajectoryInference`에서 자동으로 `dt_minutes` 로드
- 추론 시 `dt_minutes=None`이면 학습 시 사용한 step 값 사용

```python
# trajectory_inference.py
if "step" in ckpt:
    self.dt_minutes = int(ckpt["step"])
```

### 2. 변침(침로 변경) 학습이 안 되는 문제
**원인**: smoothness 제약이 침로 변화를 억제

**해결**:
- `heading_lambda`: 0.05 → 0.02 (침로 smoothness 완화)
- `w_mse`: sin/cos 가중치 1 → 3 (침로 예측 중요도 증가)
- `turn_boost`: 변침 구간 자동 감지하여 추가 가중치 부여

```python
# 변침 구간 감지
true_heading_change = ((y_true[:, 3:5] - x_last[:, 3:5]) ** 2).sum(dim=1).sqrt()
turn_weight = 1.0 + turn_boost * true_heading_change
```

### 3. 선박이 육지로 올라가는 문제
**원인**: 예측이 학습 데이터 범위를 벗어남

**해결**:
- 학습 시 위경도 범위 계산 및 저장 (`lat_bounds`, `lon_bounds`)
- 5% 마진 추가
- 추론 시 `enforce_bounds=True`로 경계 이탈 시 보정

```python
# 학습 시 범위 저장
lat_bounds = (lat_min - lat_margin, lat_max + lat_margin)
lon_bounds = (lon_min - lon_margin, lon_max + lon_margin)

# 추론 시 경계 체크 및 보정
if enforce_bounds:
    next_lat, next_lon, violated = self._clamp_to_bounds(next_lat, next_lon, cur_lat, cur_lon)
```

---

## 추론 사용법

```python
from trajectory_inference import TrajectoryInference

# 모델 로드
inferencer = TrajectoryInference(
    model_path="global_model/lstm_global.pth",
    scaler_path="global_model/scaler_global.npz"
)
# 자동 출력:
# [Inference] 항로 범위: lat=34.4521~34.8123, lon=127.6521~128.1234
# [Inference] seq_len=50, dt_minutes=3

# 예측 (30 step = 90분)
preds = inferencer.predict_multi_from_df(
    df,                    # 과거 항적 DataFrame (datetime, lat, lon, sog, cog 컬럼 필요)
    n_steps=30,            # 예측 스텝 수
    enforce_bounds=True    # 항로 범위 제약 적용
)

# 결과: datetime, pred_lat, pred_lon, pred_sog, pred_cog 컬럼
```

---

## scaler_global.npz 저장 내용

```python
np.savez(
    scaler_path,
    x_mean, x_std,           # 입력 정규화 파라미터
    y_mean, y_std,           # 출력 정규화 파라미터
    seq_len,                 # 시퀀스 길이
    step,                    # 시간 간격 (분) - 추론 시 dt_minutes로 사용
    feature_cols,            # 입력 피처 컬럼명
    target_cols,             # 출력 타겟 컬럼명
    cog_mirror,              # 좌우 반전 보정 여부
    best_epoch, best_val,    # 학습 결과
    smooth_lambda, sog_lambda, heading_lambda,  # smoothness 파라미터
    lat_bounds, lon_bounds,  # 항로 범위 (육지 침범 방지)
)
```

---

## 환경 정보

- **Python 환경**: `C:\Users\user\anaconda3\envs\kki_gpu2\python.exe`
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 사용 가능

---

## 추가 개선 아이디어

1. **항로 중심선 유도**: 학습 데이터의 평균 경로를 중심선으로 설정하고, 이탈 시 중심선으로 유도
2. **해역 경계 파일 사용**: `yeosu_zone.csv` 같은 해역 경계 파일로 육지/바다 구분
3. **Attention 메커니즘**: LSTM + Attention으로 변침 구간 더 잘 캡처
4. **앙상블**: 여러 모델 예측 평균으로 안정성 향상

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2024-12-22 | 프로젝트 초기 설정, run_train.py 생성 |
| 2024-12-22 | 추론 시 dt_minutes 자동 로드 기능 추가 |
| 2024-12-22 | 변침 학습 강화 (turn_boost, heading_lambda) |
| 2024-12-22 | 항로 범위 제약 기능 추가 (enforce_bounds) |
