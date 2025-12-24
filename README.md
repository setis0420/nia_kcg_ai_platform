# 선박 항적 예측 모델 V2

LSTM 기반 선박 항적 예측 모델입니다.

## 파일 구조

```
├── prepare_data.py      # 데이터 전처리 (1단계)
├── train_model.py       # 모델 학습 (2단계)
├── trajectory_inference_v2.py  # 추론 클래스
├── example_inference.py # 추론 예시
├── prepare_data.bat     # 전처리 배치 파일
├── train_model.bat      # 학습 배치 파일
└── prepared_data/       # 전처리된 데이터 (자동 생성)
```

## 사용 방법

### 1단계: 데이터 전처리

원본 CSV 데이터를 읽어서 보간, 정규화, 시퀀스 생성 후 npz 파일로 저장합니다.
**한 번만 실행하면 됩니다.**

```batch
prepare_data.bat
```

또는 직접 실행:
```powershell
python prepare_data.py ^
    --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" ^
    --transition_folder "area_transition_results" ^
    --output_dir prepared_data ^
    --seq_len 50 ^
    --stride 3 ^
    --grid_size 0.05
```

#### 전처리 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--data_folder` | 항적 CSV 파일 폴더 | (필수) |
| `--transition_folder` | 전이 정보 CSV 폴더 | (필수) |
| `--output_dir` | 출력 폴더 | `prepared_data` |
| `--seq_len` | 시퀀스 길이 | 50 |
| `--stride` | 슬라이딩 윈도우 이동 간격 | 3 |
| `--grid_size` | 격자 크기 (도) | 0.05 |

### 2단계: 모델 학습

전처리된 데이터를 로드하여 모델을 학습합니다.
**하이퍼파라미터 변경 시 이 단계만 다시 실행하면 됩니다.**

```batch
train_model.bat
```

또는 직접 실행:
```powershell
python train_model.py ^
    --data_dir prepared_data ^
    --epochs 300 ^
    --batch_size 256 ^
    --lr 0.001 ^
    --patience 20 ^
    --device cuda ^
    --save_dir global_model_v2
```

#### 학습 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--data_dir` | 전처리된 데이터 폴더 | `prepared_data` |
| `--epochs` | 최대 에폭 수 | 300 |
| `--batch_size` | 배치 크기 | 256 |
| `--lr` | 학습률 | 0.001 |
| `--patience` | Early stopping patience | 20 |
| `--warmup_epochs` | Warmup 에폭 수 | 30 |
| `--smooth_lambda` | Smoothness 정규화 계수 | 0.05 |
| `--heading_lambda` | 침로 smoothness 계수 | 0.02 |
| `--turn_boost` | 변침 구간 가중치 | 2.0 |
| `--embed_dim` | Embedding 차원 | 16 |
| `--val_ratio` | 검증 데이터 비율 | 0.2 |
| `--device` | 학습 장치 (cuda/cpu) | cuda |
| `--save_dir` | 모델 저장 폴더 | `global_model_v2` |

### 3단계: 추론

학습된 모델로 예측을 수행합니다.

```python
from trajectory_inference_v2 import TrajectoryInferenceV2
import pandas as pd

# 모델 로드
inferencer = TrajectoryInferenceV2(
    "global_model_v2/lstm_global_v2.pth",
    "global_model_v2/scaler_global_v2.npz"
)

# 입력 데이터 준비 (1분 간격 보간된 데이터)
# 필수 컬럼: datetime, lat, lon, sog, cog
df = pd.read_csv("your_trajectory.csv")

# 예측 (30스텝 = 30분)
predictions = inferencer.predict_multi_from_df(
    df,
    n_steps=30,
    sog_min_ratio=0.7  # SOG 최소 유지 비율
)

print(predictions)
# 출력: datetime, pred_lat, pred_lon, pred_sog, pred_cog
```

#### 추론 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_steps` | 예측 스텝 수 (분) | 80 |
| `mmsi` | 선박 MMSI (선택) | None |
| `start_area` | 출발 구역 (선택) | None |
| `end_area` | 도착 구역 (선택) | None |
| `sog_clip` | SOG 클리핑 범위 | (0.0, 35.0) |
| `sog_min_ratio` | 입력 SOG 대비 최소 비율 | 0.7 |
| `use_model_latlon` | 모델 위경도 직접 사용 | False |
| `enforce_bounds` | 항로 범위 제한 | True |

## 출력 파일

### 전처리 출력 (`prepared_data/`)
- `training_data.npz` - 학습 데이터 (Xn_num, X_cat, Yn)
- `meta.npz` - 메타 정보 (정규화 파라미터, vocab 등)
- `segment_starts.npy` - 세그먼트 시작 인덱스

### 학습 출력 (`global_model_v2/`)
- `lstm_global_v2.pth` - 최고 성능 모델
- `scaler_global_v2.npz` - 스케일러 및 메타 정보
- `checkpoints/` - 에폭별 체크포인트

## 모델 구조

- **입력 피처**
  - 수치형: lat, lon, sog, sin_cog, cos_cog (5개)
  - Categorical: mmsi_id, start_area_id, end_area_id, grid_id (4개, Embedding)

- **모델**: LSTM (hidden_dim=128, num_layers=2)

- **출력**: 다음 1분 후의 lat, lon, sog, sin_cog, cos_cog

## 주의사항

1. **데이터 형식**: 입력 데이터는 1분 간격으로 보간되어 있어야 합니다.
2. **시퀀스 길이**: 최소 `seq_len` (기본 50) 이상의 데이터가 필요합니다.
3. **GPU 메모리**: batch_size를 조절하여 GPU 메모리에 맞추세요.
4. **재학습**: 하이퍼파라미터만 변경 시 `train_model.py`만 다시 실행하면 됩니다.
