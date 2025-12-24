# -*- coding: utf-8 -*-
"""
선박 항적 예측 모델 V2 - Categorical 변수 및 격자 ID 추가
==========================================================
추가된 피처:
- mmsi: 선박 고유 ID (Embedding)
- start_area: 출발 구역 (Embedding)
- end_area: 도착 구역 (Embedding)
- grid_id: 현재 위치 기반 격자 ID (Embedding)

격자 크기: 0.05도 (위도/경도 각각)
"""

import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 하이퍼파라미터
# =========================
SEQ_LEN = 80
STRIDE  = 3    # Sliding window 이동 간격 (학습 데이터 추출 시)
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3

SAVE_DIR = "global_model_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# 좌우 반전 보정
COG_MIRROR = True

# 폭주 방지
GRAD_CLIP_NORM = 1.0

# 격자 크기 (도)
GRID_SIZE = 0.05


# =========================
# 격자 ID 계산 함수
# =========================
def compute_grid_id(lat, lon, lat_min, lon_min, grid_size=0.05):
    """
    위경도를 격자 ID로 변환
    - grid_row = (lat - lat_min) / grid_size
    - grid_col = (lon - lon_min) / grid_size
    - grid_id = grid_row * num_cols + grid_col
    """
    grid_row = ((lat - lat_min) / grid_size).astype(int)
    grid_col = ((lon - lon_min) / grid_size).astype(int)
    return grid_row, grid_col


def create_grid_mapping(lat_min, lat_max, lon_min, lon_max, grid_size=0.05):
    """격자 정보 생성"""
    num_rows = int(np.ceil((lat_max - lat_min) / grid_size)) + 1
    num_cols = int(np.ceil((lon_max - lon_min) / grid_size)) + 1
    return {
        'lat_min': lat_min,
        'lon_min': lon_min,
        'lat_max': lat_max,
        'lon_max': lon_max,
        'grid_size': grid_size,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'total_grids': num_rows * num_cols
    }


# =========================
# 1-A. segment 분리
# =========================
def split_by_gap(df, max_gap_days=1):
    if df is None or df.empty:
        return []
    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return []

    diff = df["datetime"].diff()
    new_seg = (diff >= pd.Timedelta(days=max_gap_days)) | diff.isna()
    seg_id = new_seg.cumsum()

    return [g.copy() for _, g in df.groupby(seg_id) if len(g) > 0]


# =========================
# 1-B. 1분 보간 (FULL series)
# =========================
def data_intp(df):
    if df is None or df.empty:
        return None

    df = df.drop_duplicates(subset=["datetime", "lat", "lon", "sog", "cog"], keep="first")
    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["lat", "lon", "sog", "cog"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # categorical 컬럼 보존
    keep_cols = [c for c in df.columns if c in ["datetime","mmsi","lat","lon","sog","cog","fid","start_area","end_area","mmsi_id","start_area_id","end_area_id"]]
    df = df[keep_cols].dropna(subset=["datetime", "lat", "lon", "sog", "cog"])
    if df.empty:
        return None

    dt_range = pd.date_range(
        start=df["datetime"].iloc[0].floor("T"),
        end=df["datetime"].iloc[-1].ceil("T"),
        freq="1min"
    )

    range_df = pd.DataFrame({"datetime": dt_range})

    # segment 단위로 categorical 값 고정
    range_df["mmsi"] = df["mmsi"].iloc[0] if "mmsi" in df.columns else np.nan
    range_df["fid"]  = df["fid"].iloc[0]  if "fid"  in df.columns else np.nan

    # categorical ID 컬럼 복사
    for cat_col in ["mmsi_id", "start_area_id", "end_area_id"]:
        if cat_col in df.columns:
            range_df[cat_col] = df[cat_col].iloc[0]

    merge_df = (
        pd.concat([df, range_df], axis=0)
          .set_index("datetime")
          .sort_index()
    )

    for col in ["lat", "lon", "sog", "cog"]:
        merge_df[col] = pd.to_numeric(merge_df[col], errors="coerce")

    # cog 보간 안정화: sin/cos 보간 후 각도로 복원
    merge_df["sin_course"] = np.sin(np.radians(merge_df["cog"]))
    merge_df["cos_course"] = np.cos(np.radians(merge_df["cog"]))

    # 수치형 컬럼만 보간
    numeric_cols = ["lat", "lon", "sog", "cog", "sin_course", "cos_course"]
    merge_df[numeric_cols] = merge_df[numeric_cols].astype("float")

    intp_df = merge_df.copy()
    intp_df[numeric_cols] = intp_df[numeric_cols].interpolate(method="linear")

    intp_df["cog"] = np.degrees(np.arctan2(intp_df["sin_course"], intp_df["cos_course"]))
    intp_df["cog"] = (intp_df["cog"] + 360) % 360

    intp_df = intp_df.drop(columns=["sin_course","cos_course"], errors="ignore").reset_index()
    intp_df = intp_df.dropna(subset=["lat","lon","sog","cog"])
    return intp_df


# =========================
# 2. Dataset (Categorical + Grid 지원)
# =========================
class TrajectoryDatasetV2(Dataset):
    """
    V2: Categorical 변수 및 격자 ID 추가

    입력 피처:
    - 수치형: lat, lon, sog, sin_cog, cos_cog (5개)
    - Categorical: mmsi_id, start_area_id, end_area_id, grid_id (4개)

    출력 타겟:
    - lat, lon, sog, sin_cog, cos_cog (5개)
    """
    def __init__(self, df, seq_len=80, stride=3, segment_bounds=None, cog_mirror=False,
                 grid_info=None):
        self.seq_len = seq_len
        self.stride = stride
        self.cog_mirror = cog_mirror
        self.grid_info = grid_info

        df = df.copy()

        sin_cog = np.sin(np.radians(df["cog"].values))
        cos_cog = np.cos(np.radians(df["cog"].values))
        if cog_mirror:
            sin_cog = -sin_cog

        df["sin_cog"] = sin_cog
        df["cos_cog"] = cos_cog

        # 격자 ID 계산
        if grid_info is not None:
            grid_row, grid_col = compute_grid_id(
                df["lat"].values, df["lon"].values,
                grid_info['lat_min'], grid_info['lon_min'],
                grid_info['grid_size']
            )
            # 범위 클리핑
            grid_row = np.clip(grid_row, 0, grid_info['num_rows'] - 1)
            grid_col = np.clip(grid_col, 0, grid_info['num_cols'] - 1)
            df["grid_id"] = grid_row * grid_info['num_cols'] + grid_col
        else:
            df["grid_id"] = 0

        self.numeric_cols = ["lat","lon","sog","sin_cog","cos_cog"]
        self.cat_cols = ["mmsi_id", "start_area_id", "end_area_id", "grid_id"]
        self.target_cols = ["lat","lon","sog","sin_cog","cos_cog"]

        # 수치형 데이터
        X_num = df[self.numeric_cols].values.astype(np.float32)
        Y = df[self.target_cols].values.astype(np.float32)

        # Categorical 데이터
        X_cat = df[self.cat_cols].values.astype(np.int64)

        # 정규화 (수치형만)
        self.x_mean = X_num.mean(axis=0, keepdims=True)
        self.x_std  = X_num.std(axis=0, keepdims=True) + 1e-6
        self.y_mean = Y.mean(axis=0, keepdims=True)
        self.y_std  = Y.std(axis=0, keepdims=True) + 1e-6

        self.Xn_num = (X_num - self.x_mean) / self.x_std
        self.X_cat = X_cat
        self.Yn = (Y - self.y_mean) / self.y_std

        if segment_bounds is None:
            segment_bounds = [(0, len(self.Xn_num))]
        self.segment_bounds = segment_bounds

        # segment별 start index 목록
        self.segment_starts = []
        for (s, e) in self.segment_bounds:
            starts = []
            max_start = e - 1 - self.seq_len
            if max_start >= s:
                for i in range(s, max_start + 1, self.stride):
                    starts.append(i)
            self.segment_starts.append(starts)

        self.start_indices = [i for starts in self.segment_starts for i in starts]

    def set_active_segments(self, active_segment_ids):
        """train/val을 segment 단위로 나눈 뒤, 해당 segment들만 사용"""
        self.start_indices = []
        for sid in active_segment_ids:
            self.start_indices.extend(self.segment_starts[sid])

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        s = self.start_indices[idx]
        e = s + self.seq_len

        # 수치형 시퀀스
        x_num = self.Xn_num[s:e]        # (seq_len, 5)

        # Categorical 시퀀스
        x_cat = self.X_cat[s:e]          # (seq_len, 4)

        # 타겟
        y = self.Yn[e]                   # (5,)
        x_last = self.Xn_num[e-1]        # (5,) smoothness용

        return (torch.from_numpy(x_num),
                torch.from_numpy(x_cat),
                torch.from_numpy(y),
                torch.from_numpy(x_last))


# =========================
# 3. Model (Embedding + LSTM)
# =========================
class LSTMTrajectoryModelV2(nn.Module):
    """
    V2: Categorical Embedding + LSTM

    구조:
    1. Categorical 변수들을 Embedding
    2. 수치형 + Embedding 결합
    3. LSTM 처리
    4. FC로 출력
    """
    def __init__(self,
                 num_features=5,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=5,
                 dropout=0.2,
                 # Embedding 설정
                 num_mmsi=1000,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16):
        super().__init__()

        self.num_features = num_features
        self.embed_dim = embed_dim

        # Embedding 레이어
        self.mmsi_embed = nn.Embedding(num_mmsi, embed_dim)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        # LSTM 입력 차원: 수치형(5) + 4개 embedding(각 embed_dim)
        lstm_input_dim = num_features + 4 * embed_dim

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_num, x_cat):
        """
        x_num: (B, seq_len, 5) - 정규화된 수치형 피처
        x_cat: (B, seq_len, 4) - categorical ID [mmsi_id, start_area_id, end_area_id, grid_id]
        """
        B, T, _ = x_num.shape

        # Embedding 적용
        mmsi_emb = self.mmsi_embed(x_cat[:, :, 0])           # (B, T, embed_dim)
        start_emb = self.start_area_embed(x_cat[:, :, 1])    # (B, T, embed_dim)
        end_emb = self.end_area_embed(x_cat[:, :, 2])        # (B, T, embed_dim)
        grid_emb = self.grid_embed(x_cat[:, :, 3])           # (B, T, embed_dim)

        # 결합
        x = torch.cat([x_num, mmsi_emb, start_emb, end_emb, grid_emb], dim=-1)

        # LSTM
        out, _ = self.lstm(x)

        # 마지막 타임스텝의 출력
        return self.fc(out[:, -1, :])


# =========================
# Loss 함수
# =========================
def loss_with_smoothness(y_pred, y_true, x_last,
                         w_mse=(2, 2, 1, 3, 3),
                         smooth_lambda=0.05,
                         sog_lambda=0.10,
                         heading_lambda=0.02,
                         turn_boost=2.0):
    """동일한 loss 함수 (V1과 호환)"""
    w = torch.tensor(w_mse, device=y_pred.device, dtype=y_pred.dtype).view(1, -1)

    # 변침 구간 감지
    true_heading_change = ((y_true[:, 3:5] - x_last[:, 3:5]) ** 2).sum(dim=1).sqrt()
    turn_weight = 1.0 + turn_boost * true_heading_change
    turn_weight = turn_weight.unsqueeze(1)

    # 가중 MSE
    mse = ((y_pred - y_true) ** 2 * w * turn_weight).mean()

    # smoothness
    dsog = (y_pred[:, 2] - x_last[:, 2]).abs().mean()
    dheading = (y_pred[:, 3:5] - x_last[:, 3:5]).abs().mean()

    smooth = sog_lambda * dsog + heading_lambda * dheading
    return mse + smooth_lambda * smooth


# =========================
# Categorical 인코딩 함수
# =========================
def encode_categorical(df, col_name, vocab=None):
    """
    문자열을 정수 ID로 인코딩
    vocab이 None이면 새로 생성
    """
    if vocab is None:
        unique_vals = df[col_name].unique()
        vocab = {v: i for i, v in enumerate(unique_vals)}

    df[f"{col_name}_id"] = df[col_name].map(vocab).fillna(0).astype(int)
    return vocab


# =========================
# 4. Global 학습 함수 V2
# =========================
def train_global_model_v2(
    df_all,
    seq_len=SEQ_LEN, stride=STRIDE,
    epochs=300,
    batch_size=BATCH_SIZE, lr=LR,
    save_dir=SAVE_DIR,
    device=None,
    cog_mirror=COG_MIRROR,

    # early stop
    patience=30,
    min_delta=1e-5,
    warmup_epochs=20,

    # scheduler
    use_scheduler=True,
    lr_patience=8,
    lr_factor=0.5,
    min_lr=1e-6,

    # smoothness & 변침 학습
    smooth_lambda=0.05,
    sog_lambda=0.10,
    heading_lambda=0.02,
    turn_boost=2.0,

    # embedding
    embed_dim=16,
    grid_size=GRID_SIZE,

    grad_clip_norm=1.0,
    val_ratio=0.2,
    seed=42,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # 데이터 준비
    # --------------------
    required = ["datetime","mmsi","lat","lon","sog","cog","fid"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"df_all에 필수 컬럼이 없습니다: {missing}")

    df_all = df_all.copy()
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    for c in ["lat","lon","sog","cog","mmsi","fid"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all = df_all.dropna(subset=["datetime","lat","lon","sog","cog","fid","mmsi"])
    df_all = df_all.sort_values(["fid","datetime"]).reset_index(drop=True)

    print(f"[GLOBAL-V2] 원본 rows={len(df_all)}, fid={df_all.fid.nunique()}, mmsi={df_all.mmsi.nunique()}")

    # --------------------
    # Categorical 인코딩
    # --------------------
    # start_area, end_area가 없으면 fid에서 추출 시도
    if "start_area" not in df_all.columns or "end_area" not in df_all.columns:
        print("[GLOBAL-V2] start_area/end_area 컬럼 없음 - 기본값 사용")
        df_all["start_area"] = "unknown"
        df_all["end_area"] = "unknown"

    mmsi_vocab = encode_categorical(df_all, "mmsi")
    start_vocab = encode_categorical(df_all, "start_area")
    end_vocab = encode_categorical(df_all, "end_area")

    print(f"[GLOBAL-V2] Categorical 인코딩 완료:")
    print(f"  - MMSI: {len(mmsi_vocab)} 종류")
    print(f"  - Start Area: {len(start_vocab)} 종류")
    print(f"  - End Area: {len(end_vocab)} 종류")

    # --------------------
    # 보간 + segment_bounds 생성
    # --------------------
    intp_segments, seg_lengths = [], []

    for fid, df_fid in df_all.groupby("fid"):
        segments_raw = split_by_gap(df_fid, max_gap_days=1)
        for seg_df in segments_raw:
            intp_df = data_intp(seg_df)
            if intp_df is None or len(intp_df) == 0:
                continue
            intp_df = intp_df.sort_values("datetime").reset_index(drop=True)
            intp_segments.append(intp_df)
            seg_lengths.append(len(intp_df))

    if len(intp_segments) == 0:
        raise RuntimeError("[GLOBAL-V2] 보간된 segment가 없습니다.")

    intp_all = pd.concat(intp_segments, ignore_index=True)

    segment_bounds = []
    s = 0
    for L in seg_lengths:
        e = s + L
        segment_bounds.append((s, e))
        s = e

    print(f"[GLOBAL-V2] 보간 후 rows={len(intp_all)}, segments={len(segment_bounds)}")

    # --------------------
    # 격자 정보 생성
    # --------------------
    lat_min, lat_max = intp_all["lat"].min(), intp_all["lat"].max()
    lon_min, lon_max = intp_all["lon"].min(), intp_all["lon"].max()

    # 마진 추가
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    lat_bounds = (lat_min - lat_margin, lat_max + lat_margin)
    lon_bounds = (lon_min - lon_margin, lon_max + lon_margin)

    grid_info = create_grid_mapping(
        lat_min - lat_margin, lat_max + lat_margin,
        lon_min - lon_margin, lon_max + lon_margin,
        grid_size
    )

    print(f"[GLOBAL-V2] 항로 범위: lat={lat_bounds[0]:.4f}~{lat_bounds[1]:.4f}, lon={lon_bounds[0]:.4f}~{lon_bounds[1]:.4f}")
    print(f"[GLOBAL-V2] 격자 정보: {grid_info['num_rows']}x{grid_info['num_cols']} = {grid_info['total_grids']} 격자 (크기: {grid_size}도)")

    # --------------------
    # Dataset 생성
    # --------------------
    dataset = TrajectoryDatasetV2(
        intp_all,
        seq_len=seq_len,
        stride=stride,
        segment_bounds=segment_bounds,
        cog_mirror=cog_mirror,
        grid_info=grid_info,
    )

    n_segments = len(dataset.segment_starts)
    if n_segments <= 1:
        raise RuntimeError(f"[GLOBAL-V2] segments={n_segments} 너무 적습니다.")

    # --------------------
    # Segment 단위 train/val split
    # --------------------
    rng = np.random.default_rng(seed)
    seg_ids = np.arange(n_segments)
    rng.shuffle(seg_ids)

    n_val_seg = max(1, int(n_segments * val_ratio))
    val_seg_ids = seg_ids[:n_val_seg].tolist()
    train_seg_ids = seg_ids[n_val_seg:].tolist()

    # train/val Dataset: 동일한 데이터 배열을 공유하고 인덱스만 분리
    # (메모리 절약: 데이터 복사 없이 view만 다르게 사용)
    train_ds = dataset
    train_ds.set_active_segments(train_seg_ids)

    # val_ds는 별도 객체지만 데이터 배열은 train_ds와 공유
    val_ds = object.__new__(TrajectoryDatasetV2)
    val_ds.seq_len = dataset.seq_len
    val_ds.stride = dataset.stride
    val_ds.cog_mirror = dataset.cog_mirror
    val_ds.grid_info = dataset.grid_info
    val_ds.numeric_cols = dataset.numeric_cols
    val_ds.cat_cols = dataset.cat_cols
    val_ds.target_cols = dataset.target_cols
    val_ds.x_mean = dataset.x_mean
    val_ds.x_std = dataset.x_std
    val_ds.y_mean = dataset.y_mean
    val_ds.y_std = dataset.y_std
    val_ds.Xn_num = dataset.Xn_num  # 공유 (복사 없음)
    val_ds.X_cat = dataset.X_cat    # 공유 (복사 없음)
    val_ds.Yn = dataset.Yn          # 공유 (복사 없음)
    val_ds.segment_bounds = dataset.segment_bounds
    val_ds.segment_starts = dataset.segment_starts
    val_ds.start_indices = []  # val용 인덱스
    val_ds.set_active_segments(val_seg_ids)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"[GLOBAL-V2] train/val 시퀀스가 0입니다.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[GLOBAL-V2] train segments={len(train_seg_ids)}, val segments={len(val_seg_ids)}")
    print(f"[GLOBAL-V2] train seq={len(train_ds)}, val seq={len(val_ds)}")

    # --------------------
    # 모델 생성
    # --------------------
    model = LSTMTrajectoryModelV2(
        num_features=5,
        hidden_dim=128,
        num_layers=2,
        output_dim=5,
        dropout=0.2,
        num_mmsi=len(mmsi_vocab) + 1,
        num_start_area=len(start_vocab) + 1,
        num_end_area=len(end_vocab) + 1,
        num_grids=grid_info['total_grids'] + 1,
        embed_dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr, verbose=True
        )

    # --------------------
    # 학습 루프
    # --------------------
    best_val = float("inf")
    best_epoch = -1
    bad_count = 0
    best_state = None

    # 체크포인트 저장을 위한 경로 미리 생성
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "lstm_global_v2.pth")  # best 모델
    scaler_path = os.path.join(save_dir, "scaler_global_v2.npz")

    print(f"[GLOBAL-V2] 학습 시작 | max_epochs={epochs}, warmup={warmup_epochs}, patience={patience}, device={device}")
    print(f"[GLOBAL-V2] Embedding dim={embed_dim}")
    print(f"[GLOBAL-V2] 체크포인트 저장 위치: {save_dir}")

    for epoch in range(1, epochs + 1):
        # ---- train
        model.train()
        tr_loss = 0.0
        for x_num, x_cat, y, x_last in train_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            y, x_last = y.to(device), x_last.to(device)

            optimizer.zero_grad()
            y_pred = model(x_num, x_cat)
            loss = loss_with_smoothness(
                y_pred, y, x_last,
                smooth_lambda=smooth_lambda,
                sog_lambda=sog_lambda,
                heading_lambda=heading_lambda,
                turn_boost=turn_boost,
            )
            loss.backward()

            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            tr_loss += loss.item() * x_num.size(0)

        tr_loss /= max(1, len(train_loader.dataset))

        # ---- val
        model.eval()
        va_loss = 0.0
        with torch.inference_mode():
            for x_num, x_cat, y, x_last in val_loader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                y, x_last = y.to(device), x_last.to(device)

                y_pred = model(x_num, x_cat)
                loss = loss_with_smoothness(
                    y_pred, y, x_last,
                    smooth_lambda=smooth_lambda,
                    sog_lambda=sog_lambda,
                    heading_lambda=heading_lambda,
                    turn_boost=turn_boost,
                )
                va_loss += loss.item() * x_num.size(0)
        va_loss /= max(1, len(val_loader.dataset))

        cur_lr = optimizer.param_groups[0]["lr"]

        # Epoch 결과 출력
        print("=" * 70)
        print(f"  Epoch: {epoch:03d} / {epochs}")
        print(f"  Train Loss: {tr_loss:.6f}")
        print(f"  Val Loss:   {va_loss:.6f}")
        print(f"  Learning Rate: {cur_lr:.2e}")
        if best_val < float("inf"):
            print(f"  Best Val Loss: {best_val:.6f} (Epoch {best_epoch})")
        print("=" * 70)

        if scheduler is not None:
            scheduler.step(va_loss)

        # ---- 매 epoch마다 체크포인트 저장 ----
        epoch_model_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
        epoch_scaler_path = os.path.join(checkpoint_dir, f"scaler_epoch_{epoch:03d}.npz")

        torch.save(model.state_dict(), epoch_model_path)
        np.savez(
            epoch_scaler_path,
            x_mean=train_ds.x_mean, x_std=train_ds.x_std,
            y_mean=train_ds.y_mean, y_std=train_ds.y_std,
            seq_len=int(seq_len),
            stride=int(stride),
            numeric_cols=np.array(train_ds.numeric_cols),
            cat_cols=np.array(train_ds.cat_cols),
            target_cols=np.array(train_ds.target_cols),
            mmsi_vocab=str(mmsi_vocab),
            start_vocab=str(start_vocab),
            end_vocab=str(end_vocab),
            num_mmsi=len(mmsi_vocab) + 1,
            num_start_area=len(start_vocab) + 1,
            num_end_area=len(end_vocab) + 1,
            grid_info_lat_min=grid_info['lat_min'],
            grid_info_lon_min=grid_info['lon_min'],
            grid_info_lat_max=grid_info['lat_max'],
            grid_info_lon_max=grid_info['lon_max'],
            grid_size=grid_info['grid_size'],
            num_rows=grid_info['num_rows'],
            num_cols=grid_info['num_cols'],
            total_grids=grid_info['total_grids'],
            cog_mirror=bool(cog_mirror),
            embed_dim=int(embed_dim),
            epoch=int(epoch),
            train_loss=float(tr_loss),
            val_loss=float(va_loss),
            smooth_lambda=float(smooth_lambda),
            sog_lambda=float(sog_lambda),
            heading_lambda=float(heading_lambda),
            lat_bounds=np.array(lat_bounds),
            lon_bounds=np.array(lon_bounds),
        )
        print(f"  [CHECKPOINT] epoch {epoch} 저장: {epoch_model_path}")

        improved = (best_val - va_loss) > min_delta
        if improved:
            best_val = va_loss
            best_epoch = epoch
            bad_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # best 모델 저장 (validation loss 개선 시)
            torch.save(model.state_dict(), model_path)
            np.savez(
                scaler_path,
                x_mean=train_ds.x_mean, x_std=train_ds.x_std,
                y_mean=train_ds.y_mean, y_std=train_ds.y_std,
                seq_len=int(seq_len),
                stride=int(stride),
                numeric_cols=np.array(train_ds.numeric_cols),
                cat_cols=np.array(train_ds.cat_cols),
                target_cols=np.array(train_ds.target_cols),
                mmsi_vocab=str(mmsi_vocab),
                start_vocab=str(start_vocab),
                end_vocab=str(end_vocab),
                num_mmsi=len(mmsi_vocab) + 1,
                num_start_area=len(start_vocab) + 1,
                num_end_area=len(end_vocab) + 1,
                grid_info_lat_min=grid_info['lat_min'],
                grid_info_lon_min=grid_info['lon_min'],
                grid_info_lat_max=grid_info['lat_max'],
                grid_info_lon_max=grid_info['lon_max'],
                grid_size=grid_info['grid_size'],
                num_rows=grid_info['num_rows'],
                num_cols=grid_info['num_cols'],
                total_grids=grid_info['total_grids'],
                cog_mirror=bool(cog_mirror),
                embed_dim=int(embed_dim),
                best_epoch=int(best_epoch),
                best_val=float(best_val),
                smooth_lambda=float(smooth_lambda),
                sog_lambda=float(sog_lambda),
                heading_lambda=float(heading_lambda),
                lat_bounds=np.array(lat_bounds),
                lon_bounds=np.array(lon_bounds),
            )
            print(f"  [BEST] 최고 모델 갱신됨 (epoch={epoch}, val_loss={va_loss:.6f})")
        else:
            bad_count += 1

        if epoch >= warmup_epochs and bad_count >= patience:
            print(f"[GLOBAL-V2] Early stop at epoch {epoch} (best={best_epoch}, val={best_val:.6f})")
            break

        if cur_lr <= min_lr + 1e-12 and epoch >= warmup_epochs:
            print(f"[GLOBAL-V2] Stop: lr reached min_lr (best={best_epoch}, val={best_val:.6f})")
            break

    # 학습 완료 메시지 (체크포인트는 이미 저장됨)
    print(f"\n[GLOBAL-V2] 학습 완료! best epoch={best_epoch}, val={best_val:.6f}")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")

    del model, optimizer, train_loader, val_loader, train_ds, val_ds, dataset, intp_all
    gc.collect()
    return model_path, scaler_path


if __name__ == "__main__":
    print("train_global_model_v2.py - Categorical 변수 및 격자 ID 지원")
    print("사용법: run_train_v2.py 참조")
