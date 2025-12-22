import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 하이퍼파라미터
# =========================
SEQ_LEN    = 80
STEP_SIZE  = 3
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3

SAVE_DIR = "global_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 좌우 반전 보정(남북 유지, 동서 반전)
# - 증상: 좌우만 반대로 가는 경우 True로 두는 게 맞음
COG_MIRROR = True

# (선택) 폭주 방지
GRAD_CLIP_NORM = 1.0


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

    keep_cols = [c for c in df.columns if c in ["datetime","mmsi","lat","lon","sog","cog","fid"]]
    df = df[keep_cols].dropna(subset=["datetime", "lat", "lon", "sog", "cog"])
    if df.empty:
        return None

    dt_range = pd.date_range(
        start=df["datetime"].iloc[0].floor("T"),
        end=df["datetime"].iloc[-1].ceil("T"),
        freq="1min"
    )

    range_df = pd.DataFrame({"datetime": dt_range})
    # ✅ segment 단위로 mmsi/fid 고정
    range_df["mmsi"] = df["mmsi"].iloc[0] if "mmsi" in df.columns else np.nan
    range_df["fid"]  = df["fid"].iloc[0]  if "fid"  in df.columns else np.nan

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

    exclude_cols = ["mmsi", "fid"]
    convert_cols = [c for c in merge_df.columns if c not in exclude_cols]
    merge_df[convert_cols] = merge_df[convert_cols].astype("float")

    intp_df = merge_df.interpolate(method="linear")
    intp_df["cog"] = np.degrees(np.arctan2(intp_df["sin_course"], intp_df["cos_course"]))
    intp_df["cog"] = (intp_df["cog"] + 360) % 360

    intp_df = intp_df.drop(columns=["sin_course","cos_course"], errors="ignore").reset_index()
    intp_df = intp_df.dropna(subset=["lat","lon","sog","cog"])
    return intp_df

# =========================
# 2. Dataset / Model (segment-aware + smoothness 지원)
# =========================
class TrajectoryDataset(Dataset):
    """
    - segment_bounds: [(s,e), ...]  (intp_all 기준)
    - start_indices를 segment 별로 만들어두고,
    - train/val split은 segment 단위로 수행 가능하도록 지원
    """
    def __init__(self, df, seq_len=80, step=3, segment_bounds=None, cog_mirror=False):
        self.seq_len = seq_len
        self.step = step
        self.cog_mirror = cog_mirror

        df = df.copy()

        sin_cog = np.sin(np.radians(df["cog"].values))
        cos_cog = np.cos(np.radians(df["cog"].values))
        if cog_mirror:
            sin_cog = -sin_cog  # 좌우반전 보정

        df["sin_cog"] = sin_cog
        df["cos_cog"] = cos_cog

        self.feature_cols = ["lat","lon","sog","sin_cog","cos_cog"]
        self.target_cols  = ["lat","lon","sog","sin_cog","cos_cog"]

        X = df[self.feature_cols].values.astype(np.float32)
        Y = df[self.target_cols].values.astype(np.float32)

        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std  = X.std(axis=0, keepdims=True) + 1e-6
        self.y_mean = Y.mean(axis=0, keepdims=True)
        self.y_std  = Y.std(axis=0, keepdims=True) + 1e-6

        self.Xn = (X - self.x_mean) / self.x_std
        self.Yn = (Y - self.y_mean) / self.y_std

        if segment_bounds is None:
            segment_bounds = [(0, len(self.Xn))]
        self.segment_bounds = segment_bounds

        # segment별 start index 목록을 따로 저장 (segment split용)
        self.segment_starts = []  # list[list[int]]
        for (s, e) in self.segment_bounds:
            starts = []
            max_start = e - 1 - self.seq_len
            if max_start >= s:
                for i in range(s, max_start + 1, self.step):
                    starts.append(i)
            self.segment_starts.append(starts)

        # 기본은 전체 flatten
        self.start_indices = [i for starts in self.segment_starts for i in starts]

    def set_active_segments(self, active_segment_ids):
        """train/val을 segment 단위로 나눈 뒤, 해당 segment들만 사용하도록 설정"""
        self.start_indices = []
        for sid in active_segment_ids:
            self.start_indices.extend(self.segment_starts[sid])

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        s = self.start_indices[idx]
        e = s + self.seq_len
        x = self.Xn[s:e]     # (seq_len,5)
        y = self.Yn[e]       # (5,)
        x_last = self.Xn[e-1]  # (5,)  smoothness에 필요: 직전 상태
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_last)


class LSTMTrajectoryModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, output_dim=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================
# Loss: weighted mse + smoothness penalty + 변침 학습 강화
# =========================
def loss_with_smoothness(y_pred, y_true, x_last,
                         w_mse=(2, 2, 1, 3, 3),  # sin/cos 가중치 증가 (1->3)
                         smooth_lambda=0.05,
                         sog_lambda=0.10,
                         heading_lambda=0.02,  # 침로 smoothness 감소 (0.05->0.02)
                         turn_boost=2.0):  # 변침 구간 가중치
    """
    y_pred, y_true: (B,5)  [lat, lon, sog, sin, cos]
    x_last: (B,5)  직전 상태(정규화된 값)

    변침 학습 강화:
    - sin/cos 가중치 증가
    - heading smoothness 감소 (변침 허용)
    - 변침 구간(침로 변화가 큰 샘플)에 추가 가중치
    """

    # (1) weighted MSE
    w = torch.tensor(w_mse, device=y_pred.device, dtype=y_pred.dtype).view(1, -1)

    # 변침 구간 감지: 실제 침로 변화량 계산
    # x_last와 y_true의 sin/cos 차이가 크면 변침 구간
    true_heading_change = ((y_true[:, 3:5] - x_last[:, 3:5]) ** 2).sum(dim=1).sqrt()

    # 변침 구간에 추가 가중치 (변화량이 클수록 더 중요하게 학습)
    turn_weight = 1.0 + turn_boost * true_heading_change
    turn_weight = turn_weight.unsqueeze(1)  # (B, 1)

    # 가중 MSE (변침 구간 강조)
    mse = ((y_pred - y_true) ** 2 * w * turn_weight).mean()

    # (2) smoothness: 직진 구간에서만 적용 (변침 구간에서는 약하게)
    dsog = (y_pred[:, 2] - x_last[:, 2]).abs().mean()
    dheading = (y_pred[:, 3:5] - x_last[:, 3:5]).abs().mean()

    smooth = sog_lambda * dsog + heading_lambda * dheading
    return mse + smooth_lambda * smooth


# =========================
# 3. Global 학습 함수 (segment split + early stop + scheduler + smoothness)
# =========================
def train_global_model(
    df_all,
    seq_len=SEQ_LEN, step_size=STEP_SIZE,
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
    heading_lambda=0.02,  # 침로 smoothness 감소 (변침 허용)
    turn_boost=2.0,       # 변침 구간 가중치 (클수록 변침 학습 강화)

    grad_clip_norm=1.0,
    val_ratio=0.2,
    seed=42,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # 데이터 준비 + 보간 + segment_bounds 생성 (네 코드 그대로)
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

    print(f"[GLOBAL] 원본 rows={len(df_all)}, fid={df_all.fid.nunique()}, mmsi={df_all.mmsi.nunique()}")

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
        raise RuntimeError("[GLOBAL] 보간된 segment가 없습니다.")

    intp_all = pd.concat(intp_segments, ignore_index=True)

    segment_bounds = []
    s = 0
    for L in seg_lengths:
        e = s + L
        segment_bounds.append((s, e))
        s = e

    print(f"[GLOBAL] 보간 후 rows={len(intp_all)}, segments={len(segment_bounds)}")

    # --------------------
    # ✅ 항로 범위 계산 (추론 시 육지 침범 방지용)
    # --------------------
    lat_min, lat_max = intp_all["lat"].min(), intp_all["lat"].max()
    lon_min, lon_max = intp_all["lon"].min(), intp_all["lon"].max()
    # 약간의 마진 추가 (0.01도 ≈ 1.1km)
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    lat_bounds = (lat_min - lat_margin, lat_max + lat_margin)
    lon_bounds = (lon_min - lon_margin, lon_max + lon_margin)
    print(f"[GLOBAL] 항로 범위: lat={lat_bounds[0]:.4f}~{lat_bounds[1]:.4f}, lon={lon_bounds[0]:.4f}~{lon_bounds[1]:.4f}")

    # --------------------
    # ✅ Dataset 생성 (segment-aware)
    # --------------------
    dataset = TrajectoryDataset(
        intp_all,
        seq_len=seq_len,
        step=step_size,
        segment_bounds=segment_bounds,
        cog_mirror=cog_mirror,
    )

    n_segments = len(dataset.segment_starts)
    if n_segments <= 1:
        raise RuntimeError(f"[GLOBAL] segments={n_segments} 너무 적습니다. split 불가")

    # --------------------
    # ✅ Segment 단위 train/val split
    # --------------------
    rng = np.random.default_rng(seed)
    seg_ids = np.arange(n_segments)
    rng.shuffle(seg_ids)

    n_val_seg = max(1, int(n_segments * val_ratio))
    val_seg_ids = seg_ids[:n_val_seg].tolist()
    train_seg_ids = seg_ids[n_val_seg:].tolist()

    # train용 dataset / val용 dataset을 복제해서 사용
    train_ds = dataset
    val_ds = TrajectoryDataset(
        intp_all,
        seq_len=seq_len,
        step=step_size,
        segment_bounds=segment_bounds,
        cog_mirror=cog_mirror,
    )

    train_ds.set_active_segments(train_seg_ids)
    val_ds.set_active_segments(val_seg_ids)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"[GLOBAL] train/val 시퀀스가 0입니다. (train={len(train_ds)}, val={len(val_ds)})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[GLOBAL] train segments={len(train_seg_ids)}, val segments={len(val_seg_ids)}")
    print(f"[GLOBAL] train seq={len(train_ds)}, val seq={len(val_ds)}")

    # --------------------
    # 모델/옵티마/스케줄러
    # --------------------
    model = LSTMTrajectoryModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr, verbose=True
        )

    # --------------------
    # Early stop
    # --------------------
    best_val = float("inf")
    best_epoch = -1
    bad_count = 0
    best_state = None

    print(f"[GLOBAL] 학습 시작 | max_epochs={epochs}, warmup={warmup_epochs}, patience={patience}, device={device}")

    for epoch in range(1, epochs + 1):
        # ---- train
        model.train()
        tr_loss = 0.0
        for x, y, x_last in train_loader:
            x, y, x_last = x.to(device), y.to(device), x_last.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
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
            tr_loss += loss.item() * x.size(0)

        tr_loss /= max(1, len(train_loader.dataset))

        # ---- val
        model.eval()
        va_loss = 0.0
        with torch.inference_mode():
            for x, y, x_last in val_loader:
                x, y, x_last = x.to(device), y.to(device), x_last.to(device)
                y_pred = model(x)
                loss = loss_with_smoothness(
                    y_pred, y, x_last,
                    smooth_lambda=smooth_lambda,
                    sog_lambda=sog_lambda,
                    heading_lambda=heading_lambda,
                    turn_boost=turn_boost,
                )
                va_loss += loss.item() * x.size(0)
        va_loss /= max(1, len(val_loader.dataset))

        cur_lr = optimizer.param_groups[0]["lr"]

        # Epoch 학습 결과 출력
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

        improved = (best_val - va_loss) > min_delta
        if improved:
            best_val = va_loss
            best_epoch = epoch
            bad_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_count += 1

        if epoch >= warmup_epochs and bad_count >= patience:
            print(f"[GLOBAL] 🛑 Early stop at epoch {epoch} (best={best_epoch}, val={best_val:.6f})")
            break

        if cur_lr <= min_lr + 1e-12 and epoch >= warmup_epochs:
            print(f"[GLOBAL] 🛑 Stop: lr reached min_lr (best={best_epoch}, val={best_val:.6f})")
            break

    # best 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    # 저장(best 기준)
    os.makedirs(save_dir, exist_ok=True)
    model_path  = os.path.join(save_dir, "lstm_global.pth")
    scaler_path = os.path.join(save_dir, "scaler_global.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(
        scaler_path,
        x_mean=train_ds.x_mean, x_std=train_ds.x_std,   # ✅ train 기준 스케일러(일관)
        y_mean=train_ds.y_mean, y_std=train_ds.y_std,
        seq_len=int(seq_len),
        step=int(step_size),
        feature_cols=np.array(train_ds.feature_cols),
        target_cols=np.array(train_ds.target_cols),
        cog_mirror=bool(cog_mirror),
        best_epoch=int(best_epoch),
        best_val=float(best_val),

        # smoothness meta
        smooth_lambda=float(smooth_lambda),
        sog_lambda=float(sog_lambda),
        heading_lambda=float(heading_lambda),

        # 항로 범위 (추론 시 육지 침범 방지용)
        lat_bounds=np.array(lat_bounds),
        lon_bounds=np.array(lon_bounds),
    )

    print(f"[GLOBAL] ✅ 저장 완료(best) epoch={best_epoch}, val={best_val:.6f}")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")

    del model, optimizer, train_loader, val_loader, train_ds, val_ds, dataset, intp_all
    gc.collect()
    return model_path, scaler_path


