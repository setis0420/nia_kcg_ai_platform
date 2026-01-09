# -*- coding: utf-8 -*-
"""
선박 항적 예측 모델 V5 - LLM 스타일 격자 ID 시퀀스 예측
==========================================================
핵심 변경:
- 속력/침로 벡터 완전 삭제
- 격자 ID만으로 다음 격자 ID 예측 (Language Model 스타일)
- 속력 8노트 이상만 필터링
- MMSI별로 개별 모델 학습 및 저장

입력: 격자 ID 시퀀스 (seq_len개)
출력: 다음 격자 ID (분류 문제)
"""

import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# =========================
# 하이퍼파라미터
# =========================
SEQ_LEN = 50       # 입력 시퀀스 길이
STRIDE = 1         # Sliding window 이동 간격
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

SAVE_DIR = "global_model_v5"
os.makedirs(SAVE_DIR, exist_ok=True)

# 격자 크기 (도)
GRID_SIZE = 0.01   # 약 1.1km

# 속력 필터 (노트)
MIN_SOG = 8.0


# =========================
# 격자 ID 계산 함수
# =========================
def compute_grid_id(lat, lon, lat_min, lon_min, grid_size, num_cols):
    """
    위경도를 격자 ID로 변환
    """
    grid_row = ((lat - lat_min) / grid_size).astype(int)
    grid_col = ((lon - lon_min) / grid_size).astype(int)
    grid_id = grid_row * num_cols + grid_col
    return grid_id


def create_grid_mapping(lat_min, lat_max, lon_min, lon_max, grid_size):
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


def grid_id_to_latlon(grid_id, grid_info):
    """격자 ID를 위경도 중심점으로 변환"""
    num_cols = grid_info['num_cols']
    grid_row = grid_id // num_cols
    grid_col = grid_id % num_cols

    lat = grid_info['lat_min'] + (grid_row + 0.5) * grid_info['grid_size']
    lon = grid_info['lon_min'] + (grid_col + 0.5) * grid_info['grid_size']
    return lat, lon


# =========================
# 1-A. segment 분리
# =========================
def split_by_gap(df, max_gap_minutes=10):
    """시간 간격이 큰 경우 세그먼트 분리"""
    if df is None or df.empty:
        return []
    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return []

    diff = df["datetime"].diff()
    new_seg = (diff >= pd.Timedelta(minutes=max_gap_minutes)) | diff.isna()
    seg_id = new_seg.cumsum()

    return [g.copy() for _, g in df.groupby(seg_id) if len(g) > 0]


# =========================
# 1-B. 1분 보간
# =========================
def data_intp(df):
    """1분 간격 보간"""
    if df is None or df.empty:
        return None

    df = df.drop_duplicates(subset=["datetime"], keep="first")
    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["lat", "lon", "sog"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "lat", "lon", "sog"])
    if df.empty:
        return None

    dt_range = pd.date_range(
        start=df["datetime"].iloc[0].floor("T"),
        end=df["datetime"].iloc[-1].ceil("T"),
        freq="1min"
    )

    range_df = pd.DataFrame({"datetime": dt_range})

    merge_df = (
        pd.concat([df[["datetime", "lat", "lon", "sog"]], range_df], axis=0)
          .set_index("datetime")
          .sort_index()
    )

    for col in ["lat", "lon", "sog"]:
        merge_df[col] = pd.to_numeric(merge_df[col], errors="coerce")

    intp_df = merge_df.interpolate(method="linear")
    intp_df = intp_df.reset_index()
    intp_df = intp_df.dropna(subset=["lat", "lon", "sog"])

    return intp_df


# =========================
# 2. Dataset (격자 ID 시퀀스)
# =========================
class GridSequenceDataset(Dataset):
    """
    LLM 스타일 격자 ID 시퀀스 데이터셋

    입력: 격자 ID 시퀀스 (seq_len개)
    출력: 다음 격자 ID (분류 타겟)
    """
    def __init__(self, grid_ids, seq_len=50, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.grid_ids = np.array(grid_ids, dtype=np.int64)

        # 시작 인덱스 목록 생성
        self.start_indices = []
        max_start = len(self.grid_ids) - self.seq_len - 1
        if max_start >= 0:
            for i in range(0, max_start + 1, self.stride):
                self.start_indices.append(i)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        s = self.start_indices[idx]
        e = s + self.seq_len

        # 입력: seq_len개 격자 ID
        x = self.grid_ids[s:e]

        # 타겟: 다음 격자 ID
        y = self.grid_ids[e]

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# =========================
# 3. Model (Transformer LM 스타일)
# =========================
class GridSequenceModel(nn.Module):
    """
    격자 ID 시퀀스 → 다음 격자 ID 예측 (Transformer 기반)
    """
    def __init__(self,
                 num_grids,
                 embed_dim=64,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1):
        super().__init__()

        self.num_grids = num_grids
        self.embed_dim = embed_dim

        # 격자 ID Embedding
        self.grid_embed = nn.Embedding(num_grids + 1, embed_dim)  # +1 for padding

        # Positional Embedding
        self.pos_embed = nn.Embedding(512, embed_dim)  # 최대 512 시퀀스 길이

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 출력층 (다음 격자 ID 분류)
        self.fc = nn.Linear(embed_dim, num_grids)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, seq_len) - 격자 ID 시퀀스
        return: (B, num_grids) - 다음 격자 ID 확률
        """
        B, T = x.shape

        # Position indices
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        # Embedding
        x_emb = self.grid_embed(x) + self.pos_embed(pos)
        x_emb = self.dropout(x_emb)

        # Transformer
        out = self.transformer(x_emb)

        # 마지막 위치의 출력으로 다음 격자 예측
        out = out[:, -1, :]  # (B, embed_dim)

        # 분류
        logits = self.fc(out)  # (B, num_grids)

        return logits


# =========================
# 4. MMSI별 학습 함수
# =========================
def train_mmsi_model(
    mmsi,
    df_mmsi,
    grid_info,
    seq_len=SEQ_LEN,
    stride=STRIDE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    save_dir=SAVE_DIR,
    device=None,
    patience=20,
    min_delta=1e-5,
    warmup_epochs=10,
    val_ratio=0.2,
    seed=42,
):
    """단일 MMSI에 대한 모델 학습"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"MMSI: {mmsi} 학습 시작")
    print(f"{'='*60}")

    # --------------------
    # 데이터 준비
    # --------------------
    df_mmsi = df_mmsi.copy()
    df_mmsi["datetime"] = pd.to_datetime(df_mmsi["datetime"], errors="coerce")
    for c in ["lat", "lon", "sog"]:
        df_mmsi[c] = pd.to_numeric(df_mmsi[c], errors="coerce")
    df_mmsi = df_mmsi.dropna(subset=["datetime", "lat", "lon", "sog"])

    # 속력 8노트 이상 필터링
    df_mmsi = df_mmsi[df_mmsi["sog"] >= MIN_SOG].reset_index(drop=True)

    if len(df_mmsi) < seq_len + 10:
        print(f"[SKIP] MMSI {mmsi}: 데이터 부족 ({len(df_mmsi)} rows)")
        return None, None

    print(f"[INFO] 속력 8노트 이상 필터링 후: {len(df_mmsi)} rows")

    # 격자 ID 계산
    grid_ids = compute_grid_id(
        df_mmsi["lat"].values,
        df_mmsi["lon"].values,
        grid_info['lat_min'],
        grid_info['lon_min'],
        grid_info['grid_size'],
        grid_info['num_cols']
    )
    grid_ids = np.clip(grid_ids, 0, grid_info['total_grids'] - 1)

    # 고유 격자 ID 확인
    unique_grids = np.unique(grid_ids)
    print(f"[INFO] 사용된 격자 수: {len(unique_grids)}")

    if len(unique_grids) < 5:
        print(f"[SKIP] MMSI {mmsi}: 격자 다양성 부족 ({len(unique_grids)})")
        return None, None

    # --------------------
    # 세그먼트 분리 및 보간
    # --------------------
    segments = split_by_gap(df_mmsi, max_gap_minutes=10)

    all_grid_ids = []
    for seg_df in segments:
        intp_df = data_intp(seg_df)
        if intp_df is None or len(intp_df) < seq_len + 1:
            continue

        # 보간된 데이터의 속력 필터링
        intp_df = intp_df[intp_df["sog"] >= MIN_SOG].reset_index(drop=True)
        if len(intp_df) < seq_len + 1:
            continue

        seg_grid_ids = compute_grid_id(
            intp_df["lat"].values,
            intp_df["lon"].values,
            grid_info['lat_min'],
            grid_info['lon_min'],
            grid_info['grid_size'],
            grid_info['num_cols']
        )
        seg_grid_ids = np.clip(seg_grid_ids, 0, grid_info['total_grids'] - 1)
        all_grid_ids.append(seg_grid_ids)

    if len(all_grid_ids) == 0:
        print(f"[SKIP] MMSI {mmsi}: 유효한 세그먼트 없음")
        return None, None

    # 모든 세그먼트 연결
    combined_grid_ids = np.concatenate(all_grid_ids)
    print(f"[INFO] 총 격자 시퀀스 길이: {len(combined_grid_ids)}")

    if len(combined_grid_ids) < seq_len + 10:
        print(f"[SKIP] MMSI {mmsi}: 시퀀스 부족")
        return None, None

    # --------------------
    # Dataset 생성
    # --------------------
    dataset = GridSequenceDataset(combined_grid_ids, seq_len=seq_len, stride=stride)

    if len(dataset) < 10:
        print(f"[SKIP] MMSI {mmsi}: 샘플 부족 ({len(dataset)})")
        return None, None

    # Train/Val 분리
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # --------------------
    # 모델 생성
    # --------------------
    model = GridSequenceModel(
        num_grids=grid_info['total_grids'],
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # --------------------
    # 학습 루프
    # --------------------
    best_val = float("inf")
    best_epoch = -1
    bad_count = 0
    best_state = None

    mmsi_save_dir = os.path.join(save_dir, str(mmsi))
    os.makedirs(mmsi_save_dir, exist_ok=True)

    model_path = os.path.join(mmsi_save_dir, "model.pth")
    config_path = os.path.join(mmsi_save_dir, "config.npz")

    for epoch in range(1, epochs + 1):
        # ---- Train
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_total += x.size(0)

        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        # ---- Val
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0

        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                va_loss += loss.item() * x.size(0)
                va_correct += (logits.argmax(dim=1) == y).sum().item()
                va_total += x.size(0)

        va_loss /= max(1, va_total)
        va_acc = va_correct / max(1, va_total)

        scheduler.step(va_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | "
                  f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.3f} | "
                  f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.3f}")

        # 개선 체크
        improved = (best_val - va_loss) > min_delta
        if improved:
            best_val = va_loss
            best_epoch = epoch
            bad_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_count += 1

        if epoch >= warmup_epochs and bad_count >= patience:
            print(f"  Early stop at epoch {epoch} (best={best_epoch})")
            break

    # --------------------
    # 모델 저장
    # --------------------
    if best_state is not None:
        torch.save(best_state, model_path)

        np.savez(
            config_path,
            mmsi=str(mmsi),
            seq_len=int(seq_len),
            grid_size=float(grid_info['grid_size']),
            lat_min=float(grid_info['lat_min']),
            lon_min=float(grid_info['lon_min']),
            lat_max=float(grid_info['lat_max']),
            lon_max=float(grid_info['lon_max']),
            num_rows=int(grid_info['num_rows']),
            num_cols=int(grid_info['num_cols']),
            total_grids=int(grid_info['total_grids']),
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            best_epoch=int(best_epoch),
            best_val_loss=float(best_val),
            min_sog=float(MIN_SOG),
        )

        print(f"[SAVED] MMSI {mmsi}: epoch={best_epoch}, val_loss={best_val:.4f}")
        print(f"  - {model_path}")
        print(f"  - {config_path}")

        return model_path, config_path
    else:
        print(f"[FAIL] MMSI {mmsi}: 학습 실패")
        return None, None


# =========================
# 5. 전체 학습 함수 (MMSI별)
# =========================
def train_all_mmsi_models(
    df_all,
    seq_len=SEQ_LEN,
    stride=STRIDE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    save_dir=SAVE_DIR,
    device=None,
    patience=20,
    warmup_epochs=10,
    val_ratio=0.2,
    grid_size=GRID_SIZE,
    min_sog=MIN_SOG,
    seed=42,
):
    """모든 MMSI에 대해 개별 모델 학습"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global MIN_SOG
    MIN_SOG = min_sog

    # --------------------
    # 데이터 준비
    # --------------------
    required = ["datetime", "mmsi", "lat", "lon", "sog"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df_all = df_all.copy()
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    for c in ["lat", "lon", "sog", "mmsi"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all = df_all.dropna(subset=["datetime", "lat", "lon", "sog", "mmsi"])

    # 속력 필터링
    df_all = df_all[df_all["sog"] >= min_sog].reset_index(drop=True)

    print(f"[V5] 전체 데이터: {len(df_all)} rows (속력 >= {min_sog} 노트)")

    # --------------------
    # 격자 정보 생성 (전체 범위 기준)
    # --------------------
    lat_min, lat_max = df_all["lat"].min(), df_all["lat"].max()
    lon_min, lon_max = df_all["lon"].min(), df_all["lon"].max()

    # 마진 추가
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05

    grid_info = create_grid_mapping(
        lat_min - lat_margin, lat_max + lat_margin,
        lon_min - lon_margin, lon_max + lon_margin,
        grid_size
    )

    print(f"[V5] 격자 정보: {grid_info['num_rows']}x{grid_info['num_cols']} = {grid_info['total_grids']} 격자")
    print(f"[V5] 격자 크기: {grid_size}도 (약 {grid_size * 111:.1f}km)")

    # 전역 격자 정보 저장
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "grid_info.npz"),
        grid_size=float(grid_size),
        lat_min=float(grid_info['lat_min']),
        lon_min=float(grid_info['lon_min']),
        lat_max=float(grid_info['lat_max']),
        lon_max=float(grid_info['lon_max']),
        num_rows=int(grid_info['num_rows']),
        num_cols=int(grid_info['num_cols']),
        total_grids=int(grid_info['total_grids']),
        min_sog=float(min_sog),
    )

    # --------------------
    # MMSI별 학습
    # --------------------
    mmsi_list = df_all["mmsi"].unique()
    print(f"[V5] 총 MMSI 수: {len(mmsi_list)}")

    results = {}
    success_count = 0
    fail_count = 0

    for i, mmsi in enumerate(mmsi_list):
        print(f"\n[{i+1}/{len(mmsi_list)}] MMSI: {mmsi}")

        df_mmsi = df_all[df_all["mmsi"] == mmsi].copy()

        model_path, config_path = train_mmsi_model(
            mmsi=mmsi,
            df_mmsi=df_mmsi,
            grid_info=grid_info,
            seq_len=seq_len,
            stride=stride,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=save_dir,
            device=device,
            patience=patience,
            warmup_epochs=warmup_epochs,
            val_ratio=val_ratio,
            seed=seed,
        )

        if model_path is not None:
            results[mmsi] = {
                'model_path': model_path,
                'config_path': config_path,
            }
            success_count += 1
        else:
            fail_count += 1

    # 결과 요약
    print(f"\n{'='*60}")
    print(f"[V5] 학습 완료!")
    print(f"  - 성공: {success_count} MMSI")
    print(f"  - 실패/스킵: {fail_count} MMSI")
    print(f"  - 저장 위치: {save_dir}")
    print(f"{'='*60}")

    # 결과 저장
    summary_path = os.path.join(save_dir, "training_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"V5 LLM-Style Grid Sequence Training Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total MMSI: {len(mmsi_list)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed/Skipped: {fail_count}\n")
        f.write(f"Grid Size: {grid_size} deg\n")
        f.write(f"Min SOG: {min_sog} knots\n")
        f.write(f"Seq Length: {seq_len}\n")
        f.write(f"\n")
        f.write(f"Trained MMSI:\n")
        for mmsi in results:
            f.write(f"  - {mmsi}\n")

    return results


if __name__ == "__main__":
    print("train_global_model_v5.py - LLM 스타일 격자 ID 시퀀스 학습")
    print("사용법: run_train_v5.py 참조")
