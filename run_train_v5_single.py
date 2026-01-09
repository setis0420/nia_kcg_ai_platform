# -*- coding: utf-8 -*-
"""
V5 초고속 테스트: 1개 MMSI만 학습 (최대 500파일)
================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# 설정
DATA_FOLDER = "G:/NIA_ai_project/항적데이터 추출/여수"
TRANSITION_FOLDER = "area_transition_results"
SAVE_DIR = "global_model_v5"
GRID_SIZE = 0.01
MIN_SOG = 8.0
SEQ_LEN = 50
MAX_FILES = 300  # 최대 파일 수 제한


class GridSequenceModel(nn.Module):
    """격자 ID 시퀀스 → 다음 격자 ID 예측"""
    def __init__(self, num_grids, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.grid_embed = nn.Embedding(num_grids + 1, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_grids)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x_emb = self.grid_embed(x) + self.pos_embed(pos)
        x_emb = self.dropout(x_emb)
        out = self.transformer(x_emb)
        return self.fc(out[:, -1, :])


class GridDataset(Dataset):
    def __init__(self, grid_ids, seq_len):
        self.grid_ids = np.array(grid_ids, dtype=np.int64)
        self.seq_len = seq_len
        self.indices = list(range(len(self.grid_ids) - seq_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        x = self.grid_ids[s:s + self.seq_len]
        y = self.grid_ids[s + self.seq_len]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def interpolate_1min(df):
    """1분 간격 보간"""
    if len(df) < 2:
        return None
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates('datetime')

    start_time = df['datetime'].iloc[0]
    end_time = df['datetime'].iloc[-1]
    time_range = pd.date_range(start=start_time, end=end_time, freq='1min')

    if len(time_range) < 2:
        return None

    df['_t'] = (df['datetime'] - start_time).dt.total_seconds()
    target_t = (time_range - start_time).total_seconds().values

    return pd.DataFrame({
        'datetime': time_range,
        'lat': np.interp(target_t, df['_t'].values, df['lat'].values),
        'lon': np.interp(target_t, df['_t'].values, df['lon'].values),
        'sog': np.interp(target_t, df['_t'].values, df['sog'].values),
    })


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("V5 초고속 테스트: 1개 MMSI")
    print("=" * 60)

    # 1. MMSI 선택 (고속 선박 + 데이터 풍부)
    print("\n[1] MMSI 선택")
    # 8노트+ 데이터가 6498개로 가장 많은 선박
    target_mmsi = 440315060
    print(f"  대상 MMSI: {target_mmsi} (8노트+ 6498개, max 9.9노트)")

    transition_files = [f for f in os.listdir(TRANSITION_FOLDER) if f.endswith('.csv')]
    all_trans = pd.concat([pd.read_csv(os.path.join(TRANSITION_FOLDER, f)) for f in transition_files])

    # 2. 해당 MMSI 파일만 로드 (최대 MAX_FILES개)
    print(f"\n[2] 데이터 로드 (최대 {MAX_FILES}개 파일)")
    target_trans = all_trans[all_trans['mmsi'] == target_mmsi].head(MAX_FILES)

    all_grids = []
    lat_all, lon_all = [], []
    loaded = 0

    for i, row in target_trans.iterrows():
        s_area = row['start_area']
        e_area = row['end_area']
        start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

        filename = f"{target_mmsi}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, encoding='cp949')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            for c in ['lat', 'lon', 'sog']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['datetime', 'lat', 'lon', 'sog'])

            # 속력 필터
            df = df[df['sog'] >= MIN_SOG]
            if len(df) < SEQ_LEN + 1:
                continue

            # 보간
            intp = interpolate_1min(df)
            if intp is None or len(intp) < SEQ_LEN + 1:
                continue

            intp = intp[intp['sog'] >= MIN_SOG]
            if len(intp) < SEQ_LEN + 1:
                continue

            lat_all.extend(intp['lat'].tolist())
            lon_all.extend(intp['lon'].tolist())
            loaded += 1

        except Exception as e:
            continue

        if loaded >= MAX_FILES:
            break

    print(f"  로드된 파일: {loaded}")
    print(f"  총 포인트: {len(lat_all)}")

    if len(lat_all) < SEQ_LEN * 10:
        print("[ERROR] 데이터 부족")
        return

    # 3. 격자 정보 생성
    print("\n[3] 격자 생성")
    lat_min, lat_max = min(lat_all) - 0.01, max(lat_all) + 0.01
    lon_min, lon_max = min(lon_all) - 0.01, max(lon_all) + 0.01

    num_rows = int(np.ceil((lat_max - lat_min) / GRID_SIZE)) + 1
    num_cols = int(np.ceil((lon_max - lon_min) / GRID_SIZE)) + 1
    total_grids = num_rows * num_cols

    print(f"  범위: lat [{lat_min:.3f}, {lat_max:.3f}], lon [{lon_min:.3f}, {lon_max:.3f}]")
    print(f"  격자: {num_rows} x {num_cols} = {total_grids}")

    # 4. 격자 ID 시퀀스 생성
    print("\n[4] 격자 ID 시퀀스 생성")
    grid_ids = []
    for lat, lon in zip(lat_all, lon_all):
        row = int((lat - lat_min) / GRID_SIZE)
        col = int((lon - lon_min) / GRID_SIZE)
        row = max(0, min(row, num_rows - 1))
        col = max(0, min(col, num_cols - 1))
        gid = row * num_cols + col
        grid_ids.append(gid)

    unique_grids = len(set(grid_ids))
    print(f"  시퀀스 길이: {len(grid_ids)}")
    print(f"  고유 격자 수: {unique_grids}")

    # 5. Dataset & DataLoader
    print("\n[5] 데이터셋 생성")
    dataset = GridDataset(grid_ids, SEQ_LEN)
    n_val = max(1, int(len(dataset) * 0.2))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 6. 모델 학습
    print("\n[6] 모델 학습")
    model = GridSequenceModel(num_grids=total_grids).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_count = 0

    for epoch in range(1, 101):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += x.size(0)

        # Val
        model.eval()
        va_loss, va_correct, va_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item() * x.size(0)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += x.size(0)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total
        va_loss /= va_total
        va_acc = va_correct / va_total

        scheduler.step(va_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.3f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            patience_count = 0
            # Save
            os.makedirs(os.path.join(SAVE_DIR, str(target_mmsi)), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, str(target_mmsi), 'model.pth'))
            np.savez(os.path.join(SAVE_DIR, str(target_mmsi), 'config.npz'),
                     mmsi=target_mmsi, seq_len=SEQ_LEN, grid_size=GRID_SIZE,
                     lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                     num_rows=num_rows, num_cols=num_cols, total_grids=total_grids,
                     min_sog=MIN_SOG, best_val_loss=best_val_loss, best_val_acc=best_val_acc)
        else:
            patience_count += 1
            if patience_count >= 15:
                print(f"  Early stop at epoch {epoch}")
                break

    print("\n" + "=" * 60)
    print(f"학습 완료!")
    print(f"  MMSI: {target_mmsi}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    print(f"  저장: {SAVE_DIR}/{target_mmsi}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
