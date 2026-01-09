# -*- coding: utf-8 -*-
"""
V6 H3 Hexagonal Grid 기반 항적 예측 모델
=========================================
- H3 Resolution 9 (~700m 직경) 사용
- LLM 스타일: H3 셀 ID 시퀀스 → 다음 H3 셀 예측
- 속력 >= 8 노트 필터
- 빈도 높은 상위 K개 MMSI 학습
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h3
import warnings

warnings.filterwarnings("ignore")

# 설정
DATA_FOLDER = "G:/NIA_ai_project/항적데이터 추출/여수"
TRANSITION_FOLDER = "area_transition_results"
SAVE_DIR = "global_model_v6_h3"
H3_RESOLUTION = 9  # ~700m 직경
MIN_SOG = 8.0
SEQ_LEN = 50
TOP_K = 10  # 상위 10개 MMSI 학습
MAX_FILES_PER_MMSI = 500


class H3SequenceModel(nn.Module):
    """H3 셀 ID 시퀀스 → 다음 H3 셀 ID 예측 (Transformer)"""
    def __init__(self, num_cells, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_cells)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x_emb = self.cell_embed(x) + self.pos_embed(pos)
        x_emb = self.dropout(x_emb)
        out = self.transformer(x_emb)
        return self.fc(out[:, -1, :])


class H3Dataset(Dataset):
    def __init__(self, cell_indices, seq_len):
        self.cell_indices = np.array(cell_indices, dtype=np.int64)
        self.seq_len = seq_len
        self.indices = list(range(len(self.cell_indices) - seq_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        x = self.cell_indices[s:s + self.seq_len]
        y = self.cell_indices[s + self.seq_len]
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


def lat_lon_to_h3(lat, lon, resolution=H3_RESOLUTION):
    """위경도 → H3 셀 ID"""
    return h3.latlng_to_cell(lat, lon, resolution)


def h3_to_lat_lon(h3_cell):
    """H3 셀 ID → 중심 위경도"""
    return h3.cell_to_latlng(h3_cell)


def train_mmsi_model(mmsi, all_trans, device):
    """단일 MMSI 모델 학습"""
    print(f"\n{'='*60}")
    print(f"MMSI {mmsi} 학습 시작")
    print(f"{'='*60}")

    target_trans = all_trans[all_trans['mmsi'] == mmsi].head(MAX_FILES_PER_MMSI)

    # 데이터 로드
    all_points = []
    loaded = 0

    for _, row in target_trans.iterrows():
        s_area = row['start_area']
        e_area = row['end_area']
        start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

        filename = f"{mmsi}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
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

            all_points.extend(intp[['lat', 'lon', 'sog']].values.tolist())
            loaded += 1

        except Exception as e:
            continue

        if loaded >= MAX_FILES_PER_MMSI:
            break

    print(f"  로드된 파일: {loaded}")
    print(f"  총 포인트: {len(all_points)}")

    if len(all_points) < SEQ_LEN * 10:
        print(f"  [SKIP] 데이터 부족")
        return None

    # H3 셀 변환
    print(f"  H3 Resolution {H3_RESOLUTION} 변환 중...")
    h3_cells = []
    for lat, lon, sog in all_points:
        cell = lat_lon_to_h3(lat, lon)
        h3_cells.append(cell)

    # 고유 H3 셀 → 인덱스 매핑 (0부터 시작)
    unique_cells = list(set(h3_cells))
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    idx_to_cell = {idx: cell for idx, cell in enumerate(unique_cells)}

    num_cells = len(unique_cells)
    print(f"  고유 H3 셀 수: {num_cells}")

    # 인덱스 시퀀스 생성
    cell_indices = [cell_to_idx[cell] for cell in h3_cells]

    # Dataset & DataLoader
    dataset = H3Dataset(cell_indices, SEQ_LEN)
    n_val = max(1, int(len(dataset) * 0.2))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 모델 학습
    model = H3SequenceModel(num_cells=num_cells).to(device)
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

            # 저장
            save_path = os.path.join(SAVE_DIR, str(mmsi))
            os.makedirs(save_path, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

            # 설정 저장 (H3 셀 매핑 포함)
            np.savez(os.path.join(save_path, 'config.npz'),
                     mmsi=mmsi, seq_len=SEQ_LEN, h3_resolution=H3_RESOLUTION,
                     num_cells=num_cells, min_sog=MIN_SOG,
                     best_val_loss=best_val_loss, best_val_acc=best_val_acc)

            # H3 셀 매핑 저장 (pickle)
            import pickle
            with open(os.path.join(save_path, 'cell_mapping.pkl'), 'wb') as f:
                pickle.dump({'cell_to_idx': cell_to_idx, 'idx_to_cell': idx_to_cell}, f)
        else:
            patience_count += 1
            if patience_count >= 15:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\n  학습 완료!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

    return {'mmsi': mmsi, 'val_loss': best_val_loss, 'val_acc': best_val_acc}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print(f"V6 H3 Hexagonal Grid 기반 학습")
    print(f"H3 Resolution: {H3_RESOLUTION} (~700m)")
    print("=" * 60)

    # 1. 전이 정보 로드
    print("\n[1] 전이 정보 로드")
    transition_files = [f for f in os.listdir(TRANSITION_FOLDER) if f.endswith('.csv')]
    all_trans = pd.concat([pd.read_csv(os.path.join(TRANSITION_FOLDER, f)) for f in transition_files])
    print(f"  전이 정보: {len(all_trans)} 건")

    # 2. MMSI 빈도 분석 - 고속 선박 우선
    print(f"\n[2] 상위 {TOP_K}개 고속 MMSI 선택")

    # 고속 데이터가 많은 MMSI 찾기 (샘플링)
    mmsi_speed_counts = {}
    mmsi_counts = all_trans['mmsi'].value_counts()

    for mmsi in mmsi_counts.head(30).index:
        mmsi_trans = all_trans[all_trans['mmsi'] == mmsi].head(10)
        fast_count = 0

        for _, row in mmsi_trans.iterrows():
            s_area = row['start_area']
            e_area = row['end_area']
            start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
            end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

            filename = f"{mmsi}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
            filepath = os.path.join(DATA_FOLDER, filename)

            if not os.path.exists(filepath):
                continue

            try:
                df = pd.read_csv(filepath, encoding='cp949')
                df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
                fast_count += (df['sog'] >= MIN_SOG).sum()
            except:
                continue

        if fast_count > 0:
            mmsi_speed_counts[mmsi] = fast_count

    # 고속 데이터 많은 순으로 정렬
    sorted_mmsi = sorted(mmsi_speed_counts.items(), key=lambda x: x[1], reverse=True)
    top_mmsi_list = [m[0] for m in sorted_mmsi[:TOP_K]]

    print(f"  선택된 MMSI (8노트+ 데이터 기준):")
    for i, mmsi in enumerate(top_mmsi_list):
        print(f"    {i+1}. MMSI {mmsi}: 8노트+ {mmsi_speed_counts[mmsi]}개")

    # 3. MMSI별 학습
    print(f"\n[3] MMSI별 모델 학습")
    os.makedirs(SAVE_DIR, exist_ok=True)

    results = []
    for mmsi in top_mmsi_list:
        result = train_mmsi_model(mmsi, all_trans, device)
        if result:
            results.append(result)

    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("학습 완료 요약")
    print("=" * 60)
    for r in results:
        print(f"  MMSI {r['mmsi']}: Val Acc {r['val_acc']*100:.1f}%")
    print(f"\n저장 위치: {SAVE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
