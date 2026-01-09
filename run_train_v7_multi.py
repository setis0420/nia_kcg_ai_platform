# -*- coding: utf-8 -*-
"""
V7 멀티 임베딩 H3 모델: H3 셀 + SOG 구간 + COG 구간
=====================================================
- 전처리된 pkl 파일에서 데이터 로드
- H3 Resolution 9 (~700m 직경)
- SOG: 1노트 단위 구간화
- COG: 15도 단위 구간화
- 각 임베딩을 합산하여 Transformer 입력
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h3
import pickle
import glob
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"
SAVE_DIR = "global_model_v7_multi/울산"
H3_RESOLUTION = 9
SEQ_LEN = 50  # 입력 시퀀스 길이
PRED_LEN = 20  # 예측 길이

# SOG 구간: 0-1, 1-2, ..., 20+ = 21개 (0~20)
NUM_SOG_BINS = 21
# COG 구간: 0-15, 15-30, ..., 345-360 = 24개 (0~23)
NUM_COG_BINS = 24


def sog_to_bin(sog):
    """SOG를 구간 인덱스로 변환"""
    if sog < 0:
        return 0
    bin_idx = int(sog)
    return min(bin_idx, NUM_SOG_BINS - 1)


def cog_to_bin(cog):
    """COG를 15도 단위 구간 인덱스로 변환"""
    cog = cog % 360  # 0-360 범위로 정규화
    return int(cog / 15) % NUM_COG_BINS


class H3MultiEmbedModel(nn.Module):
    """H3 + SOG + COG 멀티 임베딩 Transformer 모델"""
    def __init__(self, num_cells, num_sog_bins=NUM_SOG_BINS, num_cog_bins=NUM_COG_BINS,
                 embed_dim=128, num_heads=8, num_layers=4, dropout=0.1, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.num_cells = num_cells

        self.cell_embed = nn.Embedding(num_cells, embed_dim)
        self.sog_embed = nn.Embedding(num_sog_bins, embed_dim)
        self.cog_embed = nn.Embedding(num_cog_bins, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 다중 스텝 예측을 위한 출력 레이어
        self.fc = nn.Linear(embed_dim, num_cells * pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_ids, sog_bins, cog_bins):
        B, T = cell_ids.shape
        pos = torch.arange(T, device=cell_ids.device).unsqueeze(0).expand(B, T)

        # 멀티 임베딩 합산
        x = self.cell_embed(cell_ids) + self.sog_embed(sog_bins) + self.cog_embed(cog_bins) + self.pos_embed(pos)
        x = self.dropout(x)
        out = self.transformer(x)

        # (B, num_cells * pred_len) -> (B, pred_len, num_cells)
        logits = self.fc(out[:, -1, :])
        return logits.view(B, self.pred_len, self.num_cells)


class H3MultiEmbedDataset(Dataset):
    """전처리된 pkl 데이터를 H3 셀로 변환하여 사용하는 Dataset"""
    def __init__(self, sequences, cell_to_idx, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        self.sequences = sequences
        self.cell_to_idx = cell_to_idx
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_cells = len(cell_to_idx)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        inp = sample['input']  # (50, 4): lat, lon, sog, cog
        target = sample['target']  # (20, 2): lat, lon

        # 입력 시퀀스를 H3 셀, SOG bin, COG bin으로 변환
        cell_indices = []
        sog_bins = []
        cog_bins = []

        for i in range(self.seq_len):
            lat, lon, sog, cog = inp[i]
            cell = h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION)
            cell_idx = self.cell_to_idx.get(cell, 0)  # unknown cell은 0
            cell_indices.append(cell_idx)
            sog_bins.append(sog_to_bin(sog))
            cog_bins.append(cog_to_bin(cog))

        # 타겟 시퀀스를 H3 셀로 변환
        target_indices = []
        for i in range(self.pred_len):
            lat, lon = target[i]
            cell = h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION)
            cell_idx = self.cell_to_idx.get(cell, 0)
            target_indices.append(cell_idx)

        return (
            torch.tensor(cell_indices, dtype=torch.long),
            torch.tensor(sog_bins, dtype=torch.long),
            torch.tensor(cog_bins, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long)
        )


def load_all_sequences(data_dir, max_chunks=None):
    """모든 청크 파일에서 시퀀스 로드"""
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "sequences_chunk_*.pkl")))

    if max_chunks:
        chunk_files = chunk_files[:max_chunks]

    print(f"로드할 청크 파일: {len(chunk_files)}개")

    all_sequences = []
    for chunk_file in tqdm(chunk_files, desc="청크 로드"):
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
            all_sequences.extend(chunk)

    print(f"총 시퀀스: {len(all_sequences):,}개")
    return all_sequences


def build_h3_vocabulary(sequences, sample_ratio=0.1):
    """시퀀스에서 H3 셀 어휘 구축"""
    print(f"H3 셀 어휘 구축 중... (샘플링 비율: {sample_ratio*100:.0f}%)")

    h3_cells = set()
    sample_size = int(len(sequences) * sample_ratio)
    sample_indices = np.random.choice(len(sequences), sample_size, replace=False)

    for idx in tqdm(sample_indices, desc="H3 셀 수집"):
        sample = sequences[idx]
        inp = sample['input']
        target = sample['target']

        # 입력 시퀀스
        for i in range(inp.shape[0]):
            lat, lon = inp[i, 0], inp[i, 1]
            cell = h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION)
            h3_cells.add(cell)

        # 타겟 시퀀스
        for i in range(target.shape[0]):
            lat, lon = target[i, 0], target[i, 1]
            cell = h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUTION)
            h3_cells.add(cell)

    # 인덱스 매핑 생성 (0은 unknown용으로 예약)
    cell_to_idx = {cell: idx + 1 for idx, cell in enumerate(sorted(h3_cells))}
    idx_to_cell = {idx + 1: cell for idx, cell in enumerate(sorted(h3_cells))}
    cell_to_idx['<UNK>'] = 0
    idx_to_cell[0] = '<UNK>'

    print(f"고유 H3 셀 수: {len(h3_cells):,}개")
    return cell_to_idx, idx_to_cell


def train_model(device, max_chunks=None, epochs=50, batch_size=256, lr=1e-3):
    """모델 학습"""
    print("=" * 60)
    print("V7 멀티 임베딩 모델 학습 (전처리 데이터 사용)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"H3 Resolution: {H3_RESOLUTION}")
    print(f"SEQ_LEN: {SEQ_LEN}, PRED_LEN: {PRED_LEN}")
    print(f"SOG 구간: {NUM_SOG_BINS}개, COG 구간: {NUM_COG_BINS}개")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    sequences = load_all_sequences(DATA_DIR, max_chunks=max_chunks)

    # 2. H3 어휘 구축
    print("\n[2] H3 어휘 구축")
    cell_to_idx, idx_to_cell = build_h3_vocabulary(sequences, sample_ratio=0.1)
    num_cells = len(cell_to_idx)
    print(f"총 H3 셀 수 (어휘 크기): {num_cells:,}")

    # 3. Dataset 생성
    print("\n[3] Dataset 생성")

    # Train/Val 분할
    np.random.shuffle(sequences)
    val_size = int(len(sequences) * 0.1)
    train_sequences = sequences[:-val_size]
    val_sequences = sequences[-val_size:]

    print(f"Train: {len(train_sequences):,}, Val: {len(val_sequences):,}")

    train_dataset = H3MultiEmbedDataset(train_sequences, cell_to_idx)
    val_dataset = H3MultiEmbedDataset(val_sequences, cell_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 4. 모델 생성
    print("\n[4] 모델 생성")
    model = H3MultiEmbedModel(num_cells=num_cells, pred_len=PRED_LEN).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. 학습
    print("\n[5] 학습 시작")
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for cell_x, sog_x, cog_x, cell_y in pbar:
            cell_x = cell_x.to(device)
            sog_x = sog_x.to(device)
            cog_x = cog_x.to(device)
            cell_y = cell_y.to(device)  # (B, pred_len)

            optimizer.zero_grad()
            logits = model(cell_x, sog_x, cog_x)  # (B, pred_len, num_cells)

            # 각 예측 스텝에 대한 손실 계산
            loss = 0
            for t in range(PRED_LEN):
                loss += criterion(logits[:, t, :], cell_y[:, t])
            loss /= PRED_LEN

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * cell_x.size(0)
            # 첫 번째 예측 스텝의 정확도만 계산
            train_correct += (logits[:, 0, :].argmax(1) == cell_y[:, 0]).sum().item()
            train_total += cell_x.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for cell_x, sog_x, cog_x, cell_y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                cell_x = cell_x.to(device)
                sog_x = sog_x.to(device)
                cog_x = cog_x.to(device)
                cell_y = cell_y.to(device)

                logits = model(cell_x, sog_x, cog_x)

                loss = 0
                for t in range(PRED_LEN):
                    loss += criterion(logits[:, t, :], cell_y[:, t])
                loss /= PRED_LEN

                val_loss += loss.item() * cell_x.size(0)
                val_correct += (logits[:, 0, :].argmax(1) == cell_y[:, 0]).sum().item()
                val_total += cell_x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_count = 0

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model.pth'))

            # 설정 저장
            config = {
                'num_cells': num_cells,
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'h3_resolution': H3_RESOLUTION,
                'num_sog_bins': NUM_SOG_BINS,
                'num_cog_bins': NUM_COG_BINS,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
            }
            with open(os.path.join(SAVE_DIR, 'config.pkl'), 'wb') as f:
                pickle.dump(config, f)

            # 셀 매핑 저장
            with open(os.path.join(SAVE_DIR, 'cell_mapping.pkl'), 'wb') as f:
                pickle.dump({'cell_to_idx': cell_to_idx, 'idx_to_cell': idx_to_cell}, f)

            print(f"  -> Best 모델 저장됨 (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.3f})")
        else:
            patience_count += 1
            if patience_count >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    print(f"저장 위치: {SAVE_DIR}/")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V7 Multi-Embedding Model Training')
    parser.add_argument('--max_chunks', type=int, default=None, help='최대 청크 수 (None=전체)')
    parser.add_argument('--epochs', type=int, default=50, help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=256, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model(
        device=device,
        max_chunks=args.max_chunks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
