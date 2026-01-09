# -*- coding: utf-8 -*-
"""
V8 Direct Coordinate Transformer
=================================
- 격자 분류 대신 좌표 직접 예측 (Regression)
- 입력: (lat, lon, sog, cog) × 50 스텝
- 출력: (Δlat, Δlon) × 20 스텝 (상대 변위)
- 모델 크기 작음, 정밀도 높음
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import warnings
import math
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"
SAVE_DIR = "global_model_v8_direct/울산"
SEQ_LEN = 50  # 입력 시퀀스 길이
PRED_LEN = 20  # 예측 길이


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformer(nn.Module):
    """
    V8 Direct Coordinate Transformer
    - 입력: (lat, lon, sog, cog) 정규화된 시퀀스
    - 출력: (Δlat, Δlon) × pred_len
    """
    def __init__(self, input_dim=4, hidden_dim=128, num_heads=8,
                 num_layers=4, dropout=0.1, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # 입력 프로젝션
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 출력 레이어 (MLP)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2)  # (Δlat, Δlon) × pred_len
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, seq_len, 4) - [lat, lon, sog, cog] 정규화된 값
        return: (B, pred_len, 2) - [Δlat, Δlon]
        """
        B = x.size(0)

        # 입력 프로젝션 + Positional Encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Transformer Encoding
        encoded = self.transformer(x)

        # 마지막 토큰에서 전체 미래 예측
        out = self.output_mlp(encoded[:, -1, :])

        return out.view(B, self.pred_len, 2)


class TrajectoryDataset(Dataset):
    """전처리된 pkl 데이터를 사용하는 Dataset"""
    def __init__(self, sequences, stats):
        self.sequences = sequences
        self.stats = stats

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        inp = sample['input'].astype(np.float32)  # (50, 4): lat, lon, sog, cog
        target = sample['target'].astype(np.float32)  # (20, 2): lat, lon

        # 입력 정규화
        inp_norm = self._normalize_input(inp)

        # 타겟: 마지막 입력 위치 기준 상대 변위 (Δlat, Δlon)
        last_pos = inp[-1, :2]  # 마지막 위치
        delta_target = target - last_pos  # 상대 변위

        # 변위 스케일링 (대략 0.01도 = 1km 정도)
        delta_target = delta_target * 100  # 스케일 조정

        return (
            torch.from_numpy(inp_norm),
            torch.from_numpy(delta_target)
        )

    def _normalize_input(self, inp):
        """입력 데이터 정규화"""
        inp_norm = inp.copy()

        # lat, lon: 상대 좌표로 변환 (첫 번째 위치 기준)
        inp_norm[:, 0] = (inp[:, 0] - inp[0, 0]) * 100  # lat
        inp_norm[:, 1] = (inp[:, 1] - inp[0, 1]) * 100  # lon

        # sog: 0-30 knots 범위 정규화
        inp_norm[:, 2] = inp[:, 2] / 30.0

        # cog: sin/cos 변환 (각도 연속성)
        cog_rad = np.radians(inp[:, 3])
        inp_norm[:, 3] = np.sin(cog_rad)

        # cog cos를 5번째 feature로 추가하려면 모델 수정 필요
        # 일단 sin만 사용

        return inp_norm


class TrajectoryDatasetV2(Dataset):
    """
    개선된 Dataset - COG를 sin/cos로 분리
    입력: (lat_rel, lon_rel, sog_norm, cog_sin, cog_cos) × 50
    """
    def __init__(self, sequences, stats=None):
        self.sequences = sequences
        self.stats = stats

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        inp = sample['input'].astype(np.float32)  # (50, 4): lat, lon, sog, cog
        target = sample['target'].astype(np.float32)  # (20, 2): lat, lon

        # 입력 정규화 (5차원으로 확장)
        inp_norm = self._normalize_input(inp)

        # 타겟: 마지막 입력 위치 기준 상대 변위
        last_pos = inp[-1, :2]
        delta_target = (target - last_pos) * 100  # 스케일 조정

        return (
            torch.from_numpy(inp_norm),
            torch.from_numpy(delta_target)
        )

    def _normalize_input(self, inp):
        """입력 데이터 정규화 (5차원)"""
        seq_len = inp.shape[0]
        inp_norm = np.zeros((seq_len, 5), dtype=np.float32)

        # lat, lon: 상대 좌표 (첫 번째 위치 기준)
        inp_norm[:, 0] = (inp[:, 0] - inp[0, 0]) * 100
        inp_norm[:, 1] = (inp[:, 1] - inp[0, 1]) * 100

        # sog: 정규화 (0-30 knots)
        inp_norm[:, 2] = inp[:, 2] / 30.0

        # cog: sin/cos 변환
        cog_rad = np.radians(inp[:, 3])
        inp_norm[:, 3] = np.sin(cog_rad)
        inp_norm[:, 4] = np.cos(cog_rad)

        return inp_norm


def load_sequences(data_dir, max_chunks=None):
    """청크 파일에서 시퀀스 로드"""
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


def haversine(lat1, lon1, lat2, lon2):
    """두 점 사이 거리 (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def evaluate_model(model, val_loader, device):
    """모델 평가 - 실제 거리 오차 계산"""
    model.eval()
    all_errors = []

    with torch.no_grad():
        for inp, delta_target in val_loader:
            inp = inp.to(device)
            delta_target = delta_target.to(device)

            # 예측
            pred_delta = model(inp)  # (B, pred_len, 2)

            # 오차 계산 (스케일 복원)
            pred_delta_np = pred_delta.cpu().numpy() / 100  # 스케일 복원
            target_delta_np = delta_target.cpu().numpy() / 100

            for b in range(pred_delta_np.shape[0]):
                errors = []
                for t in range(pred_delta_np.shape[1]):
                    # 변위를 실제 거리로 변환 (근사)
                    dlat = pred_delta_np[b, t, 0] - target_delta_np[b, t, 0]
                    dlon = pred_delta_np[b, t, 1] - target_delta_np[b, t, 1]
                    # 간단한 거리 계산 (위도 1도 ≈ 111km)
                    dist_km = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)
                    errors.append(dist_km)
                all_errors.append(errors)

    all_errors = np.array(all_errors)
    return all_errors


def train_model(device, max_chunks=None, epochs=50, batch_size=256, lr=1e-3):
    """모델 학습"""
    print("=" * 60)
    print("V8 Direct Coordinate Transformer 학습")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"SEQ_LEN: {SEQ_LEN}, PRED_LEN: {PRED_LEN}")
    print(f"입력: (lat_rel, lon_rel, sog_norm, cog_sin, cog_cos) × {SEQ_LEN}")
    print(f"출력: (Δlat, Δlon) × {PRED_LEN}")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    sequences = load_sequences(DATA_DIR, max_chunks=max_chunks)

    # 2. Train/Val 분할
    print("\n[2] Train/Val 분할")
    np.random.seed(42)
    np.random.shuffle(sequences)
    val_size = int(len(sequences) * 0.1)
    train_sequences = sequences[:-val_size]
    val_sequences = sequences[-val_size:]

    print(f"Train: {len(train_sequences):,}, Val: {len(val_sequences):,}")

    # 3. Dataset 생성
    print("\n[3] Dataset 생성")
    train_dataset = TrajectoryDatasetV2(train_sequences)
    val_dataset = TrajectoryDatasetV2(val_sequences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 4. 모델 생성
    print("\n[4] 모델 생성")
    model = TrajectoryTransformer(
        input_dim=5,  # lat_rel, lon_rel, sog_norm, cog_sin, cog_cos
        hidden_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        pred_len=PRED_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {total_params:,}")

    # 옵티마이저 및 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.SmoothL1Loss()  # Huber Loss (outlier에 강건)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. 학습
    print("\n[5] 학습 시작")
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_val_loss = float('inf')
    best_val_error = float('inf')
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for inp, delta_target in pbar:
            inp = inp.to(device)
            delta_target = delta_target.to(device)

            optimizer.zero_grad()
            pred_delta = model(inp)

            loss = criterion(pred_delta, delta_target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * inp.size(0)
            train_count += inp.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= train_count

        # Validation
        model.eval()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for inp, delta_target in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                inp = inp.to(device)
                delta_target = delta_target.to(device)

                pred_delta = model(inp)
                loss = criterion(pred_delta, delta_target)

                val_loss += loss.item() * inp.size(0)
                val_count += inp.size(0)

        val_loss /= val_count

        # 실제 거리 오차 계산 (일부 샘플)
        if epoch % 5 == 0 or epoch == 1:
            errors = evaluate_model(model, val_loader, device)
            avg_error = np.mean(errors)
            error_5min = np.mean(errors[:, 4])
            error_10min = np.mean(errors[:, 9])
            error_20min = np.mean(errors[:, 19])
            print(f"  실제 오차 - 평균: {avg_error:.2f}km, 5분: {error_5min:.2f}km, "
                  f"10분: {error_10min:.2f}km, 20분: {error_20min:.2f}km")

        scheduler.step()

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 매 epoch 모델 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'model_epoch{epoch:03d}.pth'))

        # 설정 저장 (매 epoch)
        config = {
            'input_dim': 5,
            'hidden_dim': 128,
            'num_heads': 8,
            'num_layers': 4,
            'seq_len': SEQ_LEN,
            'pred_len': PRED_LEN,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        with open(os.path.join(SAVE_DIR, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model_best.pth'))
            print(f"  -> Best 모델 저장됨 (Val Loss: {best_val_loss:.6f})")
        else:
            patience_count += 1
            if patience_count >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    # 최종 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.pth')))
    errors = evaluate_model(model, val_loader, device)

    print(f"평균 오차: {np.mean(errors):.2f} km")
    print(f"5분 후: {np.mean(errors[:, 4]):.2f} km")
    print(f"10분 후: {np.mean(errors[:, 9]):.2f} km")
    print(f"15분 후: {np.mean(errors[:, 14]):.2f} km")
    print(f"20분 후: {np.mean(errors[:, 19]):.2f} km")

    print(f"\n저장 위치: {SAVE_DIR}/")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V8 Direct Coordinate Transformer Training')
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
