"""
울산 항적 예측 모델 학습 스크립트
- 선종, 길이 메타 정보를 활용한 Seq2Seq LSTM 모델
- 청크 단위로 데이터 로드하여 메모리 효율적 학습
"""

import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random


# ============= 데이터셋 =============
class TrajectoryDataset(Dataset):
    """청크 파일에서 로드한 시퀀스 데이터셋"""
    def __init__(self, sequences, lat_mean, lat_std, lon_mean, lon_std):
        self.sequences = sequences
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.lon_mean = lon_mean
        self.lon_std = lon_std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 입력: [seq_len, 4] (lat, lon, sog, cog)
        input_seq = seq['input'].copy()
        # 정규화
        input_seq[:, 0] = (input_seq[:, 0] - self.lat_mean) / self.lat_std
        input_seq[:, 1] = (input_seq[:, 1] - self.lon_mean) / self.lon_std
        input_seq[:, 2] = input_seq[:, 2] / 30.0  # SOG 정규화 (0~30)
        input_seq[:, 3] = input_seq[:, 3] / 360.0  # COG 정규화 (0~360)

        # 타겟: [pred_len, 2] (lat, lon)
        target_seq = seq['target'].copy()
        target_seq[:, 0] = (target_seq[:, 0] - self.lat_mean) / self.lat_std
        target_seq[:, 1] = (target_seq[:, 1] - self.lon_mean) / self.lon_std

        # 메타: [2] (shiptype_cat, length_cat)
        meta = seq['meta']

        return {
            'input': torch.FloatTensor(input_seq),
            'target': torch.FloatTensor(target_seq),
            'meta': torch.LongTensor(meta)
        }


# ============= 모델 =============
class TrajectoryPredictor(nn.Module):
    """선종/길이 정보를 활용한 Seq2Seq LSTM 모델"""
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2,
                 pred_len=20, num_shiptype=7, num_length=16, embed_dim=16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # 메타 임베딩
        self.shiptype_embed = nn.Embedding(num_shiptype, embed_dim)
        self.length_embed = nn.Embedding(num_length, embed_dim)

        # 인코더 LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 메타 정보 융합
        self.meta_fc = nn.Linear(embed_dim * 2, hidden_dim)

        # 디코더 LSTM
        self.decoder = nn.LSTM(
            input_size=2,  # 이전 예측값 (lat, lon)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 출력 레이어
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # lat, lon
        )

    def forward(self, x, meta, target=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        # 메타 임베딩
        shiptype_emb = self.shiptype_embed(meta[:, 0])  # [B, embed_dim]
        length_emb = self.length_embed(meta[:, 1])      # [B, embed_dim]
        meta_emb = torch.cat([shiptype_emb, length_emb], dim=-1)  # [B, embed_dim*2]
        meta_hidden = self.meta_fc(meta_emb)  # [B, hidden_dim]

        # 인코더
        encoder_out, (hidden, cell) = self.encoder(x)

        # 메타 정보를 hidden state에 더함
        hidden = hidden + meta_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # 디코더 입력 (마지막 위치)
        decoder_input = x[:, -1, :2].unsqueeze(1)  # [B, 1, 2]

        outputs = []
        for t in range(self.pred_len):
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.output_fc(decoder_out.squeeze(1))  # [B, 2]
            outputs.append(prediction)

            # Teacher forcing
            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = prediction.unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)  # [B, pred_len, 2]
        return outputs


# ============= 학습 함수 =============
def load_chunk_files(chunk_dir, num_chunks=None):
    """청크 파일 로드"""
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, 'sequences_chunk_*.pkl')))
    if num_chunks:
        chunk_files = chunk_files[:num_chunks]
    return chunk_files


def compute_normalization_stats(chunk_files, sample_size=100000):
    """정규화 통계 계산"""
    lats, lons = [], []
    total_loaded = 0

    for cf in tqdm(chunk_files, desc="Computing stats"):
        with open(cf, 'rb') as f:
            sequences = pickle.load(f)

        for seq in sequences[:10000]:  # 각 청크에서 샘플링
            lats.extend(seq['input'][:, 0].tolist())
            lons.extend(seq['input'][:, 1].tolist())
            total_loaded += 1
            if total_loaded >= sample_size:
                break
        if total_loaded >= sample_size:
            break

    lat_mean, lat_std = np.mean(lats), np.std(lats)
    lon_mean, lon_std = np.mean(lons), np.std(lons)

    return lat_mean, lat_std, lon_mean, lon_std


def train_epoch(model, dataloader, optimizer, criterion, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_seq = batch['input'].to(device)
        target_seq = batch['target'].to(device)
        meta = batch['meta'].to(device)

        optimizer.zero_grad()

        outputs = model(input_seq, meta, target_seq, teacher_forcing_ratio=0.5)
        loss = criterion(outputs, target_seq)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            meta = batch['meta'].to(device)

            outputs = model(input_seq, meta, teacher_forcing_ratio=0.0)
            loss = criterion(outputs, target_seq)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # 설정
    data_dir = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"
    save_dir = r"k:\coding_project\NIA_선박항적예측프로그램\models\ulsan"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 하이퍼파라미터
    batch_size = 256
    epochs = 30
    learning_rate = 0.001
    hidden_dim = 128
    num_layers = 2

    # 청크 파일 로드
    chunk_files = load_chunk_files(data_dir)
    print(f"Found {len(chunk_files)} chunk files")

    if len(chunk_files) == 0:
        print("No chunk files found. Please run preprocessing first.")
        return

    # 정규화 통계
    print("Computing normalization stats...")
    lat_mean, lat_std, lon_mean, lon_std = compute_normalization_stats(chunk_files)
    print(f"Lat: mean={lat_mean:.4f}, std={lat_std:.4f}")
    print(f"Lon: mean={lon_mean:.4f}, std={lon_std:.4f}")

    # 정규화 통계 저장
    norm_stats = {
        'lat_mean': lat_mean, 'lat_std': lat_std,
        'lon_mean': lon_mean, 'lon_std': lon_std
    }
    with open(os.path.join(save_dir, 'norm_stats.pkl'), 'wb') as f:
        pickle.dump(norm_stats, f)

    # 모델 초기화
    model = TrajectoryPredictor(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pred_len=20,
        num_shiptype=7,
        num_length=16
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 학습 (청크 단위로)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        # 청크 순서 셔플
        random.shuffle(chunk_files)

        epoch_train_loss = 0
        epoch_val_loss = 0
        num_train_chunks = 0
        num_val_chunks = 0

        for i, chunk_file in enumerate(chunk_files):
            print(f"\nProcessing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")

            # 청크 로드
            with open(chunk_file, 'rb') as f:
                sequences = pickle.load(f)

            # 학습/검증 분리 (9:1)
            random.shuffle(sequences)
            split_idx = int(len(sequences) * 0.9)
            train_seqs = sequences[:split_idx]
            val_seqs = sequences[split_idx:]

            # 데이터셋 생성
            train_dataset = TrajectoryDataset(train_seqs, lat_mean, lat_std, lon_mean, lon_std)
            val_dataset = TrajectoryDataset(val_seqs, lat_mean, lat_std, lon_mean, lon_std)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            # 학습
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)

            epoch_train_loss += train_loss
            epoch_val_loss += val_loss
            num_train_chunks += 1
            num_val_chunks += 1

            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # 메모리 정리
            del sequences, train_seqs, val_seqs, train_dataset, val_dataset
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 에포크 평균 손실
        avg_train_loss = epoch_train_loss / num_train_chunks
        avg_val_loss = epoch_val_loss / num_val_chunks

        print(f"\nEpoch {epoch+1} Average - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")

        # 학습률 조정
        scheduler.step(avg_val_loss)

        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'norm_stats': norm_stats
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Best model saved!")

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'norm_stats': norm_stats
        }, os.path.join(save_dir, 'checkpoint.pth'))

    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    main()
