# -*- coding: utf-8 -*-
"""
전처리된 데이터로 모델 학습
============================
prepare_data.py로 생성된 npz 파일을 로드하여 학습

사용법:
    python train_model.py --data_dir "prepared_data" \
                          --epochs 300 \
                          --device cuda
"""

import os
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 기본 하이퍼파라미터
EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3
SAVE_DIR = "global_model_v2"


class TrajectoryDatasetFromNpz(Dataset):
    """
    전처리된 npz 데이터를 로드하여 사용하는 Dataset (CPU용)
    """
    def __init__(self, Xn_num, X_cat, Yn, indices, seq_len):
        self.Xn_num = Xn_num
        self.X_cat = X_cat
        self.Yn = Yn
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.seq_len

        x_num = self.Xn_num[s:e]
        x_cat = self.X_cat[s:e]
        y = self.Yn[e]
        x_last = self.Xn_num[e-1]

        return (torch.from_numpy(x_num),
                torch.from_numpy(x_cat),
                torch.from_numpy(y),
                torch.from_numpy(x_last))


class TrajectoryDatasetGPU(Dataset):
    """
    GPU에 미리 올려둔 텐서를 사용하는 Dataset (빠른 학습용)
    시퀀스를 미리 구성하여 GPU에서 바로 사용
    """
    def __init__(self, Xn_num_t, X_cat_t, Yn_t, indices, seq_len, device):
        # 미리 모든 시퀀스를 구성 (GPU에서 직접 수행)
        n_samples = len(indices)
        indices_t = torch.tensor(indices, dtype=torch.long, device=device)

        # 시퀀스 인덱스 생성: [n_samples, seq_len]
        seq_offsets = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        all_indices = indices_t.unsqueeze(1) + seq_offsets  # [n_samples, seq_len]

        # 미리 시퀀스 데이터 구성
        self.X_num = Xn_num_t[all_indices]  # [n_samples, seq_len, num_features]
        self.X_cat = X_cat_t[all_indices]   # [n_samples, seq_len, cat_features]
        self.Y = Yn_t[indices_t + seq_len]  # [n_samples, output_dim]
        self.X_last = Xn_num_t[indices_t + seq_len - 1]  # [n_samples, num_features]

        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.Y[idx], self.X_last[idx]


class ChunkedTrainer:
    """
    큰 데이터셋을 청크 단위로 GPU에 올려서 학습
    메모리 효율적 + GPU compute 활용
    """
    def __init__(self, Xn_num, X_cat, Yn, segment_starts, seq_len,
                 train_seg_ids, val_seg_ids, device, chunk_size=50):
        """
        chunk_size: 한 번에 GPU에 올릴 segment 개수
        """
        self.Xn_num = Xn_num  # numpy array (CPU)
        self.X_cat = X_cat
        self.Yn = Yn
        self.segment_starts = segment_starts
        self.seq_len = seq_len
        self.train_seg_ids = train_seg_ids
        self.val_seg_ids = val_seg_ids
        self.device = device
        self.chunk_size = chunk_size

    def _get_segment_data_range(self, seg_ids):
        """segment들의 데이터 범위 계산"""
        if len(seg_ids) == 0:
            return None, None, []

        all_indices = []
        for sid in seg_ids:
            all_indices.extend(self.segment_starts[sid])

        if len(all_indices) == 0:
            return None, None, []

        # 필요한 데이터 범위 계산
        min_idx = min(all_indices)
        max_idx = max(all_indices) + self.seq_len + 1

        return min_idx, max_idx, all_indices

    def _load_chunk_to_gpu(self, seg_ids):
        """특정 segment들의 데이터를 GPU에 로드"""
        min_idx, max_idx, indices = self._get_segment_data_range(seg_ids)

        if min_idx is None:
            return None, None, None, []

        # 해당 범위만 GPU에 올림 (float32로 변환하여 GPU 효율성 향상)
        Xn_chunk = torch.from_numpy(self.Xn_num[min_idx:max_idx].astype(np.float32)).to(self.device)
        Xcat_chunk = torch.from_numpy(self.X_cat[min_idx:max_idx]).to(self.device)  # int는 그대로
        Yn_chunk = torch.from_numpy(self.Yn[min_idx:max_idx].astype(np.float32)).to(self.device)

        # 인덱스 조정 (청크 내 상대 인덱스로 변환)
        adjusted_indices = [i - min_idx for i in indices]

        return Xn_chunk, Xcat_chunk, Yn_chunk, adjusted_indices

    def get_train_chunks(self):
        """학습용 청크 생성기"""
        # 매 epoch마다 segment 순서 섞기
        shuffled_ids = self.train_seg_ids.copy()
        np.random.shuffle(shuffled_ids)

        for i in range(0, len(shuffled_ids), self.chunk_size):
            chunk_seg_ids = shuffled_ids[i:i + self.chunk_size]
            yield self._load_chunk_to_gpu(chunk_seg_ids)

    def get_val_chunks(self):
        """검증용 청크 생성기"""
        for i in range(0, len(self.val_seg_ids), self.chunk_size):
            chunk_seg_ids = self.val_seg_ids[i:i + self.chunk_size]
            yield self._load_chunk_to_gpu(chunk_seg_ids)

    @property
    def n_train_chunks(self):
        return (len(self.train_seg_ids) + self.chunk_size - 1) // self.chunk_size

    @property
    def n_val_chunks(self):
        return (len(self.val_seg_ids) + self.chunk_size - 1) // self.chunk_size

    @property
    def n_train_sequences(self):
        total = 0
        for sid in self.train_seg_ids:
            total += len(self.segment_starts[sid])
        return total

    @property
    def n_val_sequences(self):
        total = 0
        for sid in self.val_seg_ids:
            total += len(self.segment_starts[sid])
        return total


class LSTMTrajectoryModelV2(nn.Module):
    """LSTM 모델 (Embedding + LSTM)"""
    def __init__(self,
                 num_features=5,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=5,
                 dropout=0.2,
                 num_mmsi=1000,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16):
        super().__init__()

        self.num_features = num_features
        self.embed_dim = embed_dim

        self.mmsi_embed = nn.Embedding(num_mmsi, embed_dim)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        lstm_input_dim = num_features + 4 * embed_dim

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        mmsi_emb = self.mmsi_embed(x_cat[:, :, 0])
        start_emb = self.start_area_embed(x_cat[:, :, 1])
        end_emb = self.end_area_embed(x_cat[:, :, 2])
        grid_emb = self.grid_embed(x_cat[:, :, 3])

        x = torch.cat([x_num, mmsi_emb, start_emb, end_emb, grid_emb], dim=-1)
        out, _ = self.lstm(x)

        return self.fc(out[:, -1, :])


def loss_with_smoothness(y_pred, y_true, x_last,
                         w_mse=(2, 2, 1, 3, 3),
                         smooth_lambda=0.05,
                         sog_lambda=0.10,
                         heading_lambda=0.02,
                         turn_boost=2.0):
    """Smoothness 정규화 포함 Loss"""
    w = torch.tensor(w_mse, device=y_pred.device, dtype=y_pred.dtype).view(1, -1)

    true_heading_change = ((y_true[:, 3:5] - x_last[:, 3:5]) ** 2).sum(dim=1).sqrt()
    turn_weight = 1.0 + turn_boost * true_heading_change
    turn_weight = turn_weight.unsqueeze(1)

    mse = ((y_pred - y_true) ** 2 * w * turn_weight).mean()

    dsog = (y_pred[:, 2] - x_last[:, 2]).abs().mean()
    dheading = (y_pred[:, 3:5] - x_last[:, 3:5]).abs().mean()

    smooth = sog_lambda * dsog + heading_lambda * dheading
    return mse + smooth_lambda * smooth


def load_prepared_data(data_dir):
    """전처리된 데이터 로드"""
    print(f"[INFO] 데이터 로드: {data_dir}")

    # 학습 데이터
    data_path = os.path.join(data_dir, "training_data.npz")
    data = np.load(data_path)
    Xn_num = data['Xn_num']
    X_cat = data['X_cat']
    Yn = data['Yn']
    segment_bounds = data['segment_bounds']
    all_indices = data['all_indices']

    print(f"  - Xn_num: {Xn_num.shape}")
    print(f"  - X_cat: {X_cat.shape}")
    print(f"  - Yn: {Yn.shape}")
    print(f"  - 시퀀스 수: {len(all_indices)}")

    # 메타 정보
    meta_path = os.path.join(data_dir, "meta.npz")
    meta = dict(np.load(meta_path, allow_pickle=True))

    # segment_starts
    seg_starts_path = os.path.join(data_dir, "segment_starts.npy")
    segment_starts = np.load(seg_starts_path, allow_pickle=True)

    return {
        'Xn_num': Xn_num,
        'X_cat': X_cat,
        'Yn': Yn,
        'segment_bounds': segment_bounds,
        'segment_starts': segment_starts,
        'all_indices': all_indices,
        'meta': meta,
    }


def train_model(
    data_dir,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    save_dir=SAVE_DIR,
    device=None,
    patience=30,
    min_delta=1e-5,
    warmup_epochs=20,
    use_scheduler=True,
    lr_patience=8,
    lr_factor=0.5,
    min_lr=1e-6,
    smooth_lambda=0.05,
    sog_lambda=0.10,
    heading_lambda=0.02,
    turn_boost=2.0,
    embed_dim=16,
    grad_clip_norm=1.0,
    val_ratio=0.2,
    seed=42,
    chunk_size=100,  # 한 번에 GPU에 올릴 segment 수
):
    """모델 학습"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로드
    data = load_prepared_data(data_dir)
    Xn_num = data['Xn_num']
    X_cat = data['X_cat']
    Yn = data['Yn']
    segment_starts = data['segment_starts']
    meta = data['meta']

    seq_len = int(meta['seq_len'])
    n_segments = len(segment_starts)

    print(f"\n[INFO] 학습 설정:")
    print(f"  - seq_len: {seq_len}")
    print(f"  - segments: {n_segments}")
    print(f"  - device: {device}")

    # Train/Val 분리 (segment 단위)
    rng = np.random.default_rng(seed)
    seg_ids = np.arange(n_segments)
    rng.shuffle(seg_ids)

    n_val_seg = max(1, int(n_segments * val_ratio))
    val_seg_ids = seg_ids[:n_val_seg].tolist()
    train_seg_ids = seg_ids[n_val_seg:].tolist()

    # 인덱스 생성
    train_indices = []
    for sid in train_seg_ids:
        train_indices.extend(segment_starts[sid])
    train_indices = np.array(train_indices)

    val_indices = []
    for sid in val_seg_ids:
        val_indices.extend(segment_starts[sid])
    val_indices = np.array(val_indices)

    # ChunkedTrainer 생성 (데이터를 청크 단위로 GPU에 올림)
    chunked_trainer = ChunkedTrainer(
        Xn_num, X_cat, Yn, segment_starts, seq_len,
        train_seg_ids, val_seg_ids, device, chunk_size=chunk_size
    )

    print(f"  - train segments: {len(train_seg_ids)}, sequences: {chunked_trainer.n_train_sequences}")
    print(f"  - val segments: {len(val_seg_ids)}, sequences: {chunked_trainer.n_val_sequences}")
    print(f"  - chunk_size: {chunk_size} segments/chunk")
    print(f"  - train chunks: {chunked_trainer.n_train_chunks}, val chunks: {chunked_trainer.n_val_chunks}")

    # 모델 생성
    model = LSTMTrajectoryModelV2(
        num_features=5,
        hidden_dim=128,
        num_layers=2,
        output_dim=5,
        dropout=0.2,
        num_mmsi=int(meta['num_mmsi']),
        num_start_area=int(meta['num_start_area']),
        num_end_area=int(meta['num_end_area']),
        num_grids=int(meta['total_grids']) + 1,
        embed_dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr, verbose=True
        )

    # 저장 경로
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "lstm_global_v2.pth")
    scaler_path = os.path.join(save_dir, "scaler_global_v2.npz")

    # 학습 루프
    best_val = float("inf")
    best_epoch = -1
    bad_count = 0

    print(f"\n[INFO] 학습 시작 | epochs={epochs}, warmup={warmup_epochs}, patience={patience}")

    n_train_chunks = chunked_trainer.n_train_chunks
    n_val_chunks = chunked_trainer.n_val_chunks

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss_sum = 0.0
        tr_count = 0

        for chunk_idx, (Xn_chunk, Xcat_chunk, Yn_chunk, indices) in enumerate(chunked_trainer.get_train_chunks()):
            if Xn_chunk is None:
                continue

            # 청크 내 데이터를 GPU에서 직접 시퀀스로 구성
            chunk_ds = TrajectoryDatasetGPU(Xn_chunk, Xcat_chunk, Yn_chunk, indices, seq_len, device)
            n_samples = len(chunk_ds)

            # 배치 인덱스 셔플 (GPU에서)
            perm = torch.randperm(n_samples, device=device)

            # 청크 내 loss 누적 (GPU에서)
            chunk_loss = torch.tensor(0.0, device=device)
            chunk_count = 0

            # DataLoader 없이 직접 배치 처리 (GPU 텐서 그대로 사용)
            for start in range(0, n_samples, batch_size):
                batch_idx = perm[start:start + batch_size]
                x_num = chunk_ds.X_num[batch_idx]
                x_cat = chunk_ds.X_cat[batch_idx]
                y = chunk_ds.Y[batch_idx]
                x_last = chunk_ds.X_last[batch_idx]

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

                # GPU에서 누적 (동기화 없음)
                chunk_loss += loss.detach() * x_num.size(0)
                chunk_count += x_num.size(0)

            # 청크 끝에서만 CPU로 가져옴
            tr_loss_sum += chunk_loss.item()
            tr_count += chunk_count

            # 청크 메모리 해제
            del Xn_chunk, Xcat_chunk, Yn_chunk, chunk_ds, chunk_loss
            if device == "cuda":
                torch.cuda.empty_cache()

            # 진행률 표시
            progress = (chunk_idx + 1) / n_train_chunks * 100
            print(f"\r  [Epoch {epoch:03d}/{epochs}] Train: chunk {chunk_idx+1}/{n_train_chunks} ({progress:.0f}%)", end="", flush=True)

        tr_loss = tr_loss_sum / max(1, tr_count)
        print()  # 줄바꿈

        # Validation
        model.eval()
        va_loss_sum = 0.0
        va_count = 0

        with torch.inference_mode():
            for chunk_idx, (Xn_chunk, Xcat_chunk, Yn_chunk, indices) in enumerate(chunked_trainer.get_val_chunks()):
                if Xn_chunk is None:
                    continue

                chunk_ds = TrajectoryDatasetGPU(Xn_chunk, Xcat_chunk, Yn_chunk, indices, seq_len, device)
                n_samples = len(chunk_ds)

                # 청크 내 loss 누적 (GPU에서)
                chunk_loss = torch.tensor(0.0, device=device)
                chunk_count = 0

                # DataLoader 없이 직접 배치 처리
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    x_num = chunk_ds.X_num[start:end]
                    x_cat = chunk_ds.X_cat[start:end]
                    y = chunk_ds.Y[start:end]
                    x_last = chunk_ds.X_last[start:end]

                    y_pred = model(x_num, x_cat)
                    loss = loss_with_smoothness(
                        y_pred, y, x_last,
                        smooth_lambda=smooth_lambda,
                        sog_lambda=sog_lambda,
                        heading_lambda=heading_lambda,
                        turn_boost=turn_boost,
                    )
                    chunk_loss += loss * x_num.size(0)
                    chunk_count += x_num.size(0)

                # 청크 끝에서만 CPU로 가져옴
                va_loss_sum += chunk_loss.item()
                va_count += chunk_count

                # 청크 메모리 해제
                del Xn_chunk, Xcat_chunk, Yn_chunk, chunk_ds, chunk_loss
                if device == "cuda":
                    torch.cuda.empty_cache()

                # 진행률 표시
                progress = (chunk_idx + 1) / n_val_chunks * 100
                print(f"\r  [Epoch {epoch:03d}/{epochs}] Val: chunk {chunk_idx+1}/{n_val_chunks} ({progress:.0f}%)", end="", flush=True)

        va_loss = va_loss_sum / max(1, va_count)
        print()  # 줄바꿈

        cur_lr = optimizer.param_groups[0]["lr"]

        # Epoch 결과
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

        # 체크포인트 저장
        epoch_model_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"  [CHECKPOINT] epoch {epoch} 저장")

        # Best 모델 갱신
        improved = (best_val - va_loss) > min_delta
        if improved:
            best_val = va_loss
            best_epoch = epoch
            bad_count = 0

            # Best 모델 저장
            torch.save(model.state_dict(), model_path)
            np.savez(
                scaler_path,
                x_mean=meta['x_mean'],
                x_std=meta['x_std'],
                y_mean=meta['y_mean'],
                y_std=meta['y_std'],
                seq_len=int(seq_len),
                stride=int(meta['stride']),
                numeric_cols=meta['numeric_cols'],
                cat_cols=meta['cat_cols'],
                target_cols=meta['target_cols'],
                mmsi_vocab=meta['mmsi_vocab'],
                start_vocab=meta['start_vocab'],
                end_vocab=meta['end_vocab'],
                num_mmsi=int(meta['num_mmsi']),
                num_start_area=int(meta['num_start_area']),
                num_end_area=int(meta['num_end_area']),
                grid_info_lat_min=float(meta['grid_info_lat_min']),
                grid_info_lon_min=float(meta['grid_info_lon_min']),
                grid_info_lat_max=float(meta['grid_info_lat_max']),
                grid_info_lon_max=float(meta['grid_info_lon_max']),
                grid_size=float(meta['grid_size']),
                num_rows=int(meta['num_rows']),
                num_cols=int(meta['num_cols']),
                total_grids=int(meta['total_grids']),
                cog_mirror=bool(meta['cog_mirror']),
                embed_dim=int(embed_dim),
                best_epoch=int(best_epoch),
                best_val=float(best_val),
                smooth_lambda=float(smooth_lambda),
                sog_lambda=float(sog_lambda),
                heading_lambda=float(heading_lambda),
                lat_bounds=meta['lat_bounds'],
                lon_bounds=meta['lon_bounds'],
            )
            print(f"  [BEST] 최고 모델 갱신됨 (epoch={epoch}, val_loss={va_loss:.6f})")
        else:
            bad_count += 1

        # Early stopping
        if epoch >= warmup_epochs and bad_count >= patience:
            print(f"[INFO] Early stop at epoch {epoch} (best={best_epoch}, val={best_val:.6f})")
            break

        if cur_lr <= min_lr + 1e-12 and epoch >= warmup_epochs:
            print(f"[INFO] Stop: lr reached min_lr (best={best_epoch}, val={best_val:.6f})")
            break

    print(f"\n[INFO] 학습 완료! best epoch={best_epoch}, val={best_val:.6f}")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")

    del model, optimizer
    gc.collect()

    return model_path, scaler_path


def main():
    parser = argparse.ArgumentParser(
        description="전처리된 데이터로 모델 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str, default="prepared_data",
                        help="전처리된 데이터 폴더 (기본값: prepared_data)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="최대 학습 에폭 (기본값: 300)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="배치 크기 (기본값: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률 (기본값: 0.001)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (기본값: 20)")
    parser.add_argument("--warmup_epochs", type=int, default=30,
                        help="Warmup 에폭 수 (기본값: 30)")
    parser.add_argument("--smooth_lambda", type=float, default=0.05,
                        help="Smoothness 정규화 계수 (기본값: 0.05)")
    parser.add_argument("--heading_lambda", type=float, default=0.02,
                        help="침로 smoothness 계수 (기본값: 0.02)")
    parser.add_argument("--turn_boost", type=float, default=2.0,
                        help="변침 구간 가중치 (기본값: 2.0)")
    parser.add_argument("--embed_dim", type=int, default=16,
                        help="Embedding 차원 (기본값: 16)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="검증 데이터 비율 (기본값: 0.2)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="학습 장치 (기본값: cuda)")
    parser.add_argument("--save_dir", type=str, default="global_model_v2",
                        help="모델 저장 폴더 (기본값: global_model_v2)")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="한 번에 GPU에 올릴 segment 수 (기본값: 100)")

    args = parser.parse_args()

    print("=" * 60)
    print("모델 학습 (전처리된 데이터 사용)")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_dir}")
    print(f"저장 폴더: {args.save_dir}")
    print(f"장치: {args.device}")
    print(f"Epochs: {args.epochs}")

    # CUDA 확인
    print(f"\n[PyTorch] version: {torch.__version__}")
    print(f"[CUDA] available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA] device: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    model_path, scaler_path = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        smooth_lambda=args.smooth_lambda,
        heading_lambda=args.heading_lambda,
        turn_boost=args.turn_boost,
        embed_dim=args.embed_dim,
        val_ratio=args.val_ratio,
        chunk_size=args.chunk_size,
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"모델: {model_path}")
    print(f"스케일러: {scaler_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
