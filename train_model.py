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
    전처리된 npz 데이터를 로드하여 사용하는 Dataset
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

    print(f"  - train segments: {len(train_seg_ids)}, sequences: {len(train_indices)}")
    print(f"  - val segments: {len(val_seg_ids)}, sequences: {len(val_indices)}")

    # Dataset & DataLoader
    train_ds = TrajectoryDatasetFromNpz(Xn_num, X_cat, Yn, train_indices, seq_len)
    val_ds = TrajectoryDatasetFromNpz(Xn_num, X_cat, Yn, val_indices, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

    for epoch in range(1, epochs + 1):
        # Train
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

        # Validation
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

    del model, optimizer, train_loader, val_loader
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

    args = parser.parse_args()

    print("=" * 60)
    print("모델 학습 (전처리된 데이터 사용)")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_dir}")
    print(f"저장 폴더: {args.save_dir}")
    print(f"장치: {args.device}")
    print(f"Epochs: {args.epochs}")
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
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"모델: {model_path}")
    print(f"스케일러: {scaler_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
