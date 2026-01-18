# -*- coding: utf-8 -*-
"""
V11 Inertia-Aware Transformer - 지역별 학습
============================================
- 특정 지역 폴더 지정하여 학습
- 입력: (lat, lon, sog, cog) × 30 스텝 (30분)
- 출력: (Δlat, Δlon) × 60 스텝 (60분 예측)

V11 변경사항 (V10 대비):
- 입력 시퀀스: 50분 → 30분
- 예측 시퀀스: 100분 → 60분
- 모델 경량화 (hidden_dim: 256→192, num_layers: 6→4)

사용법:
    python run_train_v11_region.py --region 부산
    python run_train_v11_region.py --region 울산 --epochs 100
    python run_train_v11_region.py --data_dir "경로/지역폴더"
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import warnings
import math
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

# 기본 설정
BASE_DATA_DIR = r"K:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리\v11"
BASE_SAVE_DIR = r"K:\coding_project\NIA_선박항적예측프로그램\models\v11"
SEQ_LEN = 30   # 입력 시퀀스 길이 (30분)
PRED_LEN = 60  # 예측 길이 (60분)


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


class TrajectoryTransformerV11(nn.Module):
    """V11 Inertia-Aware Transformer (경량화)"""
    def __init__(self, input_dim=5, hidden_dim=192, num_heads=6,
                 num_layers=4, dropout=0.1, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        encoded = self.transformer(x)
        out = self.output_mlp(encoded[:, -1, :])
        return out.view(B, self.pred_len, 2)


class TrajectoryDatasetV11(Dataset):
    """V11 Dataset"""
    def __init__(self, sequences):
        self.sequences = sequences
        print(f"  시퀀스 수: {len(self.sequences):,}개")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        inp = sample['input'].astype(np.float32)
        target = sample['target'].astype(np.float32)

        inp_norm = self._normalize_input(inp)

        last_pos = inp[-1, :2]
        delta_target = (target - last_pos) * 100

        last_velocity = (inp[-1, :2] - inp[-2, :2]) * 100

        return (
            torch.from_numpy(inp_norm),
            torch.from_numpy(delta_target.astype(np.float32)),
            torch.from_numpy(last_velocity.astype(np.float32))
        )

    def _normalize_input(self, inp):
        seq_len = inp.shape[0]
        inp_norm = np.zeros((seq_len, 5), dtype=np.float32)

        inp_norm[:, 0] = (inp[:, 0] - inp[0, 0]) * 100
        inp_norm[:, 1] = (inp[:, 1] - inp[0, 1]) * 100
        inp_norm[:, 2] = inp[:, 2] / 30.0

        cog_rad = np.radians(inp[:, 3])
        inp_norm[:, 3] = np.sin(cog_rad)
        inp_norm[:, 4] = np.cos(cog_rad)

        return inp_norm


def load_sequences(data_dir, max_chunks=None):
    """청크 파일에서 시퀀스 로드"""
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "sequences_chunk_*.pkl")))

    if not chunk_files:
        raise FileNotFoundError(f"청크 파일이 없습니다: {data_dir}")

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


def compute_inertia_loss(pred_delta, last_velocity):
    """Inertia Loss"""
    first_pred_velocity = pred_delta[:, 0, :]
    second_pred_velocity = pred_delta[:, 1, :] - pred_delta[:, 0, :]

    loss1 = F.mse_loss(first_pred_velocity, last_velocity)
    loss2 = F.mse_loss(first_pred_velocity, second_pred_velocity)

    return loss1 + 0.5 * loss2


def compute_smoothness_loss(pred_delta, num_steps=10):
    """Smoothness Loss"""
    velocities = pred_delta[:, 1:num_steps+1, :] - pred_delta[:, :num_steps, :]
    accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]
    return torch.mean(accelerations ** 2)


def evaluate_model(model, val_loader, device, pred_len=60):
    """모델 평가"""
    model.eval()
    all_errors = []

    with torch.no_grad():
        for inp, delta_target, _ in val_loader:
            inp = inp.to(device)
            delta_target = delta_target.to(device)

            pred_delta = model(inp)

            pred_delta_np = pred_delta.cpu().numpy() / 100
            target_delta_np = delta_target.cpu().numpy() / 100

            for b in range(pred_delta_np.shape[0]):
                errors = []
                for t in range(pred_delta_np.shape[1]):
                    dlat = pred_delta_np[b, t, 0] - target_delta_np[b, t, 0]
                    dlon = pred_delta_np[b, t, 1] - target_delta_np[b, t, 1]
                    dist_km = np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(35)))**2)
                    errors.append(dist_km)
                all_errors.append(errors)

    return np.array(all_errors)


def get_available_regions(base_dir):
    """사용 가능한 지역 목록"""
    regions = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # 청크 파일이 있는지 확인
                if glob.glob(os.path.join(item_path, "sequences_chunk_*.pkl")):
                    regions.append(item)
    return sorted(regions)


def train_model(region, data_dir, save_dir, device, max_chunks=None, epochs=50,
                batch_size=128, lr=5e-4, inertia_weight=0.05, smoothness_weight=0.01):
    """모델 학습"""
    print("=" * 60)
    print(f"V11 Inertia-Aware Transformer - {region} 60분 예측")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"데이터: {data_dir}")
    print(f"저장: {save_dir}")
    print(f"SEQ_LEN: {SEQ_LEN}, PRED_LEN: {PRED_LEN}")
    print(f"Inertia Loss 가중치: {inertia_weight}")
    print(f"Smoothness Loss 가중치: {smoothness_weight}")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    sequences = load_sequences(data_dir, max_chunks=max_chunks)

    if len(sequences) < 100:
        print(f"[경고] 시퀀스가 너무 적습니다: {len(sequences)}개")
        return None

    # 2. Train/Val 분할
    print("\n[2] Train/Val 분할")
    np.random.seed(42)
    indices = np.random.permutation(len(sequences))
    val_size = int(len(sequences) * 0.1)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    print(f"Train: {len(train_sequences):,}, Val: {len(val_sequences):,}")

    # 3. Dataset 생성
    print("\n[3] Dataset 생성")
    train_dataset = TrajectoryDatasetV11(train_sequences)
    val_dataset = TrajectoryDatasetV11(val_sequences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 4. 모델 생성
    print("\n[4] 모델 생성")
    model = TrajectoryTransformerV11(
        input_dim=5,
        hidden_dim=192,
        num_heads=6,
        num_layers=4,
        dropout=0.1,
        pred_len=PRED_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    coord_criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. 학습
    print("\n[5] 학습 시작")
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_coord_loss = 0
        train_inertia_loss = 0
        train_smooth_loss = 0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for inp, delta_target, last_velocity in pbar:
            inp = inp.to(device)
            delta_target = delta_target.to(device)
            last_velocity = last_velocity.to(device)

            optimizer.zero_grad()
            pred_delta = model(inp)

            loss_coord = coord_criterion(pred_delta, delta_target)
            loss_inertia = compute_inertia_loss(pred_delta, last_velocity)
            loss_smooth = compute_smoothness_loss(pred_delta, num_steps=10)

            loss = loss_coord + inertia_weight * loss_inertia + smoothness_weight * loss_smooth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * inp.size(0)
            train_coord_loss += loss_coord.item() * inp.size(0)
            train_inertia_loss += loss_inertia.item() * inp.size(0)
            train_smooth_loss += loss_smooth.item() * inp.size(0)
            train_count += inp.size(0)

            pbar.set_postfix({
                'total': f'{loss.item():.4f}',
                'coord': f'{loss_coord.item():.4f}'
            })

        train_loss /= train_count
        train_coord_loss /= train_count
        train_inertia_loss /= train_count
        train_smooth_loss /= train_count

        # Validation
        model.eval()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for inp, delta_target, last_velocity in val_loader:
                inp = inp.to(device)
                delta_target = delta_target.to(device)
                last_velocity = last_velocity.to(device)

                pred_delta = model(inp)

                loss_coord = coord_criterion(pred_delta, delta_target)
                loss_inertia = compute_inertia_loss(pred_delta, last_velocity)
                loss_smooth = compute_smoothness_loss(pred_delta, num_steps=10)

                loss = loss_coord + inertia_weight * loss_inertia + smoothness_weight * loss_smooth

                val_loss += loss.item() * inp.size(0)
                val_count += inp.size(0)

        val_loss /= val_count

        # 실제 거리 오차 (5 에포크마다)
        if epoch % 5 == 0 or epoch == 1:
            errors = evaluate_model(model, val_loader, device, pred_len=PRED_LEN)
            avg_error = np.mean(errors)
            error_10min = np.mean(errors[:, 9]) if errors.shape[1] > 9 else 0
            error_30min = np.mean(errors[:, 29]) if errors.shape[1] > 29 else 0
            error_60min = np.mean(errors[:, 59]) if errors.shape[1] > 59 else 0
            print(f"  오차 - 평균: {avg_error:.2f}km, 10분: {error_10min:.2f}km, "
                  f"30분: {error_30min:.2f}km, 60분: {error_60min:.2f}km")

        scheduler.step()

        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 매 epoch 모델 저장
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch:03d}.pth'))

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0

            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
            print(f"  -> Best 모델 저장됨 (Val Loss: {best_val_loss:.6f})")

            # 설정 저장
            config = {
                'version': 'v11',
                'region': region,
                'input_dim': 5,
                'hidden_dim': 192,
                'num_heads': 6,
                'num_layers': 4,
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'inertia_weight': inertia_weight,
                'smoothness_weight': smoothness_weight,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }
            with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
                pickle.dump(config, f)
        else:
            patience_count += 1
            if patience_count >= 15:
                print(f"Early stopping at epoch {epoch}")
                break

    # 최종 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_best.pth')))
    errors = evaluate_model(model, val_loader, device, pred_len=PRED_LEN)

    print(f"평균 오차: {np.mean(errors):.2f} km")
    print(f"10분 후: {np.mean(errors[:, 9]):.2f} km")
    print(f"30분 후: {np.mean(errors[:, 29]):.2f} km")
    print(f"60분 후: {np.mean(errors[:, 59]):.2f} km")

    print(f"\n저장 위치: {save_dir}/")
    print("=" * 60)

    return {
        'region': region,
        'best_val_loss': best_val_loss,
        'avg_error': np.mean(errors),
        'error_60min': np.mean(errors[:, 59])
    }


def main():
    parser = argparse.ArgumentParser(
        description='V11 Inertia-Aware 60분 예측 모델 - 지역별 학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
    python run_train_v11_region.py --list
    python run_train_v11_region.py --region 울산
    python run_train_v11_region.py --region 부산 --epochs 100
        """
    )
    parser.add_argument('--region', type=str, default=None,
                        help='학습할 지역 이름 (예: 부산, 울산)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 폴더 직접 지정 (--region 대신 사용)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='모델 저장 폴더 (기본: models/v11/지역명)')
    parser.add_argument('--max_chunks', type=int, default=None,
                        help='최대 청크 수 (None=전체)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='배치 크기')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='학습률')
    parser.add_argument('--inertia_weight', type=float, default=0.05,
                        help='Inertia Loss 가중치')
    parser.add_argument('--smoothness_weight', type=float, default=0.01,
                        help='Smoothness Loss 가중치')
    parser.add_argument('--list', action='store_true',
                        help='사용 가능한 지역 목록 출력')

    args = parser.parse_args()

    # 지역 목록 출력
    if args.list:
        regions = get_available_regions(BASE_DATA_DIR)
        print("사용 가능한 지역 목록:")
        for region in regions:
            data_path = os.path.join(BASE_DATA_DIR, region)
            chunk_count = len(glob.glob(os.path.join(data_path, "sequences_chunk_*.pkl")))
            print(f"  - {region} ({chunk_count} 청크)")
        return

    # 데이터 폴더 결정
    if args.data_dir:
        data_dir = args.data_dir
        region = os.path.basename(data_dir)
    elif args.region:
        region = args.region
        data_dir = os.path.join(BASE_DATA_DIR, region)
    else:
        print("[오류] --region 또는 --data_dir 를 지정하세요.")
        print("       사용 가능한 지역: python run_train_v11_region.py --list")
        return

    # 데이터 폴더 확인
    if not os.path.exists(data_dir):
        print(f"[오류] 데이터 폴더가 없습니다: {data_dir}")
        return

    # 저장 폴더 결정 (BASE_SAVE_DIR 하위에 저장)
    if args.save_dir:
        save_dir = os.path.join(BASE_SAVE_DIR, args.save_dir)
    else:
        save_dir = os.path.join(BASE_SAVE_DIR, region)

    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 학습
    train_model(
        region=region,
        data_dir=data_dir,
        save_dir=save_dir,
        device=device,
        max_chunks=args.max_chunks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        inertia_weight=args.inertia_weight,
        smoothness_weight=args.smoothness_weight
    )


if __name__ == "__main__":
    main()
