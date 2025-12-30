# -*- coding: utf-8 -*-
"""
격자 기반 Seq2Seq 모델 학습 (V4)
================================
- Encoder-Decoder Transformer 구조
- 격자 분류 (CrossEntropy) + SOG/COG 회귀 (MSE)
- Teacher Forcing 학습

사용법:
    python train_model_v4.py --data_dir "prepared_data_v4" \
                              --epochs 100 --batch_size 256 --lr 0.001
"""

import os
import math
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import freeze_support


# ============================================
# Positional Encoding
# ============================================

class PositionalEncoding(nn.Module):
    """Transformer용 위치 인코딩"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# Grid Seq2Seq Model
# ============================================

class GridSeq2SeqModel(nn.Module):
    """
    격자 기반 Seq2Seq 모델

    - Encoder: 과거 시퀀스 인코딩
    - Decoder: 미래 시퀀스 생성 (Teacher Forcing)
    - Dual-head output: 분류(격자) + 회귀(SOG/COG)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_start_area: int = 10,
        num_end_area: int = 10,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 1. Embeddings
        self.grid_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim // 4)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim // 4)

        # 2. Input projection
        # grid_embed + sog + sin_cog + cos_cog + start_embed + end_embed
        enc_input_dim = embed_dim + 3 + 2 * (embed_dim // 4)
        self.enc_proj = nn.Linear(enc_input_dim, hidden_dim)

        # 3. Positional Encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)

        # 4. Encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5. Decoder Input projection
        dec_input_dim = embed_dim + 3  # prev grid + prev sog/cog
        self.dec_proj = nn.Linear(dec_input_dim, hidden_dim)

        # 6. Decoder (Transformer with cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 7. Output heads
        self.grid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.sog_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.cog_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # sin, cos
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        enc_grid_ids: torch.Tensor,    # [B, seq_len]
        enc_sog: torch.Tensor,         # [B, seq_len]
        enc_cog_sin: torch.Tensor,     # [B, seq_len]
        enc_cog_cos: torch.Tensor,     # [B, seq_len]
        start_area_id: torch.Tensor,   # [B]
        end_area_id: torch.Tensor,     # [B]
        dec_grid_ids: torch.Tensor,    # [B, pred_len] (teacher forcing)
        dec_sog: torch.Tensor,         # [B, pred_len]
        dec_cog_sin: torch.Tensor,     # [B, pred_len]
        dec_cog_cos: torch.Tensor,     # [B, pred_len]
    ):
        """
        학습 시 forward (Teacher Forcing)

        Returns:
            grid_logits: [B, pred_len, vocab_size]
            sog_pred: [B, pred_len]
            cog_pred: [B, pred_len, 2]
        """
        B, seq_len = enc_grid_ids.shape
        _, pred_len = dec_grid_ids.shape

        # === ENCODER ===
        # Embeddings
        grid_emb = self.grid_embed(enc_grid_ids)  # [B, seq_len, embed_dim]
        start_emb = self.start_area_embed(start_area_id).unsqueeze(1).expand(-1, seq_len, -1)
        end_emb = self.end_area_embed(end_area_id).unsqueeze(1).expand(-1, seq_len, -1)

        # Concat all features
        enc_input = torch.cat([
            grid_emb,
            enc_sog.unsqueeze(-1),
            enc_cog_sin.unsqueeze(-1),
            enc_cog_cos.unsqueeze(-1),
            start_emb,
            end_emb,
        ], dim=-1)

        enc_input = self.enc_proj(enc_input)
        enc_input = self.pos_enc(enc_input)

        # Encode
        memory = self.encoder(enc_input)  # [B, seq_len, hidden_dim]

        # === DECODER ===
        # Shift right: prepend last encoder state
        dec_grid_ids_shifted = torch.cat([
            enc_grid_ids[:, -1:], dec_grid_ids[:, :-1]
        ], dim=1)
        dec_sog_shifted = torch.cat([
            enc_sog[:, -1:], dec_sog[:, :-1]
        ], dim=1)
        dec_cog_sin_shifted = torch.cat([
            enc_cog_sin[:, -1:], dec_cog_sin[:, :-1]
        ], dim=1)
        dec_cog_cos_shifted = torch.cat([
            enc_cog_cos[:, -1:], dec_cog_cos[:, :-1]
        ], dim=1)

        dec_grid_emb = self.grid_embed(dec_grid_ids_shifted)
        dec_input = torch.cat([
            dec_grid_emb,
            dec_sog_shifted.unsqueeze(-1),
            dec_cog_sin_shifted.unsqueeze(-1),
            dec_cog_cos_shifted.unsqueeze(-1),
        ], dim=-1)

        dec_input = self.dec_proj(dec_input)
        dec_input = self.pos_enc(dec_input)

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            pred_len, device=enc_grid_ids.device
        )

        # Decode
        dec_out = self.decoder(dec_input, memory, tgt_mask=tgt_mask)

        # === OUTPUT HEADS ===
        grid_logits = self.grid_head(dec_out)  # [B, pred_len, vocab_size]
        sog_pred = self.sog_head(dec_out).squeeze(-1)  # [B, pred_len]
        cog_pred = self.cog_head(dec_out)  # [B, pred_len, 2]

        return grid_logits, sog_pred, cog_pred

    def inference(
        self,
        enc_grid_ids: torch.Tensor,
        enc_sog: torch.Tensor,
        enc_cog_sin: torch.Tensor,
        enc_cog_cos: torch.Tensor,
        start_area_id: torch.Tensor,
        end_area_id: torch.Tensor,
        pred_len: int,
    ):
        """
        추론 시 forward (Autoregressive)

        Returns:
            grid_ids: [B, pred_len]
            sog_pred: [B, pred_len]
            cog_pred: [B, pred_len, 2]
        """
        B, seq_len = enc_grid_ids.shape
        device = enc_grid_ids.device

        # === ENCODER ===
        grid_emb = self.grid_embed(enc_grid_ids)
        start_emb = self.start_area_embed(start_area_id).unsqueeze(1).expand(-1, seq_len, -1)
        end_emb = self.end_area_embed(end_area_id).unsqueeze(1).expand(-1, seq_len, -1)

        enc_input = torch.cat([
            grid_emb,
            enc_sog.unsqueeze(-1),
            enc_cog_sin.unsqueeze(-1),
            enc_cog_cos.unsqueeze(-1),
            start_emb,
            end_emb,
        ], dim=-1)

        enc_input = self.enc_proj(enc_input)
        enc_input = self.pos_enc(enc_input)
        memory = self.encoder(enc_input)

        # === AUTOREGRESSIVE DECODER ===
        # 시작 토큰: 마지막 인코더 상태
        current_grid = enc_grid_ids[:, -1:]
        current_sog = enc_sog[:, -1:]
        current_cog_sin = enc_cog_sin[:, -1:]
        current_cog_cos = enc_cog_cos[:, -1:]

        pred_grids = []
        pred_sogs = []
        pred_cogs = []

        for _ in range(pred_len):
            dec_grid_emb = self.grid_embed(current_grid)
            dec_input = torch.cat([
                dec_grid_emb,
                current_sog.unsqueeze(-1),
                current_cog_sin.unsqueeze(-1),
                current_cog_cos.unsqueeze(-1),
            ], dim=-1)

            dec_input = self.dec_proj(dec_input)
            dec_input = self.pos_enc(dec_input)

            # No mask needed for single step
            dec_out = self.decoder(dec_input, memory)

            # Output
            grid_logits = self.grid_head(dec_out[:, -1:, :])
            sog_out = self.sog_head(dec_out[:, -1:, :]).squeeze(-1)
            cog_out = self.cog_head(dec_out[:, -1:, :])

            # Argmax for grid
            pred_grid = grid_logits.argmax(dim=-1)

            pred_grids.append(pred_grid)
            pred_sogs.append(sog_out)
            pred_cogs.append(cog_out)

            # Update for next step
            current_grid = pred_grid
            current_sog = sog_out
            current_cog_sin = cog_out[:, :, 0]
            current_cog_cos = cog_out[:, :, 1]

        pred_grids = torch.cat(pred_grids, dim=1)  # [B, pred_len]
        pred_sogs = torch.cat(pred_sogs, dim=1)    # [B, pred_len]
        pred_cogs = torch.cat(pred_cogs, dim=1)    # [B, pred_len, 2]

        return pred_grids, pred_sogs, pred_cogs


# ============================================
# Dataset
# ============================================

class GridTrajectoryDataset(Dataset):
    """격자 기반 항적 데이터셋"""

    def __init__(self, data_path: str):
        data = np.load(data_path)

        self.input_grid_ids = torch.from_numpy(data['input_grid_ids'])
        self.input_sog = torch.from_numpy(data['input_sog'])
        self.input_cog_sin = torch.from_numpy(data['input_cog_sin'])
        self.input_cog_cos = torch.from_numpy(data['input_cog_cos'])

        self.target_grid_ids = torch.from_numpy(data['target_grid_ids'])
        self.target_sog = torch.from_numpy(data['target_sog'])
        self.target_cog_sin = torch.from_numpy(data['target_cog_sin'])
        self.target_cog_cos = torch.from_numpy(data['target_cog_cos'])

        self.start_area_ids = torch.from_numpy(data['start_area_ids'])
        self.end_area_ids = torch.from_numpy(data['end_area_ids'])

    def __len__(self):
        return len(self.input_grid_ids)

    def __getitem__(self, idx):
        return {
            'input_grid_ids': self.input_grid_ids[idx],
            'input_sog': self.input_sog[idx],
            'input_cog_sin': self.input_cog_sin[idx],
            'input_cog_cos': self.input_cog_cos[idx],
            'target_grid_ids': self.target_grid_ids[idx],
            'target_sog': self.target_sog[idx],
            'target_cog_sin': self.target_cog_sin[idx],
            'target_cog_cos': self.target_cog_cos[idx],
            'start_area_id': self.start_area_ids[idx],
            'end_area_id': self.end_area_ids[idx],
        }


# ============================================
# Loss Function
# ============================================

class GridPredictionLoss(nn.Module):
    """복합 손실 함수: CrossEntropy(격자) + MSE(SOG/COG)"""

    def __init__(
        self,
        grid_weight: float = 1.0,
        sog_weight: float = 0.5,
        cog_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.grid_weight = grid_weight
        self.sog_weight = sog_weight
        self.cog_weight = cog_weight

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=0  # <PAD> 무시
        )
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        grid_logits: torch.Tensor,  # [B, pred_len, vocab_size]
        sog_pred: torch.Tensor,     # [B, pred_len]
        cog_pred: torch.Tensor,     # [B, pred_len, 2]
        grid_target: torch.Tensor,  # [B, pred_len]
        sog_target: torch.Tensor,   # [B, pred_len]
        cog_target: torch.Tensor,   # [B, pred_len, 2]
    ):
        B, pred_len, vocab_size = grid_logits.shape

        # 1. Grid Classification Loss
        grid_logits_flat = grid_logits.view(-1, vocab_size)
        grid_target_flat = grid_target.view(-1)
        grid_loss = self.ce_loss(grid_logits_flat, grid_target_flat)

        # 2. SOG Regression Loss
        sog_loss = self.mse_loss(sog_pred, sog_target)

        # 3. COG Regression Loss (sin/cos)
        cog_loss = self.mse_loss(cog_pred, cog_target)

        # Total Loss
        total_loss = (
            self.grid_weight * grid_loss +
            self.sog_weight * sog_loss +
            self.cog_weight * cog_loss
        )

        return {
            'total': total_loss,
            'grid': grid_loss,
            'sog': sog_loss,
            'cog': cog_loss,
        }


# ============================================
# Training Functions
# ============================================

def calculate_grid_accuracy(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 0):
    """격자 정확도 계산

    Args:
        pred: [B, pred_len] 예측 격자 ID
        target: [B, pred_len] 타겟 격자 ID
        tolerance: 인접 격자 허용 범위 (현재 미구현, 단순 정확도만)

    Returns:
        accuracy: float
    """
    correct = (pred == target).float().mean()
    return correct.item()


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0, epoch=0):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    total_grid_loss = 0
    total_sog_loss = 0
    total_cog_loss = 0
    total_acc = 0
    n_batches = 0
    total_batches = len(dataloader)
    print_interval = max(1, total_batches // 100)  # 1% 마다 출력

    print(f"  [DEBUG] train_epoch 시작: {total_batches} 배치 (출력 간격: {print_interval})", flush=True)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            print(f"  [DEBUG] 첫 배치 로드 완료, forward 시작...", flush=True)
        # 데이터 이동
        input_grid_ids = batch['input_grid_ids'].to(device)
        input_sog = batch['input_sog'].to(device)
        input_cog_sin = batch['input_cog_sin'].to(device)
        input_cog_cos = batch['input_cog_cos'].to(device)
        target_grid_ids = batch['target_grid_ids'].to(device)
        target_sog = batch['target_sog'].to(device)
        target_cog_sin = batch['target_cog_sin'].to(device)
        target_cog_cos = batch['target_cog_cos'].to(device)
        start_area_id = batch['start_area_id'].to(device)
        end_area_id = batch['end_area_id'].to(device)

        optimizer.zero_grad()

        # Forward (Teacher Forcing)
        grid_logits, sog_pred, cog_pred = model(
            input_grid_ids, input_sog, input_cog_sin, input_cog_cos,
            start_area_id, end_area_id,
            target_grid_ids, target_sog, target_cog_sin, target_cog_cos
        )
        if batch_idx == 0:
            print(f"  [DEBUG] 첫 forward 완료, loss 계산 중...", flush=True)

        # Loss
        cog_target = torch.stack([target_cog_sin, target_cog_cos], dim=-1)
        losses = criterion(
            grid_logits, sog_pred, cog_pred,
            target_grid_ids, target_sog, cog_target
        )

        loss = losses['total']
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 통계
        total_loss += loss.item()
        total_grid_loss += losses['grid'].item()
        total_sog_loss += losses['sog'].item()
        total_cog_loss += losses['cog'].item()

        pred_grids = grid_logits.argmax(dim=-1)
        total_acc += calculate_grid_accuracy(pred_grids, target_grid_ids)
        n_batches += 1

        # 진행률 출력
        if (batch_idx + 1) % print_interval == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches * 100
            print(f"\r  [Epoch {epoch+1}] {batch_idx+1}/{total_batches} ({progress:.0f}%) - "
                  f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.1f}%", end='', flush=True)

    print()  # 줄바꿈

    return {
        'loss': total_loss / n_batches,
        'grid_loss': total_grid_loss / n_batches,
        'sog_loss': total_sog_loss / n_batches,
        'cog_loss': total_cog_loss / n_batches,
        'accuracy': total_acc / n_batches,
    }


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    total_grid_loss = 0
    total_sog_loss = 0
    total_cog_loss = 0
    total_acc = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_grid_ids = batch['input_grid_ids'].to(device)
            input_sog = batch['input_sog'].to(device)
            input_cog_sin = batch['input_cog_sin'].to(device)
            input_cog_cos = batch['input_cog_cos'].to(device)
            target_grid_ids = batch['target_grid_ids'].to(device)
            target_sog = batch['target_sog'].to(device)
            target_cog_sin = batch['target_cog_sin'].to(device)
            target_cog_cos = batch['target_cog_cos'].to(device)
            start_area_id = batch['start_area_id'].to(device)
            end_area_id = batch['end_area_id'].to(device)

            grid_logits, sog_pred, cog_pred = model(
                input_grid_ids, input_sog, input_cog_sin, input_cog_cos,
                start_area_id, end_area_id,
                target_grid_ids, target_sog, target_cog_sin, target_cog_cos
            )

            cog_target = torch.stack([target_cog_sin, target_cog_cos], dim=-1)
            losses = criterion(
                grid_logits, sog_pred, cog_pred,
                target_grid_ids, target_sog, cog_target
            )

            total_loss += losses['total'].item()
            total_grid_loss += losses['grid'].item()
            total_sog_loss += losses['sog'].item()
            total_cog_loss += losses['cog'].item()

            pred_grids = grid_logits.argmax(dim=-1)
            total_acc += calculate_grid_accuracy(pred_grids, target_grid_ids)
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'grid_loss': total_grid_loss / n_batches,
        'sog_loss': total_sog_loss / n_batches,
        'cog_loss': total_cog_loss / n_batches,
        'accuracy': total_acc / n_batches,
    }


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="격자 기반 Seq2Seq 모델 학습 (V4)")
    parser.add_argument("--data_dir", type=str, default="prepared_data_v4",
                        help="전처리된 데이터 폴더")
    parser.add_argument("--save_dir", type=str, default="model/yeosu",
                        help="모델 저장 폴더")
    parser.add_argument("--epochs", type=int, default=5,
                        help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="배치 크기")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="학습률")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="히든 차원")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="임베딩 차원")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Transformer 레이어 수")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Attention 헤드 수")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 비율")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="검증 데이터 비율")
    parser.add_argument("--device", type=str, default="cuda",
                        help="장치 (cuda/cpu)")

    args = parser.parse_args()

    # 장치 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("격자 기반 Seq2Seq 모델 학습 (V4)")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_dir}")
    print(f"장치: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    # 메타 정보 로드
    meta = dict(np.load(os.path.join(args.data_dir, "meta.npz"), allow_pickle=True))
    vocab_size = int(meta['vocab_size'])
    seq_len = int(meta['seq_len'])
    pred_len = int(meta['pred_len'])
    num_start_area = int(meta['num_start_area'])
    num_end_area = int(meta['num_end_area'])
    sog_mean = float(meta['sog_mean'])
    sog_std = float(meta['sog_std'])

    print(f"[INFO] 메타 정보:")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Seq len: {seq_len}")
    print(f"  - Pred len: {pred_len}")
    print(f"  - SOG mean/std: {sog_mean:.4f} / {sog_std:.4f}")
    print()

    # 데이터셋 로드
    data_path = os.path.join(args.data_dir, "training_data.npz")
    dataset = GridTrajectoryDataset(data_path)
    print(f"[INFO] 총 샘플 수: {len(dataset)}")

    # Train/Val 분할
    n_samples = len(dataset)
    n_val = int(n_samples * args.val_ratio)
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Windows 호환: num_workers=0 (멀티프로세싱 비활성화)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"  - Train: {n_train}, Val: {n_val}")
    print()

    # 모델 생성
    model = GridSeq2SeqModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        num_start_area=num_start_area,
        num_end_area=num_end_area,
        max_seq_len=seq_len + pred_len + 10,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 모델 파라미터:")
    print(f"  - Total: {n_params:,}")
    print(f"  - Trainable: {n_trainable:,}")
    print()

    # Loss, Optimizer, Scheduler
    criterion = GridPredictionLoss(
        grid_weight=1.0,
        sog_weight=0.3,
        cog_weight=0.3,
        label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 저장 폴더
    model_save_dir = os.path.join(args.save_dir, "model_v4")
    os.makedirs(model_save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(model_save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 학습 루프
    best_val_loss = float('inf')
    patience_counter = 0

    print("[INFO] 학습 시작")
    print("=" * 60)
    print(f"[DEBUG] 첫 배치 로딩 중...", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip, epoch=epoch-1
        )
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Grid Acc: {val_metrics['accuracy']:.4f}")

        # 체크포인트 저장
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Best 모델 저장
            torch.save(model.state_dict(), os.path.join(model_save_dir, "model_v4_best.pth"))

            # Scaler 저장
            np.savez(
                os.path.join(model_save_dir, "scaler_v4.npz"),
                version='v4',
                vocab_size=vocab_size,
                seq_len=seq_len,
                pred_len=pred_len,
                sog_mean=sog_mean,
                sog_std=sog_std,
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                start_vocab=meta['start_vocab'],
                end_vocab=meta['end_vocab'],
            )

            print(f"  [BEST] 최고 모델 저장 (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

        # 매 에폭마다 체크포인트 저장
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
        )
        print(f"  [SAVE] epoch_{epoch:03d}.pth 저장됨")

    print()
    print("=" * 60)
    print("학습 완료!")
    print(f"  - Best Val Loss: {best_val_loss:.4f}")
    print(f"  - 모델 저장: {model_save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()  # Windows 멀티프로세싱 지원
    main()
