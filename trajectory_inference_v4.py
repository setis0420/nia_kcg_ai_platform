# -*- coding: utf-8 -*-
"""
격자 기반 항적 예측 추론 (V4)
=============================
- 격자 ID 기반 예측
- Autoregressive 디코딩
- 격자 → 위경도 변환

사용 예시:
    from trajectory_inference_v4 import GridTrajectoryInferenceV4

    inferencer = GridTrajectoryInferenceV4(
        model_path="model/yeosu/model_v4/model_v4_best.pth",
        vocab_path="prepared_data_v4/grid_vocab.json",
        scaler_path="model/yeosu/model_v4/scaler_v4.npz"
    )

    predictions = inferencer.predict(
        input_df,
        n_steps=30,
        start_area="남쪽 진입",
        end_area="여수정박지B"
    )
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 격자 크기 (도 단위)
GRID_SIZE = 0.002  # 약 222m


# ============================================
# 격자 변환 유틸리티
# ============================================

def latlon_to_grid_id(lat: float, lon: float) -> str:
    """위경도 → 격자 ID (문자열)"""
    lat_idx = int(lat / GRID_SIZE)
    lon_idx = int(lon / GRID_SIZE)
    return f"{lat_idx}_{lon_idx}"


def grid_id_to_latlon(grid_id: str) -> tuple:
    """격자 ID → 위경도 (격자 중심점)"""
    if grid_id in ['<PAD>', '<UNK>']:
        return None, None
    try:
        lat_idx, lon_idx = map(int, grid_id.split('_'))
        lat = (lat_idx + 0.5) * GRID_SIZE
        lon = (lon_idx + 0.5) * GRID_SIZE
        return lat, lon
    except:
        return None, None


class GridVocabulary:
    """격자 Vocabulary (로드 전용)"""

    def __init__(self):
        self.grid_to_idx = {}
        self.idx_to_grid = {}

    def encode(self, grid_id: str) -> int:
        return self.grid_to_idx.get(grid_id, 1)  # <UNK>

    def decode(self, idx: int) -> str:
        return self.idx_to_grid.get(idx, '<UNK>')

    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = cls()
        vocab.grid_to_idx = data['grid_to_idx']
        vocab.idx_to_grid = {int(k): v for k, v in data['idx_to_grid'].items()}
        return vocab


# ============================================
# 모델 정의 (train_model_v4에서 복사)
# ============================================

import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GridSeq2SeqModel(nn.Module):
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

        self.grid_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim // 4)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim // 4)

        enc_input_dim = embed_dim + 3 + 2 * (embed_dim // 4)
        self.enc_proj = nn.Linear(enc_input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        dec_input_dim = embed_dim + 3
        self.dec_proj = nn.Linear(dec_input_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

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
            nn.Linear(hidden_dim // 2, 2)
        )

    def encode(self, enc_grid_ids, enc_sog, enc_cog_sin, enc_cog_cos, start_area_id, end_area_id):
        """인코더만 실행"""
        B, seq_len = enc_grid_ids.shape

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

        return memory

    def decode_step(self, memory, current_grid, current_sog, current_cog_sin, current_cog_cos):
        """디코더 1스텝 실행"""
        dec_grid_emb = self.grid_embed(current_grid)
        dec_input = torch.cat([
            dec_grid_emb,
            current_sog.unsqueeze(-1),
            current_cog_sin.unsqueeze(-1),
            current_cog_cos.unsqueeze(-1),
        ], dim=-1)

        dec_input = self.dec_proj(dec_input)
        dec_input = self.pos_enc(dec_input)

        dec_out = self.decoder(dec_input, memory)

        grid_logits = self.grid_head(dec_out[:, -1:, :])
        sog_out = self.sog_head(dec_out[:, -1:, :]).squeeze(-1)
        cog_out = self.cog_head(dec_out[:, -1:, :])

        return grid_logits, sog_out, cog_out


# ============================================
# 추론 클래스
# ============================================

class GridTrajectoryInferenceV4:
    """격자 기반 항적 예측 추론기"""

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        scaler_path: str,
        device: str = None,
    ):
        # 장치 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[InferenceV4] 장치: {self.device}")

        # Vocabulary 로드
        self.grid_vocab = GridVocabulary.load(vocab_path)
        print(f"[InferenceV4] Vocabulary 크기: {len(self.grid_vocab.grid_to_idx)}")

        # Scaler 로드
        scaler = np.load(scaler_path, allow_pickle=True)
        self.vocab_size = int(scaler['vocab_size'])
        self.seq_len = int(scaler['seq_len'])
        self.pred_len = int(scaler['pred_len'])
        self.sog_mean = float(scaler['sog_mean'])
        self.sog_std = float(scaler['sog_std'])
        self.num_start_area = int(scaler['num_start_area'])
        self.num_end_area = int(scaler['num_end_area'])
        self.hidden_dim = int(scaler['hidden_dim'])
        self.embed_dim = int(scaler['embed_dim'])
        self.n_layers = int(scaler['n_layers'])
        self.n_heads = int(scaler['n_heads'])

        # Start/End Vocabulary
        self.start_vocab = eval(str(scaler['start_vocab']))
        self.end_vocab = eval(str(scaler['end_vocab']))

        print(f"[InferenceV4] seq_len={self.seq_len}, pred_len={self.pred_len}")
        print(f"[InferenceV4] SOG mean={self.sog_mean:.2f}, std={self.sog_std:.2f}")

        # 모델 로드
        self.model = GridSeq2SeqModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=0.0,  # 추론 시 dropout 비활성화
            num_start_area=self.num_start_area,
            num_end_area=self.num_end_area,
            max_seq_len=self.seq_len + self.pred_len + 10,  # 학습 시와 동일
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[InferenceV4] 모델 로드 완료: {model_path}")

    def _prepare_input(self, df: pd.DataFrame, start_area: str = None, end_area: str = None):
        """입력 데이터 준비"""
        if len(df) < self.seq_len:
            raise ValueError(f"입력 데이터 길이({len(df)})가 seq_len({self.seq_len})보다 적습니다.")

        # 마지막 seq_len개 사용
        hist = df.iloc[-self.seq_len:].copy()

        # 격자 ID 생성
        grid_ids = [latlon_to_grid_id(r['lat'], r['lon']) for _, r in hist.iterrows()]
        grid_indices = np.array([self.grid_vocab.encode(g) for g in grid_ids], dtype=np.int64)

        # SOG 정규화
        sog = (hist['sog'].values - self.sog_mean) / self.sog_std

        # COG sin/cos
        cog_rad = np.radians(hist['cog'].values)
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)

        # Start/End Area ID
        start_id = self.start_vocab.get(start_area, 0) if start_area else 0
        end_id = self.end_vocab.get(end_area, 0) if end_area else 0

        return {
            'grid_ids': torch.from_numpy(grid_indices).unsqueeze(0).to(self.device),
            'sog': torch.from_numpy(sog.astype(np.float32)).unsqueeze(0).to(self.device),
            'cog_sin': torch.from_numpy(cog_sin.astype(np.float32)).unsqueeze(0).to(self.device),
            'cog_cos': torch.from_numpy(cog_cos.astype(np.float32)).unsqueeze(0).to(self.device),
            'start_area_id': torch.tensor([start_id], dtype=torch.long).to(self.device),
            'end_area_id': torch.tensor([end_id], dtype=torch.long).to(self.device),
            'last_time': hist['datetime'].iloc[-1] if 'datetime' in hist.columns else pd.Timestamp.now(),
            'last_lat': hist['lat'].iloc[-1],
            'last_lon': hist['lon'].iloc[-1],
        }

    def predict(
        self,
        df: pd.DataFrame,
        n_steps: int = None,
        start_area: str = None,
        end_area: str = None,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> pd.DataFrame:
        """
        다중 스텝 예측 (Autoregressive)

        Args:
            df: 입력 데이터 (datetime, lat, lon, sog, cog 컬럼 필요)
            n_steps: 예측 스텝 수 (None이면 pred_len 사용)
            start_area: 출발 해역
            end_area: 도착 해역
            temperature: 샘플링 온도 (1.0=argmax, <1.0=더 확실한 선택)
            top_k: Top-K 샘플링 (None=argmax)

        Returns:
            DataFrame: [datetime, lat, lon, grid_id, sog, cog, confidence]
        """
        if n_steps is None:
            n_steps = self.pred_len

        # 입력 준비
        inputs = self._prepare_input(df, start_area, end_area)

        # 인코더 실행
        with torch.inference_mode():
            memory = self.model.encode(
                inputs['grid_ids'],
                inputs['sog'],
                inputs['cog_sin'],
                inputs['cog_cos'],
                inputs['start_area_id'],
                inputs['end_area_id'],
            )

        # Autoregressive 디코딩
        current_grid = inputs['grid_ids'][:, -1:]
        current_sog = inputs['sog'][:, -1:]
        current_cog_sin = inputs['cog_sin'][:, -1:]
        current_cog_cos = inputs['cog_cos'][:, -1:]

        predictions = []
        last_time = inputs['last_time']

        for step in range(n_steps):
            with torch.inference_mode():
                grid_logits, sog_out, cog_out = self.model.decode_step(
                    memory, current_grid, current_sog, current_cog_sin, current_cog_cos
                )

            # 격자 선택
            grid_probs = F.softmax(grid_logits.squeeze(1) / temperature, dim=-1)

            if top_k is not None and top_k > 0:
                # Top-K 샘플링
                top_probs, top_indices = torch.topk(grid_probs, k=top_k, dim=-1)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                sample_idx = torch.multinomial(top_probs, num_samples=1)
                pred_grid_idx = top_indices.gather(-1, sample_idx).squeeze(-1)
                confidence = top_probs.gather(-1, sample_idx).squeeze(-1).item()
            else:
                # Argmax
                pred_grid_idx = grid_probs.argmax(dim=-1)
                confidence = grid_probs.max(dim=-1).values.item()

            # SOG 역정규화
            pred_sog = sog_out.squeeze().item() * self.sog_std + self.sog_mean
            pred_sog = max(0, min(35, pred_sog))  # 클리핑

            # COG 계산
            pred_cog_sin = cog_out[0, 0, 0].item()
            pred_cog_cos = cog_out[0, 0, 1].item()
            pred_cog = np.degrees(np.arctan2(pred_cog_sin, pred_cog_cos))
            pred_cog = (pred_cog + 360) % 360

            # 격자 → 위경도
            pred_grid_str = self.grid_vocab.decode(pred_grid_idx.item())
            pred_lat, pred_lon = grid_id_to_latlon(pred_grid_str)

            if pred_lat is None:
                # Unknown 격자인 경우 이전 위치 유지
                if predictions:
                    pred_lat = predictions[-1]['lat']
                    pred_lon = predictions[-1]['lon']
                else:
                    pred_lat = inputs['last_lat']
                    pred_lon = inputs['last_lon']

            # 시간 업데이트
            last_time = last_time + pd.Timedelta(minutes=1)

            predictions.append({
                'datetime': last_time,
                'lat': pred_lat,
                'lon': pred_lon,
                'grid_id': pred_grid_str,
                'sog': pred_sog,
                'cog': pred_cog,
                'confidence': confidence,
            })

            # 다음 스텝 입력 업데이트
            current_grid = pred_grid_idx.unsqueeze(0)
            current_sog = torch.tensor([[pred_sog]], device=self.device).float()
            current_sog = (current_sog - self.sog_mean) / self.sog_std
            current_cog_sin = torch.tensor([[pred_cog_sin]], device=self.device).float()
            current_cog_cos = torch.tensor([[pred_cog_cos]], device=self.device).float()

        return pd.DataFrame(predictions)

    def predict_with_beam(
        self,
        df: pd.DataFrame,
        n_steps: int = None,
        start_area: str = None,
        end_area: str = None,
        beam_width: int = 3,
    ) -> list:
        """
        Beam Search로 다중 경로 예측

        Returns:
            list of dict: 각 경로별 {score, predictions DataFrame}
        """
        if n_steps is None:
            n_steps = self.pred_len

        inputs = self._prepare_input(df, start_area, end_area)

        with torch.inference_mode():
            memory = self.model.encode(
                inputs['grid_ids'],
                inputs['sog'],
                inputs['cog_sin'],
                inputs['cog_cos'],
                inputs['start_area_id'],
                inputs['end_area_id'],
            )

        # 초기 beam: [(score, grid_seq, sog_seq, cog_seq, current_state)]
        initial_state = {
            'grid': inputs['grid_ids'][:, -1:],
            'sog': inputs['sog'][:, -1:],
            'cog_sin': inputs['cog_sin'][:, -1:],
            'cog_cos': inputs['cog_cos'][:, -1:],
        }

        beams = [(0.0, [], [], [], initial_state)]
        last_time = inputs['last_time']

        for step in range(n_steps):
            candidates = []

            for score, grids, sogs, cogs, state in beams:
                with torch.inference_mode():
                    grid_logits, sog_out, cog_out = self.model.decode_step(
                        memory, state['grid'], state['sog'], state['cog_sin'], state['cog_cos']
                    )

                log_probs = F.log_softmax(grid_logits.squeeze(1), dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, k=beam_width, dim=-1)

                for i in range(beam_width):
                    new_score = score + top_log_probs[0, i].item()
                    new_grid_idx = top_indices[0, i].item()

                    pred_sog = sog_out.squeeze().item() * self.sog_std + self.sog_mean
                    pred_sog = max(0, min(35, pred_sog))

                    pred_cog_sin = cog_out[0, 0, 0].item()
                    pred_cog_cos = cog_out[0, 0, 1].item()
                    pred_cog = np.degrees(np.arctan2(pred_cog_sin, pred_cog_cos))
                    pred_cog = (pred_cog + 360) % 360

                    new_state = {
                        'grid': torch.tensor([[new_grid_idx]], device=self.device),
                        'sog': torch.tensor([[(pred_sog - self.sog_mean) / self.sog_std]], device=self.device).float(),
                        'cog_sin': torch.tensor([[pred_cog_sin]], device=self.device).float(),
                        'cog_cos': torch.tensor([[pred_cog_cos]], device=self.device).float(),
                    }

                    candidates.append((
                        new_score,
                        grids + [new_grid_idx],
                        sogs + [pred_sog],
                        cogs + [pred_cog],
                        new_state
                    ))

            # 상위 beam_width개 유지
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        # 결과 변환
        results = []
        for score, grids, sogs, cogs, _ in beams:
            preds = []
            cur_time = last_time

            for grid_idx, sog, cog in zip(grids, sogs, cogs):
                cur_time = cur_time + pd.Timedelta(minutes=1)
                grid_str = self.grid_vocab.decode(grid_idx)
                lat, lon = grid_id_to_latlon(grid_str)

                if lat is None:
                    lat = inputs['last_lat']
                    lon = inputs['last_lon']

                preds.append({
                    'datetime': cur_time,
                    'lat': lat,
                    'lon': lon,
                    'grid_id': grid_str,
                    'sog': sog,
                    'cog': cog,
                })

            results.append({
                'score': score,
                'predictions': pd.DataFrame(preds)
            })

        return results


# ============================================
# 테스트 (10노트 이상 선박 10개)
# ============================================

if __name__ == "__main__":
    import sys
    from glob import glob
    import random

    print("=" * 70)
    print("GridTrajectoryInferenceV4 - 10노트 이상 선박 10개 테스트")
    print("=" * 70)

    # 모델 경로
    model_path = "model/yeosu/model_v4/model_v4_best.pth"
    vocab_path = "prepared_data_v4/grid_vocab.json"
    scaler_path = "model/yeosu/model_v4/scaler_v4.npz"

    # 모델 로드
    print("\n[1] 모델 로드...")
    try:
        inferencer = GridTrajectoryInferenceV4(
            model_path=model_path,
            vocab_path=vocab_path,
            scaler_path=scaler_path
        )
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        sys.exit(1)

    # 테스트 데이터 검색
    print("\n[2] 10노트 이상 선박 검색...")
    data_folder = "G:/NIA_ai_project/항적데이터 추출/여수"
    csv_files = glob(os.path.join(data_folder, "*.csv"))
    print(f"  - 총 파일 수: {len(csv_files)}")

    # 10노트 이상 선박 찾기 (최대 10개)
    MIN_SOG = 10.0  # 최소 평균 속력 (노트)
    NUM_SHIPS = 10  # 테스트할 선박 수

    test_files = []
    for f in csv_files:
        try:
            df_temp = pd.read_csv(f, encoding='utf-8')
            avg_sog = df_temp['sog'].mean()
            if len(df_temp) >= 70 and avg_sog >= MIN_SOG:
                # 파일명에서 MMSI 추출
                filename = os.path.basename(f)
                parts = filename.replace('.csv', '').split('_')
                mmsi = parts[0] if len(parts) >= 1 else "unknown"

                test_files.append({
                    'path': f,
                    'mmsi': mmsi,
                    'avg_sog': avg_sog,
                    'n_points': len(df_temp),
                    'start_area': parts[1] if len(parts) >= 3 else "남쪽 진입",
                    'end_area': parts[2] if len(parts) >= 3 else "여수정박지B",
                })

                if len(test_files) >= NUM_SHIPS:
                    break
        except:
            continue

    print(f"  - 찾은 선박 수: {len(test_files)}")

    if len(test_files) == 0:
        print("[ERROR] 10노트 이상 선박을 찾을 수 없습니다.")
        sys.exit(1)

    # 각 선박별 테스트 결과 저장
    all_results = []
    colors = ['red', 'orange', 'purple', 'darkgreen', 'darkblue',
              'brown', 'pink', 'gray', 'olive', 'cyan']

    print("\n[3] 선박별 예측 수행...")
    print("=" * 100)
    print(f"{'No':>3} | {'MMSI':>12} | {'평균SOG':>8} | {'출발':>10} | {'도착':>12} | {'평균오차(m)':>10} | {'SOG오차':>8}")
    print("-" * 100)

    for idx, ship_info in enumerate(test_files):
        # 데이터 로드
        df = pd.read_csv(ship_info['path'], encoding='utf-8')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # 입력/실제 분리
        input_df = df.iloc[:50].copy()
        actual_future = df.iloc[50:60].copy()

        # 예측
        predictions = inferencer.predict(
            input_df,
            n_steps=10,
            start_area=ship_info['start_area'],
            end_area=ship_info['end_area']
        )

        # 오차 계산
        errors = []
        sog_errors = []
        for pred_row, actual_row in zip(predictions.itertuples(), actual_future.itertuples()):
            dlat = (pred_row.lat - actual_row.lat) * 111000
            dlon = (pred_row.lon - actual_row.lon) * 111000 * np.cos(np.radians(actual_row.lat))
            dist_error = np.sqrt(dlat**2 + dlon**2)
            errors.append(dist_error)
            sog_errors.append(abs(pred_row.sog - actual_row.sog))

        avg_error = np.mean(errors)
        avg_sog_error = np.mean(sog_errors)

        # 결과 저장 (전체 항적 포함)
        all_results.append({
            'mmsi': ship_info['mmsi'],
            'avg_sog': ship_info['avg_sog'],
            'start_area': ship_info['start_area'],
            'end_area': ship_info['end_area'],
            'avg_error': avg_error,
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'avg_sog_error': avg_sog_error,
            'input_df': input_df,
            'actual_future': actual_future,
            'predictions': predictions,
            'full_df': df,  # 전체 항적 저장
            'color': colors[idx % len(colors)],
        })

        print(f"{idx+1:>3} | {ship_info['mmsi']:>12} | {ship_info['avg_sog']:>7.1f}kn | "
              f"{ship_info['start_area'][:10]:>10} | {ship_info['end_area'][:12]:>12} | "
              f"{avg_error:>10.1f} | {avg_sog_error:>7.2f}kn")

    print("-" * 100)

    # 전체 통계
    print("\n[4] 전체 통계:")
    all_errors = [r['avg_error'] for r in all_results]
    all_sog_errors = [r['avg_sog_error'] for r in all_results]
    print(f"  - 전체 평균 거리 오차: {np.mean(all_errors):.1f} m")
    print(f"  - 전체 최대 거리 오차: {np.max([r['max_error'] for r in all_results]):.1f} m")
    print(f"  - 전체 최소 거리 오차: {np.min([r['min_error'] for r in all_results]):.1f} m")
    print(f"  - 전체 평균 SOG 오차: {np.mean(all_sog_errors):.2f} knots")

    # Folium 시각화 (모든 선박)
    try:
        import folium
        from folium import FeatureGroup

        print("\n[5] 지도 시각화 생성...")

        # 중심점 계산 (모든 선박의 중간)
        all_lats = []
        all_lons = []
        for r in all_results:
            all_lats.extend(r['input_df']['lat'].tolist())
            all_lons.extend(r['input_df']['lon'].tolist())
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # 각 선박별 레이어 추가
        for idx, result in enumerate(all_results):
            ship_name = f"[{idx+1}] {result['mmsi']} ({result['avg_sog']:.1f}kn, 오차:{result['avg_error']:.0f}m)"
            ship_group = FeatureGroup(name=ship_name)

            # 전체 파일 다시 로드하여 전체 경로 표시
            full_df = result.get('full_df', None)
            if full_df is not None:
                # 전체 경로 (회색, 얇은 선)
                full_coords = [[r['lat'], r['lon']] for _, r in full_df.iterrows()]
                folium.PolyLine(
                    full_coords, color='gray', weight=1, opacity=0.4,
                    popup=f"{result['mmsi']} 전체 항적"
                ).add_to(ship_group)

            # 입력 구간 경로 (파랑, 굵은 선)
            input_coords = [[r['lat'], r['lon']] for _, r in result['input_df'].iterrows()]
            folium.PolyLine(
                input_coords, color='blue', weight=3, opacity=0.8,
                popup=f"{result['mmsi']} 입력 구간 (50분)"
            ).add_to(ship_group)

            # 실제 미래 경로 (초록, 굵은 선)
            actual_coords = [[r['lat'], r['lon']] for _, r in result['actual_future'].iterrows()]
            folium.PolyLine(
                actual_coords, color='green', weight=3, opacity=0.8,
                popup=f"{result['mmsi']} 실제 미래 (10분)"
            ).add_to(ship_group)

            # 입력 시작점 (파랑 마커)
            folium.CircleMarker(
                [result['input_df']['lat'].iloc[0], result['input_df']['lon'].iloc[0]],
                radius=6, color='blue', fill=True, fillOpacity=0.9,
                popup=f"<b>{result['mmsi']}</b><br>입력 시작<br>{result['input_df']['datetime'].iloc[0]}"
            ).add_to(ship_group)

            # 예측 시작점 (빨간 큰 마커 + 선박 정보)
            start_lat = result['input_df']['lat'].iloc[-1]
            start_lon = result['input_df']['lon'].iloc[-1]
            start_time = result['input_df']['datetime'].iloc[-1]
            start_sog = result['input_df']['sog'].iloc[-1]
            start_cog = result['input_df']['cog'].iloc[-1]

            popup_html = f"""
            <div style="width:200px">
                <h4 style="margin:0;color:red;">선박 {idx+1}</h4>
                <hr style="margin:5px 0">
                <b>MMSI:</b> {result['mmsi']}<br>
                <b>출발:</b> {result['start_area']}<br>
                <b>도착:</b> {result['end_area']}<br>
                <hr style="margin:5px 0">
                <b>예측시작 시각:</b><br>{start_time}<br>
                <b>위치:</b> ({start_lat:.4f}, {start_lon:.4f})<br>
                <b>SOG:</b> {start_sog:.1f} kn<br>
                <b>COG:</b> {start_cog:.0f}°<br>
                <hr style="margin:5px 0">
                <b style="color:red;">평균오차: {result['avg_error']:.0f}m</b>
            </div>
            """
            folium.Marker(
                [start_lat, start_lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color='red', icon='ship', prefix='fa')
            ).add_to(ship_group)

            # 예측 경로 (빨강, 점선 스타일)
            pred_coords = [[r['lat'], r['lon']] for _, r in result['predictions'].iterrows()]
            folium.PolyLine(
                pred_coords, color='red', weight=3, opacity=0.8,
                dash_array='10,5',  # 점선
                popup=f"{result['mmsi']} V4 예측 (오차: {result['avg_error']:.0f}m)"
            ).add_to(ship_group)

            # 예측 포인트 (빨강 원)
            for i, row in result['predictions'].iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']],
                    radius=5, color='red', fill=True, fillOpacity=0.7,
                    popup=f"예측 Step {i+1}<br>SOG: {row['sog']:.1f}kn<br>COG: {row['cog']:.0f}°<br>신뢰도: {row['confidence']:.1%}"
                ).add_to(ship_group)

            # 실제 미래 포인트 (초록 원)
            for i, row in result['actual_future'].iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']],
                    radius=4, color='green', fill=True, fillOpacity=0.6,
                    popup=f"실제 Step {i-49}<br>SOG: {row['sog']:.1f}kn<br>COG: {row['cog']:.0f}°"
                ).add_to(ship_group)

            ship_group.add_to(m)

        # 범례 추가
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border: 2px solid grey;
                    border-radius: 5px; font-size: 12px;">
            <b>범례</b><br>
            <i style="background:blue;width:20px;height:3px;display:inline-block;"></i> 입력 경로 (50분)<br>
            <i style="background:green;width:20px;height:3px;display:inline-block;"></i> 실제 미래 (10분)<br>
            <i style="background:red;width:20px;height:3px;display:inline-block;border-style:dashed;"></i> V4 예측<br>
            <i style="background:gray;width:20px;height:1px;display:inline-block;"></i> 전체 항적
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # 레이어 컨트롤 추가
        folium.LayerControl().add_to(m)

        output_html = 'prediction_v4_multi_ships.html'
        m.save(output_html)
        print(f"  - 지도 저장: {output_html}")
        print("  - 파랑: 입력 경로, 초록: 실제 미래, 빨강(점선): V4 예측")
        print("  - 좌측 레이어 패널에서 선박별 표시/숨김 가능")

    except ImportError:
        print("\n[INFO] folium이 없어 지도 시각화 생략")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
