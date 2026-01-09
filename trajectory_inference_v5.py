# -*- coding: utf-8 -*-
"""
선박 항적 예측 추론 V5 - LLM 스타일 격자 ID 시퀀스 예측
=======================================================
입력: 과거 격자 ID 시퀀스
출력: 다음 격자 ID (반복 예측)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os


class GridSequenceModel(nn.Module):
    """격자 ID 시퀀스 → 다음 격자 ID 예측 (Transformer 기반)"""
    def __init__(self,
                 num_grids,
                 embed_dim=64,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1):
        super().__init__()

        self.num_grids = num_grids
        self.embed_dim = embed_dim

        self.grid_embed = nn.Embedding(num_grids + 1, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
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
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits


class TrajectoryInferenceV5:
    """V5 추론 클래스 - MMSI별 모델 로드 및 예측"""

    def __init__(self, model_dir, mmsi, device=None):
        """
        Args:
            model_dir: 모델 저장 폴더 (예: global_model_v5)
            mmsi: 예측할 선박의 MMSI
            device: cuda 또는 cpu
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mmsi = str(mmsi)

        # MMSI별 모델 폴더
        mmsi_dir = os.path.join(model_dir, self.mmsi)

        model_path = os.path.join(mmsi_dir, "model.pth")
        config_path = os.path.join(mmsi_dir, "config.npz")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MMSI {mmsi}의 모델이 없습니다: {model_path}")

        # 설정 로드
        config = np.load(config_path, allow_pickle=True)

        self.seq_len = int(config["seq_len"])
        self.grid_size = float(config["grid_size"])
        self.lat_min = float(config["lat_min"])
        self.lon_min = float(config["lon_min"])
        self.lat_max = float(config["lat_max"])
        self.lon_max = float(config["lon_max"])
        self.num_rows = int(config["num_rows"])
        self.num_cols = int(config["num_cols"])
        self.total_grids = int(config["total_grids"])
        self.min_sog = float(config["min_sog"])

        # 격자 정보
        self.grid_info = {
            'lat_min': self.lat_min,
            'lon_min': self.lon_min,
            'lat_max': self.lat_max,
            'lon_max': self.lon_max,
            'grid_size': self.grid_size,
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'total_grids': self.total_grids,
        }

        print(f"[V5 Inference] MMSI: {mmsi}")
        print(f"[V5 Inference] seq_len={self.seq_len}, grid_size={self.grid_size}")
        print(f"[V5 Inference] 격자: {self.num_rows}x{self.num_cols} = {self.total_grids}")

        # 모델 로드
        self.model = GridSequenceModel(
            num_grids=self.total_grids,
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            dropout=0.0  # 추론 시 dropout 비활성화
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _latlon_to_grid_id(self, lat, lon):
        """위경도를 격자 ID로 변환"""
        grid_row = int((lat - self.lat_min) / self.grid_size)
        grid_col = int((lon - self.lon_min) / self.grid_size)
        grid_row = max(0, min(grid_row, self.num_rows - 1))
        grid_col = max(0, min(grid_col, self.num_cols - 1))
        return grid_row * self.num_cols + grid_col

    def _grid_id_to_latlon(self, grid_id):
        """격자 ID를 위경도 중심점으로 변환"""
        grid_row = grid_id // self.num_cols
        grid_col = grid_id % self.num_cols
        lat = self.lat_min + (grid_row + 0.5) * self.grid_size
        lon = self.lon_min + (grid_col + 0.5) * self.grid_size
        return lat, lon

    def _prepare_grid_sequence(self, df):
        """DataFrame에서 격자 ID 시퀀스 추출"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ["lat", "lon", "sog"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime", "lat", "lon"]).sort_values("datetime").reset_index(drop=True)

        if len(df) < self.seq_len:
            raise ValueError(f"데이터 부족: {len(df)} < {self.seq_len}")

        # 마지막 seq_len개 사용
        hist = df.iloc[-self.seq_len:].copy()

        # 격자 ID 계산
        grid_ids = []
        for _, row in hist.iterrows():
            gid = self._latlon_to_grid_id(row["lat"], row["lon"])
            grid_ids.append(gid)

        return np.array(grid_ids, dtype=np.int64), hist["datetime"].iloc[-1]

    def predict_multi_from_df(self, df, n_steps=50, dt_minutes=1,
                               top_k=1, temperature=1.0):
        """
        과거 항적으로부터 다중 스텝 예측

        Args:
            df: 과거 항적 DataFrame (datetime, lat, lon 필수)
            n_steps: 예측 스텝 수
            dt_minutes: 예측 시간 간격 (분)
            top_k: Top-K 샘플링 (1이면 greedy)
            temperature: 샘플링 온도

        Returns:
            예측 결과 DataFrame (datetime, pred_lat, pred_lon, pred_grid_id)
        """
        grid_ids, last_time = self._prepare_grid_sequence(df)

        preds = []
        current_seq = grid_ids.copy()

        for step in range(n_steps):
            # 모델 입력 준비
            x = torch.from_numpy(current_seq[-self.seq_len:]).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(x)  # (1, num_grids)

                if top_k == 1:
                    # Greedy 선택
                    next_grid_id = logits.argmax(dim=1).item()
                else:
                    # Top-K 샘플링
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=1)

                    # Top-K 필터링
                    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=1)
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)

                    # 샘플링
                    sample_idx = torch.multinomial(top_k_probs, num_samples=1)
                    next_grid_id = top_k_indices[0, sample_idx[0, 0]].item()

            # 격자 ID를 위경도로 변환
            pred_lat, pred_lon = self._grid_id_to_latlon(next_grid_id)

            # 시간 업데이트
            last_time = last_time + pd.Timedelta(minutes=dt_minutes)

            preds.append([last_time, pred_lat, pred_lon, next_grid_id])

            # 시퀀스 업데이트
            current_seq = np.append(current_seq, next_grid_id)

        return pd.DataFrame(preds, columns=["datetime", "pred_lat", "pred_lon", "pred_grid_id"])

    def predict_with_beam_search(self, df, n_steps=50, dt_minutes=1,
                                   beam_width=5):
        """
        빔 서치를 이용한 다중 경로 예측

        Args:
            df: 과거 항적 DataFrame
            n_steps: 예측 스텝 수
            dt_minutes: 예측 시간 간격
            beam_width: 빔 너비 (후보 경로 수)

        Returns:
            List of (경로 DataFrame, 확률) 튜플
        """
        grid_ids, last_time = self._prepare_grid_sequence(df)

        # 빔 초기화: (시퀀스, 누적 로그 확률)
        beams = [(grid_ids.copy(), 0.0)]

        for step in range(n_steps):
            all_candidates = []

            for seq, log_prob in beams:
                x = torch.from_numpy(seq[-self.seq_len:]).unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    logits = self.model(x)
                    log_probs = torch.log_softmax(logits, dim=1)

                # Top-K 후보 추출
                top_log_probs, top_indices = torch.topk(log_probs, k=beam_width, dim=1)

                for i in range(beam_width):
                    next_grid = top_indices[0, i].item()
                    next_log_prob = top_log_probs[0, i].item()

                    new_seq = np.append(seq, next_grid)
                    new_log_prob = log_prob + next_log_prob

                    all_candidates.append((new_seq, new_log_prob))

            # 확률 기준 상위 beam_width개 선택
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

        # 결과 변환
        results = []
        for seq, log_prob in beams:
            # 예측된 부분만 추출 (원본 시퀀스 이후)
            pred_grid_ids = seq[len(grid_ids):]

            preds = []
            current_time = last_time
            for gid in pred_grid_ids:
                current_time = current_time + pd.Timedelta(minutes=dt_minutes)
                lat, lon = self._grid_id_to_latlon(gid)
                preds.append([current_time, lat, lon, gid])

            pred_df = pd.DataFrame(preds, columns=["datetime", "pred_lat", "pred_lon", "pred_grid_id"])
            prob = np.exp(log_prob)  # 로그 확률을 확률로 변환

            results.append((pred_df, prob))

        return results


def get_available_mmsi(model_dir):
    """학습된 MMSI 목록 조회"""
    mmsi_list = []

    if not os.path.exists(model_dir):
        return mmsi_list

    for name in os.listdir(model_dir):
        mmsi_path = os.path.join(model_dir, name)
        if os.path.isdir(mmsi_path):
            model_file = os.path.join(mmsi_path, "model.pth")
            if os.path.exists(model_file):
                try:
                    mmsi_list.append(int(name))
                except ValueError:
                    continue

    return sorted(mmsi_list)


if __name__ == "__main__":
    print("trajectory_inference_v5.py - V5 LLM 스타일 추론")
    print()
    print("사용 예시:")
    print("  from trajectory_inference_v5 import TrajectoryInferenceV5, get_available_mmsi")
    print()
    print("  # 학습된 MMSI 확인")
    print("  mmsi_list = get_available_mmsi('global_model_v5')")
    print()
    print("  # 추론 객체 생성")
    print("  inferencer = TrajectoryInferenceV5('global_model_v5', mmsi=123456789)")
    print()
    print("  # 예측 (Greedy)")
    print("  pred_df = inferencer.predict_multi_from_df(history_df, n_steps=50)")
    print()
    print("  # 예측 (Top-K 샘플링)")
    print("  pred_df = inferencer.predict_multi_from_df(history_df, n_steps=50, top_k=5, temperature=0.8)")
    print()
    print("  # 예측 (빔 서치 - 다중 경로)")
    print("  results = inferencer.predict_with_beam_search(history_df, n_steps=50, beam_width=5)")
