# -*- coding: utf-8 -*-
"""
선박 항적 예측 추론 V2
========================
Categorical 변수 및 격자 ID 지원
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTMTrajectoryModelV2(nn.Module):
    """V2 모델 (Embedding + LSTM)"""
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


def step_latlon_dead_reckoning(lat, lon, sog_kn, cog_deg, dt_minutes=1):
    """Dead Reckoning으로 다음 위치 계산"""
    v = float(sog_kn) * 0.514444
    d = v * float(dt_minutes) * 60.0
    brng = np.radians(float(cog_deg))

    R = 6371000.0
    lat1 = np.radians(float(lat))
    lon1 = np.radians(float(lon))

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d / R) +
                     np.cos(lat1) * np.sin(d / R) * np.cos(brng))
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d / R) * np.cos(lat1),
                             np.cos(d / R) - np.sin(lat1) * np.sin(lat2))

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)
    lon2 = (lon2 + 540) % 360 - 180
    return lat2, lon2


def compute_grid_id(lat, lon, lat_min, lon_min, grid_size, num_cols):
    """위경도를 격자 ID로 변환"""
    grid_row = int((lat - lat_min) / grid_size)
    grid_col = int((lon - lon_min) / grid_size)
    return grid_row * num_cols + grid_col


class TrajectoryInferenceV2:
    """V2 추론 클래스 (Categorical + Grid 지원)"""

    def __init__(self, model_path, scaler_path, seq_len=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = np.load(scaler_path, allow_pickle=True)

        # 정규화 파라미터
        self.x_mean = ckpt["x_mean"].astype(np.float32)
        self.x_std  = ckpt["x_std"].astype(np.float32)
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std  = ckpt["y_std"].astype(np.float32)

        # 피처 컬럼
        self.numeric_cols = [str(x) for x in ckpt["numeric_cols"]]
        self.target_cols  = [str(x) for x in ckpt["target_cols"]]

        # 시퀀스 정보
        if seq_len is not None:
            self.seq_len = int(seq_len)
        elif "seq_len" in ckpt:
            self.seq_len = int(ckpt["seq_len"])
        else:
            self.seq_len = 80

        # Categorical vocab 로드
        self.mmsi_vocab = eval(str(ckpt["mmsi_vocab"]))
        self.start_vocab = eval(str(ckpt["start_vocab"]))
        self.end_vocab = eval(str(ckpt["end_vocab"]))

        num_mmsi = int(ckpt["num_mmsi"])
        num_start_area = int(ckpt["num_start_area"])
        num_end_area = int(ckpt["num_end_area"])

        # 격자 정보
        self.grid_lat_min = float(ckpt["grid_info_lat_min"])
        self.grid_lon_min = float(ckpt["grid_info_lon_min"])
        self.grid_size = float(ckpt["grid_size"])
        self.num_rows = int(ckpt["num_rows"])
        self.num_cols = int(ckpt["num_cols"])
        self.total_grids = int(ckpt["total_grids"])

        embed_dim = int(ckpt["embed_dim"]) if "embed_dim" in ckpt else 16

        # 항로 범위
        if "lat_bounds" in ckpt and "lon_bounds" in ckpt:
            self.lat_bounds = tuple(ckpt["lat_bounds"])
            self.lon_bounds = tuple(ckpt["lon_bounds"])
            print(f"[InferenceV2] 항로 범위: lat={self.lat_bounds[0]:.4f}~{self.lat_bounds[1]:.4f}, lon={self.lon_bounds[0]:.4f}~{self.lon_bounds[1]:.4f}")
        else:
            self.lat_bounds = None
            self.lon_bounds = None

        print(f"[InferenceV2] seq_len={self.seq_len}, dt_minutes={self.dt_minutes}")
        print(f"[InferenceV2] 격자: {self.num_rows}x{self.num_cols} = {self.total_grids} (크기: {self.grid_size}도)")
        print(f"[InferenceV2] Categorical: mmsi={num_mmsi}, start_area={num_start_area}, end_area={num_end_area}")

        # 모델 로드
        self.model = LSTMTrajectoryModelV2(
            num_features=len(self.numeric_cols),
            output_dim=len(self.target_cols),
            num_mmsi=num_mmsi,
            num_start_area=num_start_area,
            num_end_area=num_end_area,
            num_grids=self.total_grids + 1,
            embed_dim=embed_dim,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _get_mmsi_id(self, mmsi):
        """MMSI를 ID로 변환"""
        return self.mmsi_vocab.get(mmsi, 0)

    def _get_start_area_id(self, area):
        """start_area를 ID로 변환"""
        return self.start_vocab.get(area, 0)

    def _get_end_area_id(self, area):
        """end_area를 ID로 변환"""
        return self.end_vocab.get(area, 0)

    def _get_grid_id(self, lat, lon):
        """위경도를 격자 ID로 변환"""
        grid_id = compute_grid_id(lat, lon, self.grid_lat_min, self.grid_lon_min,
                                   self.grid_size, self.num_cols)
        return np.clip(grid_id, 0, self.total_grids)

    def _prepare_hist(self, df, mmsi=None, start_area=None, end_area=None):
        """입력 데이터 준비"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ["lat","lon","sog","cog"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime","lat","lon","sog","cog"]).sort_values("datetime").reset_index(drop=True)

        if len(df) < self.seq_len:
            raise ValueError(f"rows({len(df)}) < seq_len({self.seq_len})")

        hist = df.iloc[-self.seq_len:].copy()
        hist["sin_cog"] = np.sin(np.radians(hist["cog"]))
        hist["cos_cog"] = np.cos(np.radians(hist["cog"]))

        # Categorical ID
        if mmsi is None:
            mmsi = hist["mmsi"].iloc[0] if "mmsi" in hist.columns else 0
        if start_area is None:
            start_area = hist["start_area"].iloc[0] if "start_area" in hist.columns else "unknown"
        if end_area is None:
            end_area = hist["end_area"].iloc[0] if "end_area" in hist.columns else "unknown"

        self._current_mmsi_id = self._get_mmsi_id(mmsi)
        self._current_start_id = self._get_start_area_id(start_area)
        self._current_end_id = self._get_end_area_id(end_area)

        return hist

    def _clamp_to_bounds(self, lat, lon, prev_lat, prev_lon):
        """항로 범위 벗어나면 보정"""
        if self.lat_bounds is None or self.lon_bounds is None:
            return lat, lon, False

        out_of_bounds = False
        clamped_lat = lat
        clamped_lon = lon

        if lat < self.lat_bounds[0]:
            clamped_lat = self.lat_bounds[0]
            out_of_bounds = True
        elif lat > self.lat_bounds[1]:
            clamped_lat = self.lat_bounds[1]
            out_of_bounds = True

        if lon < self.lon_bounds[0]:
            clamped_lon = self.lon_bounds[0]
            out_of_bounds = True
        elif lon > self.lon_bounds[1]:
            clamped_lon = self.lon_bounds[1]
            out_of_bounds = True

        if out_of_bounds:
            clamped_lat = (prev_lat + clamped_lat) / 2
            clamped_lon = (prev_lon + clamped_lon) / 2

        return clamped_lat, clamped_lon, out_of_bounds

    def predict_multi_from_df(self, df, n_steps=80,
                              mmsi=None, start_area=None, end_area=None,
                              sog_clip=(0.0, 35.0),
                              sog_min_ratio=0.7,
                              use_model_latlon=False,
                              enforce_bounds=True):
        """
        다중 스텝 예측 (1분 간격)

        Parameters:
        -----------
        df: 입력 DataFrame (datetime, lat, lon, sog, cog 필요, 1분 간격 보간 데이터)
        n_steps: 예측 스텝 수
        mmsi, start_area, end_area: Categorical 값 (None이면 df에서 추출)
        sog_clip: SOG 클리핑 범위 (min, max)
        sog_min_ratio: 입력 데이터 평균 SOG 대비 최소 비율 (0.7 = 70% 이상 유지)
        """

        hist = self._prepare_hist(df, mmsi, start_area, end_area)

        # 수치형 피처
        X_num = hist[["lat","lon","sog","sin_cog","cos_cog"]].values.astype(np.float32)
        Xn_num = (X_num - self.x_mean) / self.x_std

        # 입력 데이터의 평균 SOG를 기준으로 최소값 설정
        input_sog_mean = float(hist["sog"].mean())
        sog_min_threshold = input_sog_mean * sog_min_ratio

        # Categorical 피처 (시퀀스 전체에 동일 값)
        X_cat = np.zeros((self.seq_len, 4), dtype=np.int64)
        X_cat[:, 0] = self._current_mmsi_id
        X_cat[:, 1] = self._current_start_id
        X_cat[:, 2] = self._current_end_id

        # 격자 ID는 각 위치마다 다름
        for i in range(self.seq_len):
            X_cat[i, 3] = self._get_grid_id(hist["lat"].iloc[i], hist["lon"].iloc[i])

        cur_lat = float(hist["lat"].iloc[-1])
        cur_lon = float(hist["lon"].iloc[-1])
        last_time = hist["datetime"].iloc[-1]

        preds = []
        bound_violations = 0

        for _ in range(int(n_steps)):
            x_num_t = torch.from_numpy(Xn_num).unsqueeze(0).to(self.device)
            x_cat_t = torch.from_numpy(X_cat).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                y_hat_n = self.model(x_num_t, x_cat_t).squeeze(0).cpu().numpy().astype(np.float32)

            y_hat = y_hat_n * self.y_std.squeeze(0) + self.y_mean.squeeze(0)
            pred_lat_m, pred_lon_m, pred_sog, pred_sin, pred_cos = y_hat.tolist()

            pred_cog = np.degrees(np.arctan2(pred_sin, pred_cos))
            pred_cog = (pred_cog + 360) % 360

            # SOG 클리핑: 최소값은 입력 데이터 평균의 일정 비율 이상 유지
            effective_sog_min = max(sog_clip[0], sog_min_threshold)
            pred_sog = float(np.clip(pred_sog, effective_sog_min, sog_clip[1]))

            if use_model_latlon:
                next_lat, next_lon = float(pred_lat_m), float(pred_lon_m)
            else:
                next_lat, next_lon = step_latlon_dead_reckoning(cur_lat, cur_lon, pred_sog, pred_cog, dt_minutes=1)

            if enforce_bounds:
                next_lat, next_lon, violated = self._clamp_to_bounds(next_lat, next_lon, cur_lat, cur_lon)
                if violated:
                    bound_violations += 1

            last_time = last_time + pd.Timedelta(minutes=1)
            preds.append([last_time, next_lat, next_lon, pred_sog, pred_cog])

            cur_lat, cur_lon = next_lat, next_lon

            # 시퀀스 업데이트
            new_row_num = np.array([
                cur_lat, cur_lon, pred_sog,
                np.sin(np.radians(pred_cog)),
                np.cos(np.radians(pred_cog)),
            ], dtype=np.float32)

            new_row_num_n = (new_row_num - self.x_mean.squeeze(0)) / self.x_std.squeeze(0)
            Xn_num = np.vstack([Xn_num[1:], new_row_num_n])

            # Categorical 업데이트 (격자 ID만 변경)
            new_grid_id = self._get_grid_id(cur_lat, cur_lon)
            new_cat_row = np.array([self._current_mmsi_id, self._current_start_id,
                                     self._current_end_id, new_grid_id], dtype=np.int64)
            X_cat = np.vstack([X_cat[1:], new_cat_row])

        if bound_violations > 0:
            print(f"[InferenceV2] 경고: {bound_violations}회 항로 범위 이탈 보정됨")

        return pd.DataFrame(preds, columns=["datetime","pred_lat","pred_lon","pred_sog","pred_cog"])


if __name__ == "__main__":
    print("TrajectoryInferenceV2 - Categorical 변수 및 격자 ID 지원")
    print("사용 예시:")
    print('  inf = TrajectoryInferenceV2("global_model_v2/lstm_global_v2.pth", "global_model_v2/scaler_global_v2.npz")')
    print('  preds = inf.predict_multi_from_df(df, n_steps=30, mmsi="209110000", start_area="남쪽진입", end_area="여수정박지B")')
