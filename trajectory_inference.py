import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class LSTMTrajectoryModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, output_dim=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def step_latlon_dead_reckoning(lat, lon, sog_kn, cog_deg, dt_minutes=1):
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

class TrajectoryInference:
    def __init__(self, model_path, scaler_path, seq_len=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = np.load(scaler_path, allow_pickle=True)

        self.x_mean = ckpt["x_mean"].astype(np.float32)
        self.x_std  = ckpt["x_std"].astype(np.float32)
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std  = ckpt["y_std"].astype(np.float32)

        self.feature_cols = [str(x) for x in ckpt["feature_cols"]]
        self.target_cols  = [str(x) for x in ckpt["target_cols"]]

        # scaler에서 seq_len과 step(dt_minutes) 로드
        if seq_len is not None:
            self.seq_len = int(seq_len)
        elif "seq_len" in ckpt:
            self.seq_len = int(ckpt["seq_len"])
        else:
            self.seq_len = 80  # 기본값

        # step = 학습 시 시간 간격 (분) -> 추론 시 dt_minutes로 사용
        if "step" in ckpt:
            self.dt_minutes = int(ckpt["step"])
        else:
            self.dt_minutes = 1  # 기본값

        # 항로 범위 로드 (육지 침범 방지용)
        if "lat_bounds" in ckpt and "lon_bounds" in ckpt:
            self.lat_bounds = tuple(ckpt["lat_bounds"])
            self.lon_bounds = tuple(ckpt["lon_bounds"])
            print(f"[Inference] 항로 범위: lat={self.lat_bounds[0]:.4f}~{self.lat_bounds[1]:.4f}, lon={self.lon_bounds[0]:.4f}~{self.lon_bounds[1]:.4f}")
        else:
            self.lat_bounds = None
            self.lon_bounds = None

        print(f"[Inference] seq_len={self.seq_len}, dt_minutes={self.dt_minutes}")

        self.model = LSTMTrajectoryModel(
            input_dim=len(self.feature_cols),
            output_dim=len(self.target_cols),
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _prepare_hist(self, df):
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
        return hist

    def _clamp_to_bounds(self, lat, lon, prev_lat, prev_lon):
        """
        예측 위치가 항로 범위를 벗어나면 경계로 클램핑하고,
        이전 위치 방향으로 약간 되돌림
        """
        if self.lat_bounds is None or self.lon_bounds is None:
            return lat, lon, False

        out_of_bounds = False
        clamped_lat = lat
        clamped_lon = lon

        # 위도 범위 체크
        if lat < self.lat_bounds[0]:
            clamped_lat = self.lat_bounds[0]
            out_of_bounds = True
        elif lat > self.lat_bounds[1]:
            clamped_lat = self.lat_bounds[1]
            out_of_bounds = True

        # 경도 범위 체크
        if lon < self.lon_bounds[0]:
            clamped_lon = self.lon_bounds[0]
            out_of_bounds = True
        elif lon > self.lon_bounds[1]:
            clamped_lon = self.lon_bounds[1]
            out_of_bounds = True

        if out_of_bounds:
            # 경계를 벗어나면 이전 위치와 경계 사이의 중간점으로 보정
            clamped_lat = (prev_lat + clamped_lat) / 2
            clamped_lon = (prev_lon + clamped_lon) / 2

        return clamped_lat, clamped_lon, out_of_bounds

    def predict_multi_from_df(self, df, n_steps=80, dt_minutes=None,
                              sog_clip=(0.0, 35.0),
                              use_model_latlon=False,
                              enforce_bounds=True):
        """
        기본(use_model_latlon=False):
        - pred_sog/pred_cog로 Dead Reckoning 위치 업데이트
        - 물리 일관성 보장(속력=10kn이면 10kn만큼 이동)

        use_model_latlon=True:
        - 모델 pred_lat/pred_lon 그대로 사용(기존 방식)

        dt_minutes: 예측 시간 간격 (분). None이면 학습 시 사용한 step 값 사용
        enforce_bounds: True면 항로 범위를 벗어나지 않도록 보정 (기본값: True)
        """
        # dt_minutes가 None이면 학습 시 step 값 사용
        if dt_minutes is None:
            dt_minutes = self.dt_minutes

        hist = self._prepare_hist(df)

        X = hist[self.feature_cols].values.astype(np.float32)
        Xn = (X - self.x_mean) / self.x_std

        cur_lat = float(hist["lat"].iloc[-1])
        cur_lon = float(hist["lon"].iloc[-1])
        last_time = hist["datetime"].iloc[-1]

        preds = []
        bound_violations = 0

        for _ in range(int(n_steps)):
            x_t = torch.from_numpy(Xn).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                y_hat_n = self.model(x_t).squeeze(0).cpu().numpy().astype(np.float32)

            y_hat = y_hat_n * self.y_std.squeeze(0) + self.y_mean.squeeze(0)
            pred_lat_m, pred_lon_m, pred_sog, pred_sin, pred_cos = y_hat.tolist()

            pred_cog = np.degrees(np.arctan2(pred_sin, pred_cos))
            pred_cog = (pred_cog + 360) % 360

            pred_sog = float(np.clip(pred_sog, sog_clip[0], sog_clip[1]))

            if use_model_latlon:
                next_lat, next_lon = float(pred_lat_m), float(pred_lon_m)
            else:
                next_lat, next_lon = step_latlon_dead_reckoning(cur_lat, cur_lon, pred_sog, pred_cog, dt_minutes=dt_minutes)

            # 항로 범위 제약 적용
            if enforce_bounds:
                next_lat, next_lon, violated = self._clamp_to_bounds(next_lat, next_lon, cur_lat, cur_lon)
                if violated:
                    bound_violations += 1

            last_time = last_time + pd.Timedelta(minutes=float(dt_minutes))
            preds.append([last_time, next_lat, next_lon, pred_sog, pred_cog])

            cur_lat, cur_lon = next_lat, next_lon

            new_row = np.array([
                cur_lat, cur_lon, pred_sog,
                np.sin(np.radians(pred_cog)),
                np.cos(np.radians(pred_cog)),
            ], dtype=np.float32)

            new_row_n = (new_row - self.x_mean.squeeze(0)) / self.x_std.squeeze(0)
            Xn = np.vstack([Xn[1:], new_row_n])

        if bound_violations > 0:
            print(f"[Inference] 경고: {bound_violations}회 항로 범위 이탈 보정됨")

        return pd.DataFrame(preds, columns=["datetime","pred_lat","pred_lon","pred_sog","pred_cog"])
