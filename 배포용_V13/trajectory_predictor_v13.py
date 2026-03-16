# -*- coding: utf-8 -*-
"""
V13 선박 항적 예측기 (수심 제약 + 과거 항적 방향 반영)
======================================================
V12 기반에 다음 기능 추가:
1. 수심 제약: depth >= 10m인 영역으로만 경로 생성
2. 과거 항적 방향: 기존 선박들의 항적 방향 반영

사용법:
    from trajectory_predictor_v13 import predict_trajectory_v13

    result = predict_trajectory_v13(
        input_data=input_df,
        region='울산',
        shiptype=80,
        length=150,
        device='cuda'
    )
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from collections import defaultdict

# ============================================
# 선종/길이 카테고리 정의 (V12 호환)
# ============================================

SHIPTYPE_NAMES = ['화물선', '여객선', '유조선', '예부선', '기타선']
LENGTH_NAMES = ['0-40m', '40-80m', '80-120m', '120-160m', '160-200m',
                '200-240m', '240-280m', '280-320m', '320m+']


def get_shiptype_category(shiptype_code):
    """
    선종 코드 → 카테고리 (5개)

    Args:
        shiptype_code: AIS 선종 코드 (정수)

    Returns:
        int: 카테고리 (0-4)
            0: 화물선 (70-79)
            1: 여객선 (60-69)
            2: 유조선 (80-89)
            3: 예부선 (31-32)
            4: 기타선
    """
    try:
        code = int(shiptype_code)
    except (ValueError, TypeError):
        return 4  # 기타선

    if 70 <= code <= 79:
        return 0  # 화물선
    elif 60 <= code <= 69:
        return 1  # 여객선
    elif 80 <= code <= 89:
        return 2  # 유조선
    elif 31 <= code <= 32:
        return 3  # 예부선
    else:
        return 4  # 기타선


def get_length_category(length_m):
    """
    선박 길이 → 카테고리 (9개, 40m 단위)

    Args:
        length_m: 선박 길이 (미터)

    Returns:
        int: 카테고리 (0-8)
    """
    try:
        length = int(length_m)
    except (ValueError, TypeError):
        return 0  # 기본값

    if length <= 0:
        return 0

    category = length // 40
    return min(category, 8)  # 최대 8 (320m+)


# ============================================
# V12 모델 정의 (독립 실행용)
# ============================================

class PositionalEncoding(nn.Module):
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


class TrajectoryTransformerV12(nn.Module):
    """V12 Inertia-Aware Transformer (선종/길이 정보 포함)"""
    def __init__(self, input_dim=5, hidden_dim=192, num_heads=6,
                 num_layers=4, dropout=0.1, pred_len=60,
                 num_shiptype=5, num_length=9, meta_embed_dim=16):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # 메타 정보 임베딩
        self.shiptype_embed = nn.Embedding(num_shiptype, meta_embed_dim)
        self.length_embed = nn.Embedding(num_length, meta_embed_dim)

        # 입력 프로젝션 (기본 5차원 + 메타 임베딩 2*16 = 37)
        total_input_dim = input_dim + meta_embed_dim * 2
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
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

    def forward(self, x, shiptype, length):
        """
        Args:
            x: (B, seq_len, 5) - [lat_rel, lon_rel, sog_norm, cog_sin, cog_cos]
            shiptype: (B,) - 선종 카테고리 (0-4)
            length: (B,) - 길이 카테고리 (0-8)
        Returns:
            (B, pred_len, 2) - 예측 좌표 델타
        """
        B, seq_len, _ = x.size()

        # 메타 임베딩
        shiptype_emb = self.shiptype_embed(shiptype)  # (B, meta_embed_dim)
        length_emb = self.length_embed(length)        # (B, meta_embed_dim)

        # 메타 정보를 각 시퀀스 스텝에 추가
        meta = torch.cat([shiptype_emb, length_emb], dim=-1)  # (B, meta_embed_dim*2)
        meta = meta.unsqueeze(1).expand(-1, seq_len, -1)      # (B, seq_len, meta_embed_dim*2)

        # 입력과 메타 결합
        x_with_meta = torch.cat([x, meta], dim=-1)  # (B, seq_len, 5 + meta_embed_dim*2)

        # Transformer 처리
        x_proj = self.input_proj(x_with_meta)
        x_enc = self.pos_encoding(x_proj)
        encoded = self.transformer(x_enc)
        out = self.output_mlp(encoded[:, -1, :])

        return out.view(B, self.pred_len, 2)


# ============================================
# V12 예측기 클래스 (독립 실행용)
# ============================================

class TrajectoryPredictorV12:
    """V12 선박 항적 예측기"""

    SEQ_LEN = 30   # 입력 시퀀스 길이 (분)
    PRED_LEN = 60  # 예측 시퀀스 길이 (분)

    # 지원 지역
    SUPPORTED_REGIONS = ['울산', '인천', '목포', '부산', '여수', '통영', '완도', '제주', '군산', '포항', '동해']

    def __init__(self, region='울산', model_dir=None, device='cuda'):
        """
        Args:
            region: 지역명
            model_dir: 모델 폴더 경로 (None이면 기본 경로 사용)
            device: 'cuda' 또는 'cpu'
        """
        self.region = region
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'

        # 모델 경로 설정
        if model_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "models", region)

        self.model_dir = model_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        """모델 로드"""
        model_path = os.path.join(self.model_dir, "model_best.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = TrajectoryTransformerV12(
            input_dim=5,
            hidden_dim=192,
            num_heads=6,
            num_layers=4,
            pred_len=self.PRED_LEN,
            num_shiptype=5,
            num_length=9,
            meta_embed_dim=16
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"[V12] 모델 로드 완료: {self.region} (device: {self.device})")

    def _normalize_input(self, input_data):
        """입력 데이터 정규화"""
        seq_len = input_data.shape[0]
        normalized = np.zeros((seq_len, 5), dtype=np.float32)

        normalized[:, 0] = (input_data[:, 0] - input_data[0, 0]) * 100
        normalized[:, 1] = (input_data[:, 1] - input_data[0, 1]) * 100
        normalized[:, 2] = input_data[:, 2] / 30.0

        cog_rad = np.radians(input_data[:, 3])
        normalized[:, 3] = np.sin(cog_rad)
        normalized[:, 4] = np.cos(cog_rad)

        return normalized

    def _interpolate_1min(self, df):
        """1분 단위 보간"""
        if len(df) < 2:
            return None

        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').drop_duplicates('datetime')

        start_time = df['datetime'].iloc[0]
        end_time = df['datetime'].iloc[-1]
        time_range = pd.date_range(start=start_time, end=end_time, freq='1min')

        if len(time_range) < 2:
            return None

        df['_t'] = (df['datetime'] - start_time).dt.total_seconds()
        target_t = (time_range - start_time).total_seconds().values

        cog_rad = np.radians(df['cog'].values)
        cog_sin = np.interp(target_t, df['_t'].values, np.sin(cog_rad))
        cog_cos = np.interp(target_t, df['_t'].values, np.cos(cog_rad))
        cog_interp = np.degrees(np.arctan2(cog_sin, cog_cos)) % 360

        return pd.DataFrame({
            'datetime': time_range,
            'lat': np.interp(target_t, df['_t'].values, df['lat'].values),
            'lon': np.interp(target_t, df['_t'].values, df['lon'].values),
            'sog': np.interp(target_t, df['_t'].values, df['sog'].values),
            'cog': cog_interp,
        })

    def predict(self, input_data, shiptype=None, length=None, interpolate=True):
        """항적 예측 실행"""
        if isinstance(input_data, pd.DataFrame):
            if interpolate:
                input_interp = self._interpolate_1min(input_data)
                if input_interp is None or len(input_interp) < self.SEQ_LEN:
                    raise ValueError(f"데이터가 부족합니다. 최소 {self.SEQ_LEN}분 데이터 필요")
                input_interp = input_interp.tail(self.SEQ_LEN).reset_index(drop=True)
                input_array = input_interp[['lat', 'lon', 'sog', 'cog']].values
            else:
                input_array = input_data[['lat', 'lon', 'sog', 'cog']].values[-self.SEQ_LEN:]

        elif isinstance(input_data, dict):
            input_array = np.column_stack([
                input_data['lat'], input_data['lon'],
                input_data['sog'], input_data['cog']
            ]).astype(np.float32)

        elif isinstance(input_data, np.ndarray):
            input_array = input_data.astype(np.float32)
        else:
            raise TypeError(f"지원하지 않는 입력 타입: {type(input_data)}")

        if len(input_array) < self.SEQ_LEN:
            raise ValueError(f"데이터가 부족합니다. 현재: {len(input_array)}, 필요: {self.SEQ_LEN}")

        input_array = input_array[-self.SEQ_LEN:]
        last_position = input_array[-1, :2].copy()

        shiptype_cat = get_shiptype_category(shiptype) if shiptype is not None else 4
        length_cat = get_length_category(length) if length is not None else 0

        normalized = self._normalize_input(input_array)
        input_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

        shiptype_tensor = torch.tensor([shiptype_cat], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor([length_cat], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_delta = self.model(input_tensor, shiptype_tensor, length_tensor)

        pred_delta_np = pred_delta.cpu().numpy()[0]
        pred_delta_real = pred_delta_np / 100.0

        predicted_coords = np.zeros((self.PRED_LEN, 2), dtype=np.float32)
        predicted_coords[:, 0] = last_position[0] + pred_delta_real[:, 0]
        predicted_coords[:, 1] = last_position[1] + pred_delta_real[:, 1]

        return {
            'predicted_coords': predicted_coords,
            'last_position': last_position,
            'shiptype_cat': shiptype_cat,
            'length_cat': length_cat,
            'shiptype_name': SHIPTYPE_NAMES[shiptype_cat],
            'length_name': LENGTH_NAMES[length_cat],
        }


# ============================================
# 지역별 좌표 범위 (여유 포함)
# ============================================

REGION_BOUNDS = {
    '울산': {
        'lat_min': 35.0, 'lat_max': 36.0,
        'lon_min': 129.0, 'lon_max': 130.0
    },
    '인천': {
        'lat_min': 36.5, 'lat_max': 37.8,
        'lon_min': 125.5, 'lon_max': 127.0
    },
    '목포': {
        'lat_min': 33.5, 'lat_max': 35.7,
        'lon_min': 125.2, 'lon_max': 126.8
    },
    '부산': {
        'lat_min': 34.8, 'lat_max': 35.5,
        'lon_min': 128.5, 'lon_max': 129.5
    },
}


# ============================================
# 수심 체크 클래스 (450m 고해상도 NC 파일 지원)
# ============================================

class DepthChecker:
    """
    수심 데이터 기반 항해 가능 영역 체크
    - 450m 해상도 NC 파일 (depth_450m_*.nc) 지원
    - 직접 격자 인덱싱으로 빠른 조회
    """

    MIN_DEPTH = 10.0  # 최소 수심 (m) - 이 값 이상이어야 항해 가능

    def __init__(self, depth_file=None, region=None):
        """
        Args:
            depth_file: 수심 NC 파일 경로 (None이면 자동 탐색)
            region: 지역명 (울산, 인천 등) - 메모리 최적화용
        """
        self.depth_file = depth_file
        self.region = region
        self.elevation = None  # 2D numpy array (lat, lon)
        self.lat_arr = None    # 1D array
        self.lon_arr = None    # 1D array
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        self.lat_res = None    # 위도 해상도
        self.lon_res = None    # 경도 해상도
        self.loaded = False

        self._find_and_load_depth_file()

    def _find_and_load_depth_file(self):
        """수심 파일 찾기 및 로드"""
        if self.depth_file and os.path.exists(self.depth_file):
            self._load_nc_file(self.depth_file)
            return

        # 자동 탐색 (스크립트 위치 기준)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        search_paths = [
            # 1. 스크립트와 같은 폴더
            os.path.join(script_dir, "depth_450m_s27.0_w119.0_e137.0.nc"),
            os.path.join(script_dir, "depth_450m.nc"),
            # 2. 상위 폴더 (기존 호환)
            os.path.join(parent_dir, "depth_450m_s27.0_w119.0_e137.0.nc"),
            os.path.join(parent_dir, "depth_450m.nc"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                self._load_nc_file(path)
                return

        print(f"[V13] 수심 NC 파일을 찾을 수 없습니다.")
        print(f"      검색 경로: {search_paths}")

    def _load_nc_file(self, filepath):
        """NC 파일 로드 (지역별 부분 로드)"""
        try:
            from netCDF4 import Dataset
        except ImportError:
            print("[V13] netCDF4 모듈이 없습니다. pip install netCDF4 실행 필요")
            return

        print(f"[V13] 수심 NC 파일 로드 중... (지역: {self.region or '전체'})")

        # Windows에서 한글 경로 문제 해결
        nc = None
        original_cwd = os.getcwd()

        # 방법 1: 작업 디렉토리 변경 후 파일명만으로 열기
        try:
            nc_dir = os.path.dirname(filepath)
            nc_filename = os.path.basename(filepath)
            if nc_dir:
                os.chdir(nc_dir)
            nc = Dataset(nc_filename, 'r')
        except OSError:
            pass
        finally:
            os.chdir(original_cwd)

        # 방법 2: 직접 경로로 시도
        if nc is None:
            try:
                nc = Dataset(filepath, 'r')
            except OSError as e:
                print(f"[V13] NC 파일 열기 실패: {e}")
                print(f"      파일 경로: {filepath}")
                return

        # 좌표 배열 로드
        full_lat = nc.variables['lat'][:]
        full_lon = nc.variables['lon'][:]

        # 지역별 필터링 (메모리 절약)
        if self.region and self.region in REGION_BOUNDS:
            bounds = REGION_BOUNDS[self.region]
            # 여유 범위 추가 (0.5도)
            lat_min = bounds['lat_min'] - 0.5
            lat_max = bounds['lat_max'] + 0.5
            lon_min = bounds['lon_min'] - 0.5
            lon_max = bounds['lon_max'] + 0.5

            # 인덱스 범위 계산
            lat_idx = np.where((full_lat >= lat_min) & (full_lat <= lat_max))[0]
            lon_idx = np.where((full_lon >= lon_min) & (full_lon <= lon_max))[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                print(f"[V13] 경고: {self.region} 지역에 해당하는 수심 데이터 없음")
                nc.close()
                return

            lat_start, lat_end = lat_idx[0], lat_idx[-1] + 1
            lon_start, lon_end = lon_idx[0], lon_idx[-1] + 1

            self.lat_arr = full_lat[lat_start:lat_end]
            self.lon_arr = full_lon[lon_start:lon_end]
            self.elevation = nc.variables['elevation'][lat_start:lat_end, lon_start:lon_end]
        else:
            self.lat_arr = full_lat[:]
            self.lon_arr = full_lon[:]
            self.elevation = nc.variables['elevation'][:, :]

        nc.close()

        # numpy array로 변환
        self.lat_arr = np.array(self.lat_arr)
        self.lon_arr = np.array(self.lon_arr)
        self.elevation = np.array(self.elevation, dtype=np.float32)

        # 범위 및 해상도 저장
        self.lat_min = float(self.lat_arr[0])
        self.lat_max = float(self.lat_arr[-1])
        self.lon_min = float(self.lon_arr[0])
        self.lon_max = float(self.lon_arr[-1])

        if len(self.lat_arr) > 1:
            self.lat_res = float(self.lat_arr[1] - self.lat_arr[0])
        if len(self.lon_arr) > 1:
            self.lon_res = float(self.lon_arr[1] - self.lon_arr[0])

        self.loaded = True

        # 통계 출력
        land_count = np.sum(self.elevation >= 0)
        shallow_count = np.sum((self.elevation < 0) & (self.elevation > -self.MIN_DEPTH))
        navigable_count = np.sum(self.elevation <= -self.MIN_DEPTH)
        total = self.elevation.size

        print(f"[V13] 수심 데이터 로드 완료:")
        print(f"      격자: {self.elevation.shape[0]} x {self.elevation.shape[1]} ({total:,}개)")
        print(f"      해상도: {abs(self.lat_res)*111*1000:.0f}m x {abs(self.lon_res)*111*1000:.0f}m")
        print(f"      위도: {self.lat_min:.3f} ~ {self.lat_max:.3f}")
        print(f"      경도: {self.lon_min:.3f} ~ {self.lon_max:.3f}")
        print(f"      육지: {land_count:,}개 ({100*land_count/total:.1f}%)")
        print(f"      얕은물(<{self.MIN_DEPTH}m): {shallow_count:,}개 ({100*shallow_count/total:.1f}%)")
        print(f"      항해가능(>={self.MIN_DEPTH}m): {navigable_count:,}개 ({100*navigable_count/total:.1f}%)")

    def _get_indices(self, lat, lon):
        """좌표를 격자 인덱스로 변환"""
        if not self.loaded:
            return None, None

        # 범위 체크
        if lat < self.lat_min or lat > self.lat_max:
            return None, None
        if lon < self.lon_min or lon > self.lon_max:
            return None, None

        # 인덱스 계산 (직접 계산 - 빠름)
        lat_idx = int(round((lat - self.lat_min) / self.lat_res))
        lon_idx = int(round((lon - self.lon_min) / self.lon_res))

        # 경계 체크
        lat_idx = max(0, min(lat_idx, len(self.lat_arr) - 1))
        lon_idx = max(0, min(lon_idx, len(self.lon_arr) - 1))

        return lat_idx, lon_idx

    def get_depth(self, lat, lon):
        """
        특정 좌표의 수심 반환

        Returns:
            float: 수심 (음수 = 바다, 양수 = 육지) 또는 None
        """
        lat_idx, lon_idx = self._get_indices(lat, lon)
        if lat_idx is None:
            return None

        elev = self.elevation[lat_idx, lon_idx]

        # 양수 = 육지 높이, 음수 = 수심
        # 반환: 음수로 통일 (수심), 육지는 양수 그대로
        return float(elev)

    def is_navigable(self, lat, lon, check_neighbors=True):
        """
        항해 가능 여부 확인

        - elevation >= 0: 육지 (항해 불가)
        - -MIN_DEPTH < elevation < 0: 얕은 물 (항해 불가)
        - elevation <= -MIN_DEPTH: 항해 가능

        Args:
            lat, lon: 좌표
            check_neighbors: True면 주변 격자도 확인 (육지 근접 감지)
        """
        if not self.loaded:
            return True  # 수심 데이터 없으면 통과

        lat_idx, lon_idx = self._get_indices(lat, lon)
        if lat_idx is None:
            return True  # 범위 밖은 통과

        elev = self.elevation[lat_idx, lon_idx]

        # 현재 위치 체크 (수심 10m 이상이어야 함)
        if elev > -self.MIN_DEPTH:  # 육지 또는 얕은 물
            return False

        # 주변 격자 확인 (육지 근접 감지)
        if check_neighbors:
            # 주변 3x3 격자 확인 (약 1.3km 범위)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = lat_idx + di
                    nj = lon_idx + dj
                    if 0 <= ni < len(self.lat_arr) and 0 <= nj < len(self.lon_arr):
                        if self.elevation[ni, nj] >= 0:  # 육지
                            return False

        return True

    def is_land(self, lat, lon):
        """육지 여부 확인 (elevation >= 0)"""
        if not self.loaded:
            return False

        lat_idx, lon_idx = self._get_indices(lat, lon)
        if lat_idx is None:
            return False

        return self.elevation[lat_idx, lon_idx] >= 0

    def find_nearest_navigable(self, lat, lon, max_search_radius=0.05):
        """
        가장 가까운 항해 가능 지점 찾기

        Args:
            lat, lon: 현재 좌표
            max_search_radius: 최대 검색 반경 (도, 기본 ~5km)

        Returns:
            (lat, lon) 또는 None
        """
        if not self.loaded:
            return (lat, lon)

        # 현재 위치가 항해 가능하면 그대로 반환
        if self.is_navigable(lat, lon, check_neighbors=False):
            return (lat, lon)

        lat_idx, lon_idx = self._get_indices(lat, lon)
        if lat_idx is None:
            return None

        # 나선형 검색 (가까운 곳부터)
        search_steps = int(max_search_radius / abs(self.lat_res)) + 1

        for radius in range(1, search_steps + 1):
            candidates = []

            # 정사각형 테두리 검색
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    # 테두리만 검색 (이전 반경은 이미 검색함)
                    if abs(di) != radius and abs(dj) != radius:
                        continue

                    ni = lat_idx + di
                    nj = lon_idx + dj

                    if 0 <= ni < len(self.lat_arr) and 0 <= nj < len(self.lon_arr):
                        if self.elevation[ni, nj] <= -self.MIN_DEPTH:
                            # 항해 가능 지점 발견
                            nav_lat = self.lat_arr[ni]
                            nav_lon = self.lon_arr[nj]
                            dist = np.sqrt((nav_lat - lat)**2 + (nav_lon - lon)**2)
                            candidates.append((dist, nav_lat, nav_lon))

            if candidates:
                # 가장 가까운 것 반환
                candidates.sort(key=lambda x: x[0])
                return (float(candidates[0][1]), float(candidates[0][2]))

        return None

    def get_depth_grid_for_display(self, region=None):
        """
        시각화용 수심 격자 데이터 반환 (육지 + 얕은 물만)

        Returns:
            list: [{'lat': ..., 'lon': ..., 'depth': ..., 'type': 'land'/'shallow'}, ...]
        """
        if not self.loaded:
            return []

        grids = []

        # 샘플링 (너무 많으면 HTML이 느려짐)
        # 450m 해상도는 너무 촘촘하므로 5~10배 샘플링
        sample_step = max(1, min(len(self.lat_arr), len(self.lon_arr)) // 200)

        for i in range(0, len(self.lat_arr), sample_step):
            for j in range(0, len(self.lon_arr), sample_step):
                elev = self.elevation[i, j]

                if elev >= 0:  # 육지
                    grids.append({
                        'lat': float(self.lat_arr[i]),
                        'lon': float(self.lon_arr[j]),
                        'depth': float(elev),
                        'type': 'land'
                    })
                elif elev > -self.MIN_DEPTH:  # 얕은 물
                    grids.append({
                        'lat': float(self.lat_arr[i]),
                        'lon': float(self.lon_arr[j]),
                        'depth': float(elev),
                        'type': 'shallow'
                    })

        return grids


# ============================================
# 과거 항적 밀도 그리드 클래스
# ============================================

class HistoricalTrackGrid:
    """
    과거 항적 데이터 기반 그리드별 밀도 분석

    핵심 원리:
    - 각 격자에 지나간 선박 수(밀도)를 저장
    - 예측 경로 보정 시 밀도가 높은 쪽(실제 항로)으로 유도
    """

    def __init__(self, grid_resolution=0.0001):
        """
        Args:
            grid_resolution: 그리드 해상도 (도), 기본 0.0001도 ≈ 11m
        """
        self.grid_resolution = grid_resolution
        self.count_grid = defaultdict(int)  # {(grid_lat, grid_lon): count}
        self.max_density = 1  # 정규화용 최대 밀도
        self.loaded = False

        # 레거시 호환 (기존 파일 로드용)
        self.mean_direction = {}
        self.direction_grid = defaultdict(list)

    def _get_grid_key(self, lat, lon):
        """좌표를 그리드 키로 변환"""
        grid_lat = round(lat / self.grid_resolution) * self.grid_resolution
        grid_lon = round(lon / self.grid_resolution) * self.grid_resolution
        return (grid_lat, grid_lon)

    def _get_line_grids(self, lat1, lon1, lat2, lon2):
        """
        두 점 사이의 선이 지나가는 모든 격자 반환 (Bresenham 알고리즘)

        Args:
            lat1, lon1: 시작점
            lat2, lon2: 끝점

        Returns:
            set: 선이 지나가는 모든 격자 키 집합
        """
        grids = set()

        # 격자 인덱스로 변환
        res = self.grid_resolution
        x0 = int(round(lon1 / res))
        y0 = int(round(lat1 / res))
        x1 = int(round(lon2 / res))
        y1 = int(round(lat2 / res))

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # 현재 격자 추가
            grid_lat = y0 * res
            grid_lon = x0 * res
            grids.add((grid_lat, grid_lon))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return grids

    def get_density(self, lat, lon):
        """
        특정 위치의 밀도(통과 선박 수) 반환

        Returns:
            int: 해당 격자의 통과 선박 수 (0이면 항적 없음)
        """
        grid_key = self._get_grid_key(lat, lon)
        return self.count_grid.get(grid_key, 0)

    def get_normalized_density(self, lat, lon):
        """
        정규화된 밀도 반환 (0~1)
        """
        density = self.get_density(lat, lon)
        if self.max_density > 0:
            return min(density / self.max_density, 1.0)
        return 0.0

    def get_density_gradient(self, lat, lon, shiptype_cat=None, length_cat=None):
        """
        주변 격자 밀도 기반 끌어당김 벡터 계산

        밀도가 높은 방향으로 향하는 벡터 반환

        Args:
            lat, lon: 좌표
            shiptype_cat: 선종 카테고리 (0-4) - 현재 미사용, 호환성용
            length_cat: 길이 카테고리 (0-8) - 현재 미사용, 호환성용

        Returns:
            (dlat, dlon, strength) or None
            - dlat, dlon: 끌어당김 방향 (정규화됨)
            - strength: 끌어당김 강도 (0~1)
        """
        if not self.loaded:
            return None

        res = self.grid_resolution
        current_density = self.get_density(lat, lon)

        # 주변 8방향 격자 밀도 수집
        neighbors = [
            (res, 0),      # 북
            (-res, 0),     # 남
            (0, res),      # 동
            (0, -res),     # 서
            (res, res),    # 북동
            (res, -res),   # 북서
            (-res, res),   # 남동
            (-res, -res),  # 남서
        ]

        # 밀도 가중 평균 방향 계산
        total_weight = 0
        weighted_dlat = 0
        weighted_dlon = 0

        for dlat, dlon in neighbors:
            neighbor_density = self.get_density(lat + dlat, lon + dlon)

            # 현재보다 밀도가 높은 쪽으로만 끌어당김
            if neighbor_density > current_density:
                weight = neighbor_density - current_density

                # 방향 정규화
                dist = np.sqrt(dlat**2 + dlon**2)
                weighted_dlat += (dlat / dist) * weight
                weighted_dlon += (dlon / dist) * weight
                total_weight += weight

        if total_weight < 1:  # 의미 있는 기울기 없음
            return None

        # 정규화
        magnitude = np.sqrt(weighted_dlat**2 + weighted_dlon**2)
        if magnitude < 1e-9:
            return None

        # 강도: 주변 밀도 차이에 비례 (최대 밀도 대비)
        strength = min(total_weight / (self.max_density + 1), 1.0)

        return (weighted_dlat / magnitude, weighted_dlon / magnitude, strength)

    def load_from_training_data(self, data_dir, region=None):
        """
        학습 데이터에서 과거 항적 밀도 로드

        Args:
            data_dir: 전처리된 데이터 폴더
            region: 지역명 (선택)
        """
        import pickle
        import glob

        if region:
            search_path = os.path.join(data_dir, region, "*.pkl")
        else:
            search_path = os.path.join(data_dir, "**", "*.pkl")

        pkl_files = glob.glob(search_path, recursive=True)

        if not pkl_files:
            print(f"[경고] 학습 데이터 없음: {search_path}")
            return

        print(f"[V13] 과거 항적 밀도 로드 중... ({len(pkl_files)}개 파일)")

        total_segments = 0
        total_grids = 0
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    sequences = pickle.load(f)

                for seq in sequences:
                    if 'input' in seq:
                        # input: (30, 4) - [lat, lon, sog, cog]
                        points = seq['input']
                        # 연속 점들을 선으로 연결하여 지나는 모든 격자 카운트
                        for i in range(len(points) - 1):
                            lat1, lon1 = points[i, 0], points[i, 1]
                            lat2, lon2 = points[i + 1, 0], points[i + 1, 1]
                            # 선이 지나는 모든 격자
                            line_grids = self._get_line_grids(lat1, lon1, lat2, lon2)
                            for grid_key in line_grids:
                                self.count_grid[grid_key] += 1
                                total_grids += 1
                            total_segments += 1
            except Exception as e:
                continue

        # 최대 밀도 계산
        self._update_max_density()
        self.loaded = True

        print(f"[V13] 과거 항적 밀도 로드 완료: {total_segments:,}개 선분, {len(self.count_grid):,}개 그리드")
        print(f"      총 격자 카운트: {total_grids:,}회, 최대 밀도: {self.max_density:,}회")

    def load_from_raw_data(self, data_dir, region=None, sample_rate=0.1):
        """
        원본 parquet/csv 파일에서 과거 항적 밀도 로드

        Args:
            data_dir: 원본 데이터 폴더 (학습데이터/)
            region: 지역명 (선택)
            sample_rate: 샘플링 비율 (메모리 절약)
        """
        import glob

        if region:
            search_dir = os.path.join(data_dir, region)
        else:
            search_dir = data_dir

        parquet_files = glob.glob(os.path.join(search_dir, "**", "*.parquet"), recursive=True)
        csv_files = glob.glob(os.path.join(search_dir, "**", "*.csv"), recursive=True)
        all_files = parquet_files + csv_files

        if not all_files:
            print(f"[경고] 원본 데이터 없음: {search_dir}")
            return

        # 샘플링
        np.random.seed(42)
        sampled_files = np.random.choice(all_files,
                                         size=max(1, int(len(all_files) * sample_rate)),
                                         replace=False)

        print(f"[V13] 원본 항적 밀도 로드 중... ({len(sampled_files)}/{len(all_files)}개 파일)")

        total_segments = 0
        total_grids = 0
        for fpath in sampled_files:
            try:
                if fpath.endswith('.parquet'):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath, encoding='utf-8')

                if 'lat' not in df.columns or 'lon' not in df.columns:
                    continue

                # 시간순 정렬 (datetime 컬럼이 있으면)
                if 'datetime' in df.columns:
                    df = df.sort_values('datetime')

                # MMSI별 그룹화 (있으면)
                if 'mmsi' in df.columns:
                    for mmsi, group in df.groupby('mmsi'):
                        coords = group[['lat', 'lon']].values
                        for i in range(len(coords) - 1):
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[i + 1]
                            line_grids = self._get_line_grids(lat1, lon1, lat2, lon2)
                            for grid_key in line_grids:
                                self.count_grid[grid_key] += 1
                                total_grids += 1
                            total_segments += 1
                else:
                    # MMSI 없으면 전체를 하나의 궤적으로
                    coords = df[['lat', 'lon']].values
                    for i in range(len(coords) - 1):
                        lat1, lon1 = coords[i]
                        lat2, lon2 = coords[i + 1]
                        line_grids = self._get_line_grids(lat1, lon1, lat2, lon2)
                        for grid_key in line_grids:
                            self.count_grid[grid_key] += 1
                            total_grids += 1
                        total_segments += 1

            except Exception as e:
                continue

        self._update_max_density()
        self.loaded = True

        print(f"[V13] 원본 항적 밀도 로드 완료: {total_segments:,}개 선분, {len(self.count_grid):,}개 그리드")
        print(f"      총 격자 카운트: {total_grids:,}회, 최대 밀도: {self.max_density:,}회")

    def _update_max_density(self):
        """최대 밀도 계산"""
        if self.count_grid:
            self.max_density = max(self.count_grid.values())
        else:
            self.max_density = 1

    def _compute_mean_directions(self):
        """레거시: 그리드별 평균 방향 계산 (원형 평균) - 호환성용"""
        for grid_key, cogs in self.direction_grid.items():
            if len(cogs) < 3:  # 최소 3개 이상
                continue

            # 원형 평균 계산
            cog_rad = np.radians(cogs)
            mean_sin = np.mean(np.sin(cog_rad))
            mean_cos = np.mean(np.cos(cog_rad))
            mean_cog = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

            # 방향 일관성 (원형 분산)
            r = np.sqrt(mean_sin**2 + mean_cos**2)  # 0~1, 1이면 방향이 일관됨

            if r > 0.3:  # 일관성 있는 방향만 저장
                self.mean_direction[grid_key] = {
                    'cog': mean_cog,
                    'consistency': r,
                    'count': len(cogs)
                }

    def get_preferred_direction(self, lat, lon):
        """
        특정 위치의 선호 이동 방향 반환

        Returns:
            dict: {'cog': float, 'consistency': float, 'count': int} 또는 None
        """
        if not self.loaded:
            return None

        grid_key = self._get_grid_key(lat, lon)
        return self.mean_direction.get(grid_key)

    def save(self, filepath):
        """밀도 그리드 데이터 저장"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'grid_resolution': self.grid_resolution,
                'count_grid': dict(self.count_grid),
                'max_density': self.max_density,
                # 레거시 호환용
                'mean_direction': self.mean_direction
            }, f)
        print(f"[V13] 항적 밀도 그리드 저장: {filepath}")
        print(f"      {len(self.count_grid):,}개 그리드, 최대 밀도: {self.max_density:,}")

    def load(self, filepath):
        """밀도 그리드 데이터 로드"""
        import pickle
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.grid_resolution = data.get('grid_resolution', 0.01)
        self.count_grid = defaultdict(int, data.get('count_grid', {}))

        # 최대 밀도 로드 또는 계산
        if 'max_density' in data:
            self.max_density = data['max_density']
        else:
            self._update_max_density()

        # 레거시 호환
        self.mean_direction = data.get('mean_direction', {})

        self.loaded = True

        print(f"[V13] 항적 밀도 그리드 로드: {len(self.count_grid):,}개 그리드")
        print(f"      최대 밀도: {self.max_density:,}회")
        return True


# ============================================
# 경로 보정 클래스
# ============================================

class PathCorrector:
    """
    예측 경로 보정 - 궤적 부드러움 최우선

    핵심 원칙:
    1. 궤적의 부드러움이 최우선 (점핑 절대 금지)
    2. 수심 제약은 부드럽게 궤적을 굽히는 방식으로만 적용
    3. 현재 진행 방향(관성)을 최대한 유지
    """

    def __init__(self, depth_checker=None, track_grid=None):
        self.depth_checker = depth_checker
        self.track_grid = track_grid

        # 보정 파라미터
        self.max_angle_change_deg = 6.0  # 한 스텝당 최대 방향 변화 (도) - 부드러움 우선
        self.smooth_window = 21  # 스무딩 윈도우 크기 (홀수, 클수록 더 부드러움)
        self.smooth_passes = 3  # 스무딩 반복 횟수

    def set_params(self, max_correction_km=None, max_angle_deg=None,
                   smooth_window=None, smooth_passes=None):
        """보정 파라미터 동적 조정"""
        if max_angle_deg is not None:
            self.max_angle_change_deg = max_angle_deg
        if smooth_window is not None:
            self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        if smooth_passes is not None:
            self.smooth_passes = smooth_passes

    def correct_path(self, predicted_coords, last_position, last_cog=None,
                     depth_weight=0.7, track_weight=0.3, smooth_factor=1.0,
                     shiptype_cat=None, length_cat=None):
        """
        예측 경로 보정 - 밀도 기반 + 육지 회피 + 스무딩

        원칙:
        1. 모델 예측을 기본으로 존중
        2. 밀도가 높은 쪽(실제 항로)으로 부드럽게 유도
        3. 육지 회피
        4. 부드러운 곡선으로 스무딩

        Args:
            predicted_coords: 예측된 좌표
            last_position: 마지막 위치
            last_cog: 마지막 침로
            depth_weight: 수심 보정 가중치
            track_weight: 밀도 보정 가중치
            smooth_factor: 스무딩 강도 (0~1, 높을수록 더 부드러움)
            shiptype_cat: 선종 카테고리 (0-4)
            length_cat: 길이 카테고리 (0-8)

        Returns:
            (N, 2) 보정된 좌표
        """
        corrected = predicted_coords.copy()

        # 1. 밀도 기반 보정 (실제 항로로 유도) - 선종/길이별
        if self.track_grid is not None and self.track_grid.loaded:
            corrected = self._apply_density_guidance(corrected, track_weight,
                                                     shiptype_cat, length_cat)

        # 2. 가우시안 스무딩 (급격한 꺾임 완화)
        corrected = self._smooth_path_gaussian(corrected, smooth_factor)

        # 3. 육지 회피 (필요한 경우에만)
        if self.depth_checker is not None and self.depth_checker.loaded:
            corrected = self._gentle_land_avoidance(corrected, last_position, last_cog)

        # 4. 육지 강제 이동 (육지에 있는 점을 가장 가까운 바다로 이동)
        if self.depth_checker is not None and self.depth_checker.loaded:
            corrected = self._force_off_land(corrected)

        # 5. 좌표 유효성 검사
        corrected = self._validate_coordinates(corrected, predicted_coords, last_position)

        # 6. 최종 스플라인 스무딩 (부드러운 곡선 생성)
        corrected = self._smooth_path_spline(corrected, smooth_factor)

        return corrected

    def _apply_density_guidance(self, coords, strength=0.3,
                                shiptype_cat=None, length_cat=None):
        """
        밀도 기반 경로 보정 - 밀도가 높은 쪽(실제 항로)으로 유도

        각 예측 점에서:
        - 주변 격자의 밀도를 확인 (선종/길이별)
        - 밀도가 높은 방향으로 살짝 끌어당김

        Args:
            coords: 좌표 배열
            strength: 보정 강도
            shiptype_cat: 선종 카테고리 (0-4)
            length_cat: 길이 카테고리 (0-8)
        """
        if not self.track_grid or not self.track_grid.loaded:
            return coords

        result = coords.copy()
        grid_res = self.track_grid.grid_resolution

        for i in range(len(coords)):
            lat, lon = coords[i]

            # 밀도 기울기 계산 (선종/길이별)
            gradient = self.track_grid.get_density_gradient(lat, lon,
                                                            shiptype_cat, length_cat)

            if gradient is None:
                continue

            dlat, dlon, grad_strength = gradient

            # 보정량 계산 (최대 격자 1칸의 절반까지만)
            max_shift = grid_res * 0.5
            shift_amount = max_shift * strength * grad_strength

            # 위치 조정 (밀도 높은 쪽으로)
            result[i, 0] = lat + dlat * shift_amount
            result[i, 1] = lon + dlon * shift_amount

        return result

    def _apply_track_grid_guidance(self, coords, last_position, last_cog, track_weight=0.5):
        """
        [레거시] COG 기반 경로 조정 - 호환성용

        해당 그리드에서 과거 선박들이 주로 이동한 방향으로 궤적을 유도
        """
        if not self.track_grid or not self.track_grid.loaded:
            return coords

        result = coords.copy()
        n_points = len(coords)

        # 1단계: 각 점에서의 과거 항적 방향 수집 (주변 그리드 평균)
        track_directions = []
        for i in range(n_points):
            lat, lon = coords[i]
            smoothed_dir = self._get_smoothed_track_direction(lat, lon)
            track_directions.append(smoothed_dir)

        # 2단계: 수집된 방향들을 추가로 스무딩 (궤적 수준에서)
        smoothed_track_dirs = self._smooth_directions(track_directions)

        # 3단계: 스무딩된 과거 항적 방향을 적용
        if last_cog is not None:
            prev_cog = last_cog
        else:
            dlat = coords[0, 0] - last_position[0]
            dlon = coords[0, 1] - last_position[1]
            if abs(dlat) > 1e-9 or abs(dlon) > 1e-9:
                prev_cog = np.degrees(np.arctan2(dlon, dlat)) % 360
            else:
                prev_cog = 0.0

        prev_pos = last_position.copy()

        for i in range(n_points):
            # 원본에서 이 점까지의 거리와 방향
            dlat = coords[i, 0] - prev_pos[0]
            dlon = coords[i, 1] - prev_pos[1]
            dist = np.sqrt(dlat**2 + dlon**2)

            if dist < 1e-9:
                result[i] = prev_pos.copy()
                continue

            original_cog = np.degrees(np.arctan2(dlon, dlat)) % 360

            track_dir = smoothed_track_dirs[i]

            if track_dir is not None:
                track_cog = track_dir['cog']
                consistency = track_dir['consistency']

                # 일관성에 따라 가중치 조정 (더 보수적으로)
                effective_weight = track_weight * consistency * 0.7

                # 현재 방향과 과거 항적 방향의 차이
                angle_diff_track = ((track_cog - prev_cog + 180) % 360) - 180

                # 원본 방향과의 차이
                angle_diff_original = ((original_cog - prev_cog + 180) % 360) - 180

                # 두 방향을 블렌딩
                blended_diff = angle_diff_original * (1 - effective_weight) + angle_diff_track * effective_weight

                # 최대 방향 변화 제한 (스텝당 8도로 줄임)
                max_change = 8.0
                if abs(blended_diff) > max_change:
                    blended_diff = max_change * np.sign(blended_diff)

                new_cog = (prev_cog + blended_diff) % 360
            else:
                # 과거 항적 없으면 원본 방향 유지
                angle_diff = ((original_cog - prev_cog + 180) % 360) - 180
                max_change = 6.0
                if abs(angle_diff) > max_change:
                    angle_diff = max_change * np.sign(angle_diff)
                new_cog = (prev_cog + angle_diff) % 360

            new_cog_rad = np.radians(new_cog)

            # 새 위치 계산
            result[i, 0] = prev_pos[0] + dist * np.cos(new_cog_rad)
            result[i, 1] = prev_pos[1] + dist * np.sin(new_cog_rad)

            prev_pos = result[i].copy()
            prev_cog = new_cog

        return result

    def _get_smoothed_track_direction(self, lat, lon):
        """주변 그리드들의 방향을 평균하여 부드러운 방향 반환"""
        if not self.track_grid or not self.track_grid.loaded:
            return None

        # 현재 위치와 주변 8방향 그리드 확인
        grid_res = self.track_grid.grid_resolution
        offsets = [
            (0, 0),  # 현재 위치
            (grid_res, 0), (-grid_res, 0),
            (0, grid_res), (0, -grid_res),
            (grid_res, grid_res), (grid_res, -grid_res),
            (-grid_res, grid_res), (-grid_res, -grid_res)
        ]

        cogs = []
        weights = []

        for dlat, dlon in offsets:
            info = self.track_grid.get_preferred_direction(lat + dlat, lon + dlon)
            if info is not None:
                cogs.append(info['cog'])
                # 중심에 가까울수록 높은 가중치, 일관성도 반영
                dist_weight = 1.0 if dlat == 0 and dlon == 0 else 0.5
                weights.append(dist_weight * info['consistency'])

        if not cogs:
            return None

        # 원형 평균 계산
        weights = np.array(weights)
        weights = weights / weights.sum()

        cog_rad = np.radians(cogs)
        mean_sin = np.sum(np.sin(cog_rad) * weights)
        mean_cos = np.sum(np.cos(cog_rad) * weights)
        mean_cog = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

        # 평균 일관성
        avg_consistency = np.mean([w for w in weights]) * len(weights) / 9

        return {
            'cog': mean_cog,
            'consistency': min(avg_consistency, 1.0)
        }

    def _smooth_directions(self, track_directions):
        """궤적 수준에서 방향들을 스무딩"""
        n = len(track_directions)
        if n < 3:
            return track_directions

        smoothed = track_directions.copy()
        window = 5  # 앞뒤 2개씩 고려

        for i in range(n):
            # 윈도우 범위
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)

            # 유효한 방향들 수집
            valid_dirs = [track_directions[j] for j in range(start, end) if track_directions[j] is not None]

            if not valid_dirs:
                continue

            # 원형 평균 계산
            cogs = [d['cog'] for d in valid_dirs]
            consistencies = [d['consistency'] for d in valid_dirs]

            cog_rad = np.radians(cogs)
            mean_sin = np.mean(np.sin(cog_rad))
            mean_cos = np.mean(np.cos(cog_rad))
            mean_cog = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

            smoothed[i] = {
                'cog': mean_cog,
                'consistency': np.mean(consistencies)
            }

        return smoothed

    def _enforce_direction_continuity(self, coords, last_position, last_cog):
        """
        방향 연속성 강제 - 궤적이 부드럽게 이어지도록
        각 점이 이전 점으로부터 자연스러운 방향으로만 이동
        """
        if len(coords) < 2:
            return coords

        result = coords.copy()

        # 초기 방향 설정
        if last_cog is not None:
            prev_cog = last_cog
        else:
            dlat = coords[0, 0] - last_position[0]
            dlon = coords[0, 1] - last_position[1]
            if abs(dlat) > 1e-9 or abs(dlon) > 1e-9:
                prev_cog = np.degrees(np.arctan2(dlon, dlat)) % 360
            else:
                prev_cog = 0.0

        prev_pos = last_position.copy()

        for i in range(len(coords)):
            # 원본에서 이 점까지의 거리 계산
            dlat = coords[i, 0] - prev_pos[0]
            dlon = coords[i, 1] - prev_pos[1]
            dist = np.sqrt(dlat**2 + dlon**2)

            if dist < 1e-9:
                result[i] = prev_pos.copy()
                continue

            # 원본의 방향
            original_cog = np.degrees(np.arctan2(dlon, dlat)) % 360

            # 이전 방향과의 차이
            angle_diff = ((original_cog - prev_cog + 180) % 360) - 180

            # 최대 방향 변화 제한
            if abs(angle_diff) > self.max_angle_change_deg:
                angle_diff = self.max_angle_change_deg * np.sign(angle_diff)

            # 새 방향 계산
            new_cog = (prev_cog + angle_diff) % 360
            new_cog_rad = np.radians(new_cog)

            # 새 위치 계산
            result[i, 0] = prev_pos[0] + dist * np.cos(new_cog_rad)
            result[i, 1] = prev_pos[1] + dist * np.sin(new_cog_rad)

            prev_pos = result[i].copy()
            prev_cog = new_cog

        return result

    def _gentle_land_avoidance(self, coords, last_position, last_cog):
        """
        부드러운 육지 회피 - 실제 육지(elevation >= 0)일 때만 회피

        주의: 얕은 물(수심 < 10m)은 회피하지 않음 - 실제로 항해 가능한 경우가 많음
        """
        if not self.depth_checker or not self.depth_checker.loaded:
            return coords

        n_points = len(coords)

        # 실제 육지에 닿는 점 찾기 (is_land 사용 - elevation >= 0)
        land_indices = []
        for i in range(n_points):
            if self.depth_checker.is_land(coords[i, 0], coords[i, 1]):
                land_indices.append(i)

        if not land_indices:
            return coords  # 육지 없음

        # 첫 번째 육지 접촉 지점
        first_land_idx = land_indices[0]

        if first_land_idx <= 1:
            # 시작부터 육지면 전체 회전 시도
            return self._rotate_to_avoid_land(coords, last_position, last_cog)

        # 육지 접촉 전까지는 유지, 이후부터 점진적으로 굽힘
        result = coords.copy()

        # 육지 직전 방향 계산
        safe_idx = max(0, first_land_idx - 2)
        if safe_idx > 0:
            dlat = coords[safe_idx, 0] - coords[safe_idx - 1, 0]
            dlon = coords[safe_idx, 1] - coords[safe_idx - 1, 1]
        else:
            dlat = coords[safe_idx, 0] - last_position[0]
            dlon = coords[safe_idx, 1] - last_position[1]

        if abs(dlat) < 1e-9 and abs(dlon) < 1e-9:
            return coords

        current_cog = np.degrees(np.arctan2(dlon, dlat)) % 360

        # 회피 방향 결정 (좌/우 중 더 나은 쪽)
        best_offset = 0
        best_score = -1

        for test_offset in [10, -10, 20, -20, 30, -30]:
            test_cog = (current_cog + test_offset) % 360
            test_cog_rad = np.radians(test_cog)

            # 테스트 점 계산 (육지 접촉 지점에서)
            test_lat = coords[safe_idx, 0] + 0.05 * np.cos(test_cog_rad)
            test_lon = coords[safe_idx, 1] + 0.05 * np.sin(test_cog_rad)

            # 실제 육지가 아니면 OK (is_land 사용)
            if not self.depth_checker.is_land(test_lat, test_lon):
                score = 1.0 / (abs(test_offset) + 1)  # 작은 각도 선호
                if score > best_score:
                    best_score = score
                    best_offset = test_offset

        if best_offset == 0:
            return coords  # 회피 방향 찾지 못함

        # 육지 접촉 지점부터 점진적으로 방향 수정
        prev_pos = result[safe_idx].copy()
        prev_cog = current_cog

        for i in range(safe_idx + 1, n_points):
            # 원본 거리
            dlat = coords[i, 0] - coords[i - 1, 0]
            dlon = coords[i, 1] - coords[i - 1, 1]
            dist = np.sqrt(dlat**2 + dlon**2)

            # 점진적으로 회피 방향으로 틀기
            progress = (i - safe_idx) / (n_points - safe_idx)
            angle_adjust = best_offset * min(progress * 2, 1.0)  # 최대 best_offset까지

            new_cog = (prev_cog + angle_adjust * 0.3) % 360  # 매우 점진적
            new_cog_rad = np.radians(new_cog)

            result[i, 0] = prev_pos[0] + dist * np.cos(new_cog_rad)
            result[i, 1] = prev_pos[1] + dist * np.sin(new_cog_rad)

            prev_pos = result[i].copy()
            prev_cog = new_cog

        return result

    def _rotate_to_avoid_land(self, coords, last_position, last_cog):
        """시작부터 육지인 경우 전체 궤적을 부드럽게 회전"""
        if last_cog is None:
            dlat = coords[0, 0] - last_position[0]
            dlon = coords[0, 1] - last_position[1]
            if abs(dlat) > 1e-9 or abs(dlon) > 1e-9:
                last_cog = np.degrees(np.arctan2(dlon, dlat)) % 360
            else:
                return coords

        # 여러 각도로 회전 시도
        best_coords = coords
        best_score = 0

        for angle_offset in [15, -15, 30, -30, 45, -45, 60, -60]:
            rotated = self._rotate_trajectory(coords, last_position, angle_offset)

            # 육지가 아닌 점 개수 계산 (is_land 사용)
            non_land_count = 0
            for i in range(0, len(rotated), 5):
                if not self.depth_checker.is_land(rotated[i, 0], rotated[i, 1]):
                    non_land_count += 1

            # 작은 각도 보너스
            score = non_land_count - abs(angle_offset) / 60.0

            if score > best_score:
                best_score = score
                best_coords = rotated

        return best_coords

    def _rotate_trajectory(self, coords, center, offset_deg):
        """궤적을 중심점 기준으로 회전"""
        rotated = coords.copy()
        offset_rad = np.radians(offset_deg)

        for i in range(len(coords)):
            # 중심으로부터의 상대 위치
            dlat = coords[i, 0] - center[0]
            dlon = coords[i, 1] - center[1]

            # 회전 변환 (위도/경도 비율 보정)
            cos_lat = np.cos(np.radians(center[0]))
            dlon_normalized = dlon * cos_lat

            # 회전
            new_dlat = dlat * np.cos(offset_rad) - dlon_normalized * np.sin(offset_rad)
            new_dlon = (dlat * np.sin(offset_rad) + dlon_normalized * np.cos(offset_rad)) / cos_lat

            rotated[i, 0] = center[0] + new_dlat
            rotated[i, 1] = center[1] + new_dlon

        return rotated

    def _force_off_land(self, coords):
        """
        육지에 있는 점을 강제로 가장 가까운 바다로 이동

        이 함수는 모든 보정 후 마지막에 실행되어
        여전히 육지에 있는 점을 강제로 이동시킴
        """
        if not self.depth_checker or not self.depth_checker.loaded:
            return coords

        result = coords.copy()

        for i in range(len(coords)):
            lat, lon = coords[i, 0], coords[i, 1]

            # 육지인 경우 가장 가까운 항해 가능 지점으로 이동
            if self.depth_checker.is_land(lat, lon):
                nearest = self.depth_checker.find_nearest_navigable(lat, lon, max_search_radius=0.1)
                if nearest is not None:
                    result[i, 0] = nearest[0]
                    result[i, 1] = nearest[1]

        return result

    def _validate_coordinates(self, corrected, original, last_position):
        """좌표 유효성 검사 - 비정상 좌표는 원본으로 복원"""
        # 유효 범위 (한국 근해)
        LAT_MIN, LAT_MAX = 30.0, 40.0
        LON_MIN, LON_MAX = 120.0, 135.0

        # 최대 허용 이동 거리 (약 50km)
        MAX_DIST_DEG = 0.5

        result = corrected.copy()

        for i in range(len(corrected)):
            lat, lon = corrected[i]

            # 1. NaN/Inf 체크
            if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
                result[i] = original[i].copy()
                continue

            # 2. 범위 체크
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                result[i] = original[i].copy()
                continue

            # 3. 마지막 위치로부터 너무 멀리 이동했는지 체크
            dist_from_last = np.sqrt((lat - last_position[0])**2 + (lon - last_position[1])**2)
            if dist_from_last > MAX_DIST_DEG:
                result[i] = original[i].copy()
                continue

        return result

    def _smooth_path_gaussian(self, coords, factor):
        """Gaussian 가중치 기반 경로 스무딩 (급격한 꺾임 완화)"""
        if len(coords) < 3:
            return coords

        smoothed = coords.copy()
        window = self.smooth_window
        half_window = window // 2

        # Gaussian 가중치 생성
        sigma = half_window / 2.0
        weights = np.array([np.exp(-(x - half_window)**2 / (2 * sigma**2))
                           for x in range(window)])
        weights = weights / weights.sum()

        # 시작점(i=0)은 고정, 나머지만 스무딩
        for i in range(1, len(coords)):
            # 윈도우 범위 계산
            start = max(0, i - half_window)
            end = min(len(coords), i + half_window + 1)

            # 실제 적용할 가중치 (가장자리 처리)
            w_start = half_window - (i - start)
            w_end = w_start + (end - start)
            local_weights = weights[w_start:w_end]
            local_weights = local_weights / local_weights.sum()

            # 가중치 적용한 평균
            weighted_lat = np.sum(coords[start:end, 0] * local_weights)
            weighted_lon = np.sum(coords[start:end, 1] * local_weights)

            # 원본과 스무딩 결과 블렌딩
            smoothed[i, 0] = coords[i, 0] * (1 - factor) + weighted_lat * factor
            smoothed[i, 1] = coords[i, 1] * (1 - factor) + weighted_lon * factor

        return smoothed

    def _smooth_path_spline(self, coords, smoothing_factor=0.5):
        """
        스플라인 기반 경로 스무딩 (부드러운 곡선 생성)

        Args:
            coords: (N, 2) 좌표 배열 [lat, lon]
            smoothing_factor: 스무딩 강도 (0~1, 높을수록 더 부드러움)

        Returns:
            (N, 2) 스무딩된 좌표
        """
        if len(coords) < 4:
            return coords

        try:
            # 시간 축 생성 (정규화된 0~1 구간)
            t = np.linspace(0, 1, len(coords))

            # 스무딩 강도를 스플라인 파라미터로 변환
            # smoothing_factor가 높을수록 s값이 커져서 더 부드러워짐
            s = smoothing_factor * len(coords) * 0.0001  # 스케일 조정

            # 위도와 경도 각각에 대해 스플라인 피팅
            lat_spline = UnivariateSpline(t, coords[:, 0], s=s, k=3)
            lon_spline = UnivariateSpline(t, coords[:, 1], s=s, k=3)

            # 스플라인으로 스무딩된 좌표 계산
            smoothed_lat = lat_spline(t)
            smoothed_lon = lon_spline(t)

            # 원본과 스무딩 결과 블렌딩 (시작점은 고정)
            result = coords.copy()
            result[1:, 0] = coords[1:, 0] * (1 - smoothing_factor) + smoothed_lat[1:] * smoothing_factor
            result[1:, 1] = coords[1:, 1] * (1 - smoothing_factor) + smoothed_lon[1:] * smoothing_factor
            # result[0]은 coords[0] 그대로 유지됨

            return result

        except Exception as e:
            # 스플라인 피팅 실패시 원본 반환
            return coords


# ============================================
# V13 예측기 클래스
# ============================================

class TrajectoryPredictorV13(TrajectoryPredictorV12):
    """V13 선박 항적 예측기 (수심 제약 + 과거 항적 방향)"""

    def __init__(self, region='울산', model_dir=None, device='cuda',
                 depth_file=None, track_grid_file=None, enable_correction=True):
        """
        Args:
            region: 지역명 ('울산', '인천')
            model_dir: 모델 폴더 경로
            device: 'cuda' 또는 'cpu'
            depth_file: 수심 CSV 파일 경로
            track_grid_file: 과거 항적 그리드 파일 경로
            enable_correction: 경로 보정 활성화 여부
        """
        super().__init__(region=region, model_dir=model_dir, device=device)

        self.enable_correction = enable_correction
        self.depth_checker = None
        self.track_grid = None
        self.path_corrector = None

        if enable_correction:
            self._init_corrector(depth_file, track_grid_file)

    def _init_corrector(self, depth_file, track_grid_file):
        """보정기 초기화"""
        # 수심 체커 초기화 (지역 필터링)
        self.depth_checker = DepthChecker(depth_file, region=self.region)

        # 과거 항적 그리드 초기화
        self.track_grid = HistoricalTrackGrid(grid_resolution=0.0001)

        # 저장된 그리드 파일이 있으면 로드
        if track_grid_file and os.path.exists(track_grid_file):
            self.track_grid.load(track_grid_file)
        else:
            # 기본 경로들 시도 (우선순위 순)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)

            search_paths = [
                # 1. models/{지역}/track_grid.pkl (권장)
                os.path.join(script_dir, 'models', self.region, 'track_grid.pkl'),
                # 2. 상위 폴더의 track_grid_{지역}.pkl (레거시)
                os.path.join(base_dir, f"track_grid_{self.region}.pkl"),
                # 3. 현재 폴더의 track_grid_{지역}.pkl
                os.path.join(script_dir, f"track_grid_{self.region}.pkl"),
            ]

            loaded = False
            for grid_path in search_paths:
                if os.path.exists(grid_path):
                    self.track_grid.load(grid_path)
                    loaded = True
                    break

            if not loaded:
                print(f"[V13] 과거 항적 그리드 없음. 다음 경로에서 검색:")
                for p in search_paths:
                    print(f"  - {p}")
                print(f"[V13] preprocess_all_regions_v12.py --track-grid-only 실행 필요")

        # 경로 보정기 초기화
        self.path_corrector = PathCorrector(self.depth_checker, self.track_grid)

    def build_track_grid(self, data_dir=None, save_path=None):
        """
        과거 항적 그리드 생성

        Args:
            data_dir: 학습데이터_전처리/v12/{지역}/ 폴더
            save_path: 저장 경로 (None이면 models/{지역}/track_grid.pkl)
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "학습데이터_전처리", "v12")

        self.track_grid = HistoricalTrackGrid(grid_resolution=0.0001)
        self.track_grid.load_from_training_data(data_dir, self.region)

        if save_path is None:
            # models/{지역}/track_grid.pkl에 저장 (권장)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, 'models', self.region)
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, 'track_grid.pkl')

        self.track_grid.save(save_path)

        # 경로 보정기 업데이트
        self.path_corrector = PathCorrector(self.depth_checker, self.track_grid)

    def predict(self, input_data, shiptype=None, length=None, interpolate=True,
                enable_correction=None):
        """
        항적 예측 실행 (수심/항적 보정 포함)

        Args:
            input_data: 과거 데이터
            shiptype: 선종 코드
            length: 선박 길이 (m)
            interpolate: 보간 적용 여부
            enable_correction: 경로 보정 여부 (None이면 인스턴스 설정 사용)

        Returns:
            dict: 예측 결과 + 보정 정보
        """
        # V12 예측 실행
        result = super().predict(input_data, shiptype=shiptype, length=length,
                                  interpolate=interpolate)

        # 보정 여부 결정
        do_correction = enable_correction if enable_correction is not None else self.enable_correction

        if do_correction and self.path_corrector is not None:
            # 마지막 COG 추출
            if isinstance(input_data, pd.DataFrame):
                last_cog = input_data['cog'].iloc[-1]
            elif isinstance(input_data, dict):
                last_cog = input_data['cog'][-1]
            elif isinstance(input_data, np.ndarray):
                last_cog = input_data[-1, 3]
            else:
                last_cog = None

            # 경로 보정
            original_coords = result['predicted_coords'].copy()
            corrected_coords = self.path_corrector.correct_path(
                result['predicted_coords'],
                result['last_position'],
                last_cog=last_cog
            )

            result['predicted_coords'] = corrected_coords
            result['original_coords'] = original_coords
            result['correction_applied'] = True

            # 보정 거리 계산
            correction_dist = np.sqrt(
                np.sum((corrected_coords - original_coords)**2, axis=1)
            )
            result['correction_distance_km'] = correction_dist * 111  # 대략적 km 변환
        else:
            result['correction_applied'] = False

        return result


# ============================================
# 간편 사용 함수
# ============================================

# 전역 예측기 캐시
_predictor_cache_v13 = {}


def predict_trajectory_v13(input_data, region='울산', shiptype=None, length=None,
                           device='cuda', model_dir=None, interpolate=True,
                           enable_correction=True, depth_file=None, track_grid_file=None):
    """
    V13 선박 항적 예측 (수심 제약 + 과거 항적 방향 반영)

    Args:
        input_data: 과거 30분 항적 데이터
        region: 지역명 ('울산', '인천')
        shiptype: 선종 코드 (선택)
        length: 선박 길이 (m, 선택)
        device: 'cuda' 또는 'cpu'
        model_dir: 모델 폴더 경로
        interpolate: 보간 적용 여부
        enable_correction: 수심/항적 보정 적용 여부
        depth_file: 수심 CSV 파일 경로
        track_grid_file: 과거 항적 그리드 파일 경로

    Returns:
        dict:
            - predicted_coords: (60, 2) 보정된 예측 좌표
            - original_coords: (60, 2) 보정 전 원본 좌표 (보정 시)
            - last_position: (2,) 마지막 입력 위치
            - correction_applied: 보정 적용 여부
            - correction_distance_km: 보정 거리 (km)
    """
    global _predictor_cache_v13

    cache_key = f"v13_{region}_{device}_{enable_correction}"

    if cache_key not in _predictor_cache_v13:
        _predictor_cache_v13[cache_key] = TrajectoryPredictorV13(
            region=region,
            model_dir=model_dir,
            device=device,
            depth_file=depth_file,
            track_grid_file=track_grid_file,
            enable_correction=enable_correction
        )

    predictor = _predictor_cache_v13[cache_key]

    return predictor.predict(input_data, shiptype=shiptype, length=length,
                             interpolate=interpolate)


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("V13 선박 항적 예측기")
    print("=" * 60)
    print("특징:")
    print("  1. 수심 제약: depth >= 10m 영역으로만 경로 생성")
    print("  2. 과거 항적: 기존 선박 항적 방향 반영")
    print()
    print("사용법:")
    print("  from trajectory_predictor_v13 import predict_trajectory_v13")
    print()
    print("  result = predict_trajectory_v13(")
    print("      input_data=df,")
    print("      region='울산',")
    print("      enable_correction=True")
    print("  )")
    print()
    print("과거 항적 그리드 생성:")
    print("  predictor = TrajectoryPredictorV13(region='울산')")
    print("  predictor.build_track_grid()")
    print()
