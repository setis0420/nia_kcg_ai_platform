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
from collections import defaultdict

# V12 모델 임포트
from trajectory_predictor_v12 import (
    TrajectoryTransformerV12,
    TrajectoryPredictorV12,
    get_shiptype_category,
    get_length_category,
    SHIPTYPE_NAMES,
    LENGTH_NAMES
)


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
# 수심 체크 클래스
# ============================================

class DepthChecker:
    """수심 데이터 기반 항해 가능 영역 체크"""

    MIN_DEPTH = 10.0  # 최소 수심 (m)
    LAND_VALUE = -99999.0  # 육지 값

    def __init__(self, depth_file=None, region=None):
        """
        Args:
            depth_file: 수심 CSV 파일 경로
            region: 지역명 (울산, 인천 등) - 해당 지역만 로드
        """
        if depth_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            depth_file = os.path.join(base_dir, "depth_0.025deg_value.csv")

        self.depth_file = depth_file
        self.region = region
        self.grid_data = None
        self.kdtree = None
        self.coords = None
        self.depths = None
        self.grid_resolution = 0.025  # 그리드 해상도 (도)

        self._load_depth_data()

    def _load_depth_data(self):
        """수심 데이터 로드 (지역 필터링)"""
        if not os.path.exists(self.depth_file):
            print(f"[경고] 수심 파일 없음: {self.depth_file}")
            return

        print(f"[V13] 수심 데이터 로드 중... (지역: {self.region or '전체'})")
        df = pd.read_csv(self.depth_file)

        # 지역별 필터링
        if self.region and self.region in REGION_BOUNDS:
            bounds = REGION_BOUNDS[self.region]
            df = df[
                (df['cen_y'] >= bounds['lat_min']) & (df['cen_y'] <= bounds['lat_max']) &
                (df['cen_x'] >= bounds['lon_min']) & (df['cen_x'] <= bounds['lon_max'])
            ]
            print(f"[V13] {self.region} 지역 필터링: {len(df):,}개 그리드")

        # 좌표와 수심 추출
        self.coords = df[['cen_y', 'cen_x']].values  # (lat, lon)
        self.depths = df['depth'].values

        # KDTree 생성 (빠른 최근접 이웃 검색)
        self.kdtree = cKDTree(self.coords)

        # 항해 가능 영역 마스크 (depth >= 10m)
        self.navigable_mask = (self.depths >= self.MIN_DEPTH) & (self.depths != self.LAND_VALUE)

        print(f"[V13] 수심 데이터 로드 완료: {len(self.coords):,}개 그리드")
        print(f"      항해 가능 영역: {self.navigable_mask.sum():,}개 ({100*self.navigable_mask.mean():.1f}%)")

    def get_depth(self, lat, lon):
        """특정 좌표의 수심 반환"""
        if self.kdtree is None:
            return None

        _, idx = self.kdtree.query([lat, lon])
        return self.depths[idx]

    def is_navigable(self, lat, lon, check_neighbors=True):
        """
        항해 가능 여부 확인 (depth >= 10m)

        Args:
            lat, lon: 좌표
            check_neighbors: True면 주변 그리드도 확인 (육지 근접 감지)
        """
        if self.kdtree is None:
            return True  # 수심 데이터 없으면 통과

        _, idx = self.kdtree.query([lat, lon])

        # 가장 가까운 그리드가 육지이면 False
        if not self.navigable_mask[idx]:
            return False

        # 주변 그리드 확인 (육지 근접 감지)
        if check_neighbors:
            # 가까운 4개 그리드 확인
            distances, indices = self.kdtree.query([lat, lon], k=min(4, len(self.coords)))

            # 육지 그리드가 있으면서 매우 가까우면 (0.03도 = ~3km 이내) 항해 불가
            for dist, idx in zip(distances, indices):
                if dist < 0.03 and self.depths[idx] == self.LAND_VALUE:
                    return False

        return True

    def find_nearest_navigable(self, lat, lon, max_search_radius=0.1):
        """
        가장 가까운 항해 가능 지점 찾기

        Args:
            lat, lon: 현재 좌표
            max_search_radius: 최대 검색 반경 (도)

        Returns:
            (lat, lon) 또는 None
        """
        if self.kdtree is None:
            return (lat, lon)

        # 현재 위치가 항해 가능하면 그대로 반환
        _, idx = self.kdtree.query([lat, lon])
        if self.navigable_mask[idx]:
            return (lat, lon)

        # 반경 내 항해 가능 지점 검색
        indices = self.kdtree.query_ball_point([lat, lon], max_search_radius)

        if not indices:
            return None

        # 항해 가능한 지점 중 가장 가까운 것 찾기
        navigable_indices = [i for i in indices if self.navigable_mask[i]]

        if not navigable_indices:
            return None

        # 가장 가까운 항해 가능 지점
        distances = [np.sqrt((self.coords[i, 0] - lat)**2 + (self.coords[i, 1] - lon)**2)
                     for i in navigable_indices]
        nearest_idx = navigable_indices[np.argmin(distances)]

        return (self.coords[nearest_idx, 0], self.coords[nearest_idx, 1])


# ============================================
# 과거 항적 방향 분석 클래스
# ============================================

class HistoricalTrackGrid:
    """과거 항적 데이터 기반 그리드별 주요 이동 방향 분석"""

    def __init__(self, grid_resolution=0.01):
        """
        Args:
            grid_resolution: 그리드 해상도 (도), 기본 0.01도 ≈ 1km
        """
        self.grid_resolution = grid_resolution
        self.direction_grid = defaultdict(list)  # {(grid_lat, grid_lon): [cog1, cog2, ...]}
        self.count_grid = defaultdict(int)  # {(grid_lat, grid_lon): count}
        self.mean_direction = {}  # {(grid_lat, grid_lon): mean_cog}
        self.loaded = False

    def _get_grid_key(self, lat, lon):
        """좌표를 그리드 키로 변환"""
        grid_lat = round(lat / self.grid_resolution) * self.grid_resolution
        grid_lon = round(lon / self.grid_resolution) * self.grid_resolution
        return (grid_lat, grid_lon)

    def load_from_training_data(self, data_dir, region=None):
        """
        학습 데이터에서 과거 항적 방향 로드

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

        print(f"[V13] 과거 항적 데이터 로드 중... ({len(pkl_files)}개 파일)")

        total_points = 0
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    sequences = pickle.load(f)

                # 샘플링: 10%만 사용 (속도 향상)
                sample_size = max(1, len(sequences) // 10)
                sampled_seqs = sequences[::10][:sample_size * 3]

                for seq in sampled_seqs:
                    if 'input' in seq:
                        # input: (30, 4) - [lat, lon, sog, cog]
                        points = seq['input']
                        for i in range(len(points) - 1):
                            lat, lon = points[i, 0], points[i, 1]
                            # 이동 방향 계산
                            dlat = points[i+1, 0] - points[i, 0]
                            dlon = points[i+1, 1] - points[i, 1]
                            if abs(dlat) > 1e-6 or abs(dlon) > 1e-6:
                                cog = np.degrees(np.arctan2(dlon, dlat)) % 360
                                grid_key = self._get_grid_key(lat, lon)
                                self.direction_grid[grid_key].append(cog)
                                self.count_grid[grid_key] += 1
                                total_points += 1
            except Exception as e:
                continue

        # 그리드별 평균 방향 계산
        self._compute_mean_directions()
        self.loaded = True

        print(f"[V13] 과거 항적 로드 완료: {total_points:,}개 포인트, {len(self.mean_direction):,}개 그리드")

    def load_from_raw_data(self, data_dir, region=None, sample_rate=0.1):
        """
        원본 parquet/csv 파일에서 과거 항적 방향 로드

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

        print(f"[V13] 원본 항적 데이터 로드 중... ({len(sampled_files)}/{len(all_files)}개 파일)")

        total_points = 0
        for fpath in sampled_files:
            try:
                if fpath.endswith('.parquet'):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath, encoding='utf-8')

                if 'lat' not in df.columns or 'lon' not in df.columns:
                    continue

                df = df.sort_values('datetime').reset_index(drop=True)

                for i in range(len(df) - 1):
                    lat, lon = df.iloc[i]['lat'], df.iloc[i]['lon']
                    dlat = df.iloc[i+1]['lat'] - lat
                    dlon = df.iloc[i+1]['lon'] - lon

                    if abs(dlat) > 1e-6 or abs(dlon) > 1e-6:
                        cog = np.degrees(np.arctan2(dlon, dlat)) % 360
                        grid_key = self._get_grid_key(lat, lon)
                        self.direction_grid[grid_key].append(cog)
                        self.count_grid[grid_key] += 1
                        total_points += 1

            except Exception as e:
                continue

        self._compute_mean_directions()
        self.loaded = True

        print(f"[V13] 원본 항적 로드 완료: {total_points:,}개 포인트, {len(self.mean_direction):,}개 그리드")

    def _compute_mean_directions(self):
        """그리드별 평균 방향 계산 (원형 평균)"""
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
        """그리드 데이터 저장"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'grid_resolution': self.grid_resolution,
                'mean_direction': self.mean_direction,
                'count_grid': dict(self.count_grid)
            }, f)
        print(f"[V13] 항적 그리드 저장: {filepath}")

    def load(self, filepath):
        """그리드 데이터 로드"""
        import pickle
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.grid_resolution = data['grid_resolution']
        self.mean_direction = data['mean_direction']
        self.count_grid = defaultdict(int, data['count_grid'])
        self.loaded = True

        print(f"[V13] 항적 그리드 로드: {len(self.mean_direction):,}개 그리드")
        return True


# ============================================
# 경로 보정 클래스
# ============================================

class PathCorrector:
    """예측 경로 보정 (수심 제약 + 과거 항적 방향) - 급격한 꺾임 방지"""

    def __init__(self, depth_checker=None, track_grid=None):
        """
        Args:
            depth_checker: DepthChecker 인스턴스
            track_grid: HistoricalTrackGrid 인스턴스
        """
        self.depth_checker = depth_checker
        self.track_grid = track_grid

        # 보정 제한 파라미터 (급격한 꺾임 방지)
        self.max_correction_per_step_km = 0.5  # 한 스텝당 최대 보정 거리 (km) - 0.3->0.5 증가
        self.max_angle_change_deg = 20.0  # 한 스텝당 최대 방향 변화 (도) - 15->20 증가
        self.smooth_window = 7  # 스무딩 윈도우 크기 (홀수)
        self.smooth_passes = 2  # 스무딩 반복 횟수

    def set_params(self, max_correction_km=None, max_angle_deg=None,
                   smooth_window=None, smooth_passes=None):
        """보정 파라미터 동적 조정"""
        if max_correction_km is not None:
            self.max_correction_per_step_km = max_correction_km
        if max_angle_deg is not None:
            self.max_angle_change_deg = max_angle_deg
        if smooth_window is not None:
            self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        if smooth_passes is not None:
            self.smooth_passes = smooth_passes

    def correct_path(self, predicted_coords, last_position, last_cog=None,
                     depth_weight=0.7, track_weight=0.3, smooth_factor=0.6):
        """
        예측 경로 보정 (점진적 보정으로 급격한 꺾임 방지)

        Args:
            predicted_coords: (N, 2) 예측 좌표 [lat, lon]
            last_position: (2,) 마지막 입력 위치
            last_cog: 마지막 입력 COG (선택)
            depth_weight: 수심 제약 가중치
            track_weight: 과거 항적 가중치
            smooth_factor: 경로 스무딩 강도 (0~1)

        Returns:
            (N, 2) 보정된 좌표
        """
        corrected = predicted_coords.copy()
        n_points = len(corrected)

        # 0. 궤적 수준 육지 회피 (전체 경로가 육지로 향하면 방향 전환)
        if self.depth_checker is not None:
            corrected = self._avoid_land_trajectory(corrected, last_position, last_cog)

        # 1. 수심 기반 보정 (점진적)
        if self.depth_checker is not None:
            corrected = self._correct_depth_gradual(corrected, last_position)

        # 2. 과거 항적 방향 기반 보정 (점진적)
        if self.track_grid is not None and self.track_grid.loaded:
            corrected = self._correct_direction_gradual(corrected, last_position, last_cog, track_weight)

        # 3. 다중 패스 스무딩 (Gaussian-like)
        if smooth_factor > 0:
            for _ in range(self.smooth_passes):
                corrected = self._smooth_path_gaussian(corrected, smooth_factor)

        # 4. 최종 수심 체크 (부드러운 우회)
        if self.depth_checker is not None:
            corrected = self._final_depth_check_smooth(corrected, last_position)

        # 5. 방향 연속성 보정 (급격한 각도 변화 완화)
        corrected = self._ensure_direction_continuity(corrected, last_position, last_cog)

        # 6. 최종 좌표 유효성 검사 (폭발 방지)
        corrected = self._validate_coordinates(corrected, predicted_coords, last_position)

        return corrected

    def _avoid_land_trajectory(self, coords, last_position, last_cog):
        """
        궤적 수준 육지 회피
        - 전체 경로가 육지로 향하는지 감지
        - 육지로 향하면 방향을 틀어서 회피
        """
        if self.depth_checker is None or self.depth_checker.kdtree is None:
            return coords

        corrected = coords.copy()
        n_points = len(corrected)

        # 종점 및 중간점 육지 여부 확인
        end_point = corrected[-1]
        mid_point = corrected[n_points // 2]

        end_navigable = self.depth_checker.is_navigable(end_point[0], end_point[1])
        mid_navigable = self.depth_checker.is_navigable(mid_point[0], mid_point[1])

        # 종점 또는 중간점이 육지면 궤적 전체를 회전
        if not end_navigable or not mid_navigable:
            # 현재 진행 방향 계산
            if last_cog is not None:
                current_direction = last_cog
            else:
                dlat = corrected[0, 0] - last_position[0]
                dlon = corrected[0, 1] - last_position[1]
                current_direction = np.degrees(np.arctan2(dlon, dlat)) % 360

            # 여러 방향으로 회전 시도 (+30, -30, +60, -60, +90, -90도)
            best_coords = None
            best_score = -1

            for angle_offset in [30, -30, 60, -60, 90, -90, 120, -120]:
                rotated = self._rotate_trajectory(
                    coords, last_position, current_direction, angle_offset
                )

                # 회전된 궤적의 항해 가능 점수 계산
                navigable_count = 0
                for i in range(0, n_points, 5):  # 5분 간격으로 체크
                    if self.depth_checker.is_navigable(rotated[i, 0], rotated[i, 1], check_neighbors=False):
                        navigable_count += 1

                # 중간점과 종점 가중치 추가
                if self.depth_checker.is_navigable(rotated[n_points // 2, 0], rotated[n_points // 2, 1]):
                    navigable_count += 3
                if self.depth_checker.is_navigable(rotated[-1, 0], rotated[-1, 1]):
                    navigable_count += 5

                # 작은 각도 보너스 (자연스러운 회전 선호)
                angle_penalty = abs(angle_offset) / 180.0
                score = navigable_count - angle_penalty * 2

                if score > best_score:
                    best_score = score
                    best_coords = rotated

            if best_coords is not None and best_score > 5:
                return best_coords

        return corrected

    def _rotate_trajectory(self, coords, center, base_angle, offset_deg):
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

    def _correct_depth_gradual(self, coords, last_position):
        """수심 기반 점진적 경로 보정 (급격한 이동 방지)"""
        corrected = coords.copy()
        prev_valid = last_position.copy()

        # 위도 1도 ≈ 111km, 경도 1도 ≈ 88km (한국 위도 기준)
        max_lat_change = self.max_correction_per_step_km / 111.0
        max_lon_change = self.max_correction_per_step_km / 88.0

        for i in range(len(corrected)):
            lat, lon = corrected[i]
            original_lat, original_lon = lat, lon

            if not self.depth_checker.is_navigable(lat, lon):
                # 항해 불가능 → 가장 가까운 항해 가능 지점 찾기
                nearest = self.depth_checker.find_nearest_navigable(lat, lon)

                if nearest is not None:
                    target_lat, target_lon = nearest

                    # 보정량 계산
                    dlat = target_lat - lat
                    dlon = target_lon - lon

                    # 최대 보정량 제한 (점진적 이동)
                    if abs(dlat) > max_lat_change:
                        dlat = max_lat_change * np.sign(dlat)
                    if abs(dlon) > max_lon_change:
                        dlon = max_lon_change * np.sign(dlon)

                    corrected[i, 0] = lat + dlat
                    corrected[i, 1] = lon + dlon
                else:
                    # 찾지 못하면 이전 유효 위치 방향으로 점진적 이동
                    dlat = (prev_valid[0] - lat) * 0.3
                    dlon = (prev_valid[1] - lon) * 0.3

                    # 최대 보정량 제한
                    if abs(dlat) > max_lat_change:
                        dlat = max_lat_change * np.sign(dlat)
                    if abs(dlon) > max_lon_change:
                        dlon = max_lon_change * np.sign(dlon)

                    corrected[i, 0] = lat + dlat
                    corrected[i, 1] = lon + dlon

            # 유효 위치 업데이트 (항해 가능한 경우에만)
            if self.depth_checker.is_navigable(corrected[i, 0], corrected[i, 1]):
                prev_valid = corrected[i].copy()

        return corrected

    def _correct_direction_gradual(self, coords, last_position, last_cog, weight):
        """과거 항적 방향 기반 점진적 보정 (급격한 방향 전환 방지)"""
        corrected = coords.copy()

        # 이전 보정 각도 (관성 유지용)
        prev_correction = 0.0

        for i in range(len(corrected)):
            lat, lon = corrected[i]

            # 과거 항적의 선호 방향 조회
            pref = self.track_grid.get_preferred_direction(lat, lon)

            if pref is None:
                # 과거 항적 정보 없으면 이전 보정 관성 유지 (감쇠)
                prev_correction *= 0.8
                continue

            # 현재 이동 방향 계산
            if i == 0:
                prev_lat, prev_lon = last_position
            else:
                prev_lat, prev_lon = corrected[i-1]

            dlat = lat - prev_lat
            dlon = lon - prev_lon

            if abs(dlat) < 1e-7 and abs(dlon) < 1e-7:
                prev_correction *= 0.8
                continue

            current_cog = np.degrees(np.arctan2(dlon, dlat)) % 360
            pref_cog = pref['cog']
            consistency = pref['consistency']

            # 방향 차이 계산 (각도 차이, -180~180)
            angle_diff = ((pref_cog - current_cog + 180) % 360) - 180

            # 방향 차이가 크면 (90도 이상) 보정하지 않음 (역주행 방지)
            if abs(angle_diff) > 90:
                prev_correction *= 0.5
                continue

            # 목표 보정량 계산 (일관성과 가중치 적용)
            target_correction = angle_diff * weight * consistency * 0.3

            # 점진적 보정: 이전 보정과 목표 보정의 블렌딩
            # + 최대 각도 변화 제한
            correction_change = target_correction - prev_correction
            if abs(correction_change) > self.max_angle_change_deg:
                correction_change = self.max_angle_change_deg * np.sign(correction_change)

            correction = prev_correction + correction_change * 0.5
            prev_correction = correction

            # 새로운 방향으로 좌표 보정
            dist = np.sqrt(dlat**2 + dlon**2)
            new_cog = np.radians(current_cog + correction)

            corrected[i, 0] = prev_lat + dist * np.cos(new_cog)
            corrected[i, 1] = prev_lon + dist * np.sin(new_cog)

        return corrected

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

        for i in range(len(coords)):
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

    def _final_depth_check_smooth(self, coords, last_position):
        """최종 수심 체크 및 부드러운 우회 보정"""
        corrected = coords.copy()
        prev_valid = last_position.copy()

        # 이전 이동 방향 (관성 유지용)
        prev_direction = None

        for i in range(len(corrected)):
            lat, lon = corrected[i]

            # 현재 이동 방향 계산
            if i > 0:
                prev_direction = np.array([lat - corrected[i-1, 0], lon - corrected[i-1, 1]])
            elif prev_direction is None:
                prev_direction = np.array([lat - last_position[0], lon - last_position[1]])

            if not self.depth_checker.is_navigable(lat, lon):
                # 부드러운 우회: 여러 방향 시도
                best_pos = None
                best_score = -np.inf

                # 더 세밀한 alpha 값 (점진적 접근)
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    mid_lat = prev_valid[0] * alpha + lat * (1 - alpha)
                    mid_lon = prev_valid[1] * alpha + lon * (1 - alpha)

                    if self.depth_checker.is_navigable(mid_lat, mid_lon):
                        # 방향 유지도 점수 계산 (진행 방향과 일치할수록 높은 점수)
                        if prev_direction is not None and np.linalg.norm(prev_direction) > 1e-7:
                            new_dir = np.array([mid_lat - prev_valid[0], mid_lon - prev_valid[1]])
                            if np.linalg.norm(new_dir) > 1e-7:
                                cos_sim = np.dot(prev_direction, new_dir) / (
                                    np.linalg.norm(prev_direction) * np.linalg.norm(new_dir))
                                score = (1 - alpha) + cos_sim * 0.5  # 원래 위치 + 방향 유지
                            else:
                                score = (1 - alpha)
                        else:
                            score = (1 - alpha)

                        if score > best_score:
                            best_score = score
                            best_pos = [mid_lat, mid_lon]

                if best_pos is not None:
                    corrected[i] = best_pos
                else:
                    # 중간점도 불가능하면 이전 위치에서 약간만 이동
                    corrected[i, 0] = prev_valid[0] + prev_direction[0] * 0.1
                    corrected[i, 1] = prev_valid[1] + prev_direction[1] * 0.1

            # 유효 위치 업데이트
            if self.depth_checker.is_navigable(corrected[i, 0], corrected[i, 1]):
                prev_valid = corrected[i].copy()

        return corrected

    def _ensure_direction_continuity(self, coords, last_position, last_cog):
        """방향 연속성 보장 (급격한 각도 변화 완화) - 안정화 버전"""
        if len(coords) < 3:
            return coords

        corrected = coords.copy()
        original_coords = coords.copy()  # 원본 보존

        # 좌표 유효 범위 (한국 근해)
        LAT_MIN, LAT_MAX = 30.0, 40.0
        LON_MIN, LON_MAX = 120.0, 135.0

        # 전체 이동 거리 계산 (저속 선박 판별용)
        total_dist = np.sqrt(
            (coords[-1, 0] - coords[0, 0])**2 +
            (coords[-1, 1] - coords[0, 1])**2
        )

        # 저속 선박 (60분간 0.01도 = ~1km 미만 이동)이면 방향 보정 스킵
        # 작은 움직임에 방향 보정 적용하면 진동 발생
        if total_dist < 0.01:
            return coords

        # 이전 방향 초기화
        if last_cog is not None:
            prev_cog = last_cog
        else:
            dlat = coords[0, 0] - last_position[0]
            dlon = coords[0, 1] - last_position[1]
            if abs(dlat) < 1e-9 and abs(dlon) < 1e-9:
                prev_cog = 0.0
            else:
                prev_cog = np.degrees(np.arctan2(dlon, dlat)) % 360

        # 기준 위치 (last_position에서 시작)
        base_lat, base_lon = last_position[0], last_position[1]

        for i in range(1, len(coords)):
            prev_lat, prev_lon = corrected[i-1]
            lat, lon = original_coords[i]  # 원본 좌표 사용

            # 좌표 유효성 검사
            if not (LAT_MIN <= prev_lat <= LAT_MAX and LON_MIN <= prev_lon <= LON_MAX):
                # 이전 좌표가 유효하지 않으면 원본 사용
                corrected[i] = original_coords[i].copy()
                continue

            dlat = lat - prev_lat
            dlon = lon - prev_lon
            dist = np.sqrt(dlat**2 + dlon**2)

            # 거리가 너무 작거나 너무 크면 스킵 (저속 선박 보호)
            if dist < 0.0001 or dist > 0.1:  # 0.0001도 ≈ 11m, 0.1도 ≈ 11km
                corrected[i] = original_coords[i].copy()
                continue

            current_cog = np.degrees(np.arctan2(dlon, dlat)) % 360

            # 방향 변화 계산
            angle_diff = ((current_cog - prev_cog + 180) % 360) - 180

            # 최대 각도 변화 제한
            if abs(angle_diff) > self.max_angle_change_deg:
                limited_diff = self.max_angle_change_deg * np.sign(angle_diff)
                new_cog = (prev_cog + limited_diff) % 360
                new_cog_rad = np.radians(new_cog)

                new_lat = prev_lat + dist * np.cos(new_cog_rad)
                new_lon = prev_lon + dist * np.sin(new_cog_rad)

                # 새 좌표 유효성 검사
                if LAT_MIN <= new_lat <= LAT_MAX and LON_MIN <= new_lon <= LON_MAX:
                    corrected[i, 0] = new_lat
                    corrected[i, 1] = new_lon
                    prev_cog = new_cog
                else:
                    # 유효하지 않으면 원본 유지
                    corrected[i] = original_coords[i].copy()
                    prev_cog = current_cog
            else:
                prev_cog = current_cog

        return corrected


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
        self.track_grid = HistoricalTrackGrid(grid_resolution=0.01)

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

        self.track_grid = HistoricalTrackGrid(grid_resolution=0.01)
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
