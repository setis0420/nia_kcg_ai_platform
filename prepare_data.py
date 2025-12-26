# -*- coding: utf-8 -*-
"""
학습 데이터 전처리 및 저장 (Multiprocessing 버전)
================================================
멀티프로세싱을 사용하여 병렬 처리
+ 해역 경계 마스크 및 밀도 기반 기준 항로 생성

사용법:
    python prepare_data.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                           --transition_folder "area_transition_results" \
                           --output_dir "prepared_data" \
                           --seq_len 50 --stride 3 --num_workers 4
"""

import os
import gc
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import pandas as pd
import warnings

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")

# 격자 크기 (도)
GRID_SIZE = 0.05


def compute_grid_id(lat, lon, lat_min, lon_min, grid_size, num_cols):
    """위경도를 격자 ID로 변환"""
    grid_row = int((lat - lat_min) / grid_size)
    grid_col = int((lon - lon_min) / grid_size)
    return grid_row * num_cols + grid_col


def create_grid_mapping(lat_min, lat_max, lon_min, lon_max, grid_size=0.05):
    """격자 정보 생성"""
    num_rows = int(np.ceil((lat_max - lat_min) / grid_size)) + 1
    num_cols = int(np.ceil((lon_max - lon_min) / grid_size)) + 1
    return {
        'lat_min': lat_min,
        'lon_min': lon_min,
        'lat_max': lat_max,
        'lon_max': lon_max,
        'grid_size': grid_size,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'total_grids': num_rows * num_cols
    }


# ============================================================
# 해역 경계 및 기준 항로 생성 함수들
# ============================================================

def create_valid_sea_mask(all_lat, all_lon, grid_size=0.01, min_count=1, expand_pixels=1):
    """
    유효 해역 마스크 생성 - 항적이 있었던 격자만 허용

    Args:
        all_lat, all_lon: 모든 항적의 위경도 배열
        grid_size: 격자 크기 (도 단위, 기본값 0.01도 ≈ 1.1km)
        min_count: 최소 항적 수
        expand_pixels: 마스크 확장 픽셀 수

    Returns:
        dict: 해역 마스크 정보
    """
    lat_min, lat_max = all_lat.min(), all_lat.max()
    lon_min, lon_max = all_lon.min(), all_lon.max()

    # 여유 추가
    margin = grid_size * 2
    lat_min -= margin
    lat_max += margin
    lon_min -= margin
    lon_max += margin

    num_rows = int(np.ceil((lat_max - lat_min) / grid_size)) + 1
    num_cols = int(np.ceil((lon_max - lon_min) / grid_size)) + 1

    # 격자별 카운트
    grid_count = np.zeros((num_rows, num_cols), dtype=np.int32)

    for lat, lon in zip(all_lat, all_lon):
        row = int((lat - lat_min) / grid_size)
        col = int((lon - lon_min) / grid_size)
        if 0 <= row < num_rows and 0 <= col < num_cols:
            grid_count[row, col] += 1

    # 유효 해역 마스크
    valid_mask = grid_count >= min_count

    # 확장 (주변 격자도 허용)
    if expand_pixels > 0 and HAS_SCIPY:
        valid_mask = ndimage.binary_dilation(valid_mask, iterations=expand_pixels)

    valid_count = np.sum(valid_mask)
    total_count = num_rows * num_cols

    print(f"[SEA-MASK] 유효 해역: {valid_count:,} / {total_count:,} ({100*valid_count/total_count:.1f}%)")

    return {
        'valid_mask': valid_mask,
        'grid_count': grid_count,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'grid_size': grid_size,
        'num_rows': num_rows,
        'num_cols': num_cols,
    }


def create_density_routes(all_lat, all_lon, grid_size=0.005, top_percentile=80):
    """
    밀도 기반 기준 항로 생성 - 항적이 밀집된 격자들을 추출

    Args:
        all_lat, all_lon: 모든 항적의 위경도 배열
        grid_size: 격자 크기 (도 단위, 기본값 0.005도 ≈ 550m)
        top_percentile: 상위 N% 밀도를 기준 항로로 지정

    Returns:
        dict: 기준 항로 정보
    """
    lat_min, lat_max = all_lat.min(), all_lat.max()
    lon_min, lon_max = all_lon.min(), all_lon.max()

    num_rows = int(np.ceil((lat_max - lat_min) / grid_size)) + 1
    num_cols = int(np.ceil((lon_max - lon_min) / grid_size)) + 1

    # 격자별 밀도
    density_grid = np.zeros((num_rows, num_cols), dtype=np.float32)

    for lat, lon in zip(all_lat, all_lon):
        row = int((lat - lat_min) / grid_size)
        col = int((lon - lon_min) / grid_size)
        if 0 <= row < num_rows and 0 <= col < num_cols:
            density_grid[row, col] += 1

    # 정규화
    if density_grid.max() > 0:
        density_grid = density_grid / density_grid.max()

    # 상위 밀도 격자 추출
    nonzero_density = density_grid[density_grid > 0]
    if len(nonzero_density) > 0:
        threshold = np.percentile(nonzero_density, top_percentile)
        high_density_mask = density_grid >= threshold
    else:
        threshold = 0
        high_density_mask = density_grid > 0

    # 고밀도 격자의 중심점 추출
    high_density_points = []
    rows, cols = np.where(high_density_mask)
    for r, c in zip(rows, cols):
        center_lat = lat_min + (r + 0.5) * grid_size
        center_lon = lon_min + (c + 0.5) * grid_size
        high_density_points.append([center_lat, center_lon, density_grid[r, c]])

    high_density_points = np.array(high_density_points) if high_density_points else np.array([]).reshape(0, 3)

    print(f"[ROUTE] 기준 항로 격자: {len(high_density_points):,} 개 (상위 {100-top_percentile}%)")

    return {
        'density_grid': density_grid,
        'high_density_mask': high_density_mask,
        'high_density_points': high_density_points,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'grid_size': grid_size,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'threshold': threshold,
    }


def save_boundary_data(sea_mask_info, route_info, output_path):
    """해역 경계 데이터 저장 (npz 형식)"""
    np.savez_compressed(
        output_path,
        # 해역 마스크
        valid_mask=sea_mask_info['valid_mask'],
        grid_count=sea_mask_info['grid_count'],
        mask_lat_min=sea_mask_info['lat_min'],
        mask_lat_max=sea_mask_info['lat_max'],
        mask_lon_min=sea_mask_info['lon_min'],
        mask_lon_max=sea_mask_info['lon_max'],
        mask_grid_size=sea_mask_info['grid_size'],
        mask_num_rows=sea_mask_info['num_rows'],
        mask_num_cols=sea_mask_info['num_cols'],
        # 밀도 항로
        density_grid=route_info['density_grid'],
        high_density_mask=route_info['high_density_mask'],
        high_density_points=route_info['high_density_points'],
        route_lat_min=route_info['lat_min'],
        route_lat_max=route_info['lat_max'],
        route_lon_min=route_info['lon_min'],
        route_lon_max=route_info['lon_max'],
        route_grid_size=route_info['grid_size'],
        route_num_rows=route_info['num_rows'],
        route_num_cols=route_info['num_cols'],
        route_threshold=route_info['threshold'],
    )
    print(f"[SAVED] 해역 경계 데이터: {output_path}")


def split_by_gap(df, max_gap_days=1):
    """시간 간격으로 segment 분리"""
    if df is None or df.empty:
        return []
    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return []

    diff = df["datetime"].diff()
    new_seg = (diff >= pd.Timedelta(days=max_gap_days)) | diff.isna()
    seg_id = new_seg.cumsum()

    return [g.copy() for _, g in df.groupby(seg_id) if len(g) > 0]


def data_intp(df):
    """1분 간격 보간"""
    if df is None or df.empty:
        return None

    df = df.drop_duplicates(subset=["datetime", "lat", "lon", "sog", "cog"], keep="first")
    df = df.sort_values("datetime").copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["lat", "lon", "sog", "cog"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [c for c in df.columns if c in ["datetime","mmsi","lat","lon","sog","cog","fid","start_area","end_area"]]
    df = df[keep_cols].dropna(subset=["datetime", "lat", "lon", "sog", "cog"])
    if df.empty:
        return None

    dt_range = pd.date_range(
        start=df["datetime"].iloc[0].floor("T"),
        end=df["datetime"].iloc[-1].ceil("T"),
        freq="1min"
    )

    range_df = pd.DataFrame({"datetime": dt_range})
    range_df["mmsi"] = df["mmsi"].iloc[0] if "mmsi" in df.columns else np.nan
    range_df["fid"] = df["fid"].iloc[0] if "fid" in df.columns else np.nan
    range_df["start_area"] = df["start_area"].iloc[0] if "start_area" in df.columns else "unknown"
    range_df["end_area"] = df["end_area"].iloc[0] if "end_area" in df.columns else "unknown"

    merge_df = (
        pd.concat([df, range_df], axis=0)
          .set_index("datetime")
          .sort_index()
    )

    for col in ["lat", "lon", "sog", "cog"]:
        merge_df[col] = pd.to_numeric(merge_df[col], errors="coerce")

    merge_df["sin_course"] = np.sin(np.radians(merge_df["cog"]))
    merge_df["cos_course"] = np.cos(np.radians(merge_df["cog"]))

    numeric_cols = ["lat", "lon", "sog", "cog", "sin_course", "cos_course"]
    merge_df[numeric_cols] = merge_df[numeric_cols].astype("float")

    intp_df = merge_df.copy()
    intp_df[numeric_cols] = intp_df[numeric_cols].interpolate(method="linear")

    intp_df["cog"] = np.degrees(np.arctan2(intp_df["sin_course"], intp_df["cos_course"]))
    intp_df["cog"] = (intp_df["cog"] + 360) % 360

    intp_df = intp_df.drop(columns=["sin_course","cos_course"], errors="ignore").reset_index()
    intp_df = intp_df.dropna(subset=["lat","lon","sog","cog"])
    return intp_df


def parse_filename(filename):
    """파일명에서 mmsi, start_area, end_area 추출"""
    name = os.path.splitext(filename)[0]
    parts = name.split('_')

    if len(parts) >= 3:
        mmsi = parts[0]
        start_area = parts[1]
        end_area = parts[2]
        return mmsi, start_area, end_area

    return None, None, None


def load_transition_data(transition_folder):
    """전이 정보 CSV 파일들을 로드하여 병합"""
    if not os.path.exists(transition_folder):
        raise FileNotFoundError(f"전이 정보 폴더가 존재하지 않습니다: {transition_folder}")

    file_list = [f for f in os.listdir(transition_folder) if f.lower().endswith('.csv')]

    if len(file_list) == 0:
        raise ValueError(f"전이 정보 폴더에 CSV 파일이 없습니다: {transition_folder}")

    df_list = []
    for f in file_list:
        file_path = os.path.join(transition_folder, f)
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['source_file'] = f
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] 전이 정보 로드 완료: {len(merged_df)} 건, {len(file_list)} 파일")
    return merged_df


def get_file_list(transition_df, data_folder):
    """유효한 파일 목록 생성"""
    file_list = []

    for i in range(len(transition_df)):
        mmsi = transition_df.mmsi.iloc[i]
        s_area = transition_df.start_area.iloc[i]
        e_area = transition_df.end_area.iloc[i]
        start_time = pd.to_datetime(transition_df.start_time.iloc[i]) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(transition_df.end_time.iloc[i]) + pd.Timedelta('1 hour')

        start_time_str = start_time.strftime("%Y%m%d%H%M%S")
        end_time_str = end_time.strftime("%Y%m%d%H%M%S")

        filename = f'{mmsi}_{s_area}_{e_area}_{start_time_str}_{end_time_str}.csv'
        filepath = os.path.join(data_folder, filename)

        if os.path.exists(filepath):
            file_list.append({
                'filepath': filepath,
                'filename': filename,
                'fid': i,
                'mmsi': mmsi,
                'start_area': s_area,
                'end_area': e_area,
            })

    return file_list


def process_single_file_for_stats(file_info, seq_len, cog_mirror=True):
    """
    단일 파일 처리 - 통계 수집용 (Pass 1)
    Returns: dict with stats or None
    """
    try:
        trj = pd.read_csv(file_info['filepath'], encoding='cp949')
        trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
        trj['fid'] = file_info['fid']

        parsed_mmsi, parsed_start, parsed_end = parse_filename(file_info['filename'])
        trj['start_area'] = parsed_start if parsed_start else file_info['start_area']
        trj['end_area'] = parsed_end if parsed_end else file_info['end_area']

        trj["datetime"] = pd.to_datetime(trj["datetime"], errors="coerce")
        for c in ["lat", "lon", "sog", "cog", "mmsi"]:
            if c in trj.columns:
                trj[c] = pd.to_numeric(trj[c], errors="coerce")
        trj = trj.dropna(subset=["datetime", "lat", "lon", "sog", "cog"])

        if trj.empty:
            return None

        segments = split_by_gap(trj, max_gap_days=1)

        result = {
            'file_idx': file_info['fid'],
            'n_total': 0,
            'sum_x': np.zeros(5, dtype=np.float64),
            'sum_x2': np.zeros(5, dtype=np.float64),
            'lat_min': float('inf'),
            'lat_max': float('-inf'),
            'lon_min': float('inf'),
            'lon_max': float('-inf'),
            'mmsi_set': set(),
            'start_area_set': set(),
            'end_area_set': set(),
            'segment_infos': [],
            'all_lat': [],
            'all_lon': [],
        }

        for seg_idx, seg_df in enumerate(segments):
            intp_df = data_intp(seg_df)
            if intp_df is None or len(intp_df) == 0:
                continue

            intp_df = intp_df.sort_values("datetime").reset_index(drop=True)
            n = len(intp_df)

            if n < seq_len + 1:
                continue

            sin_cog = np.sin(np.radians(intp_df["cog"].values))
            cos_cog = np.cos(np.radians(intp_df["cog"].values))
            if cog_mirror:
                sin_cog = -sin_cog

            x = np.column_stack([
                intp_df["lat"].values,
                intp_df["lon"].values,
                intp_df["sog"].values,
                sin_cog,
                cos_cog,
            ]).astype(np.float64)

            result['n_total'] += n
            result['sum_x'] += x.sum(axis=0)
            result['sum_x2'] += (x ** 2).sum(axis=0)

            result['lat_min'] = min(result['lat_min'], intp_df["lat"].min())
            result['lat_max'] = max(result['lat_max'], intp_df["lat"].max())
            result['lon_min'] = min(result['lon_min'], intp_df["lon"].min())
            result['lon_max'] = max(result['lon_max'], intp_df["lon"].max())

            # 해역 경계 생성용 좌표 수집 (샘플링하여 메모리 절약)
            lat_arr = intp_df["lat"].values
            lon_arr = intp_df["lon"].values
            # 10개 중 1개만 샘플링
            sample_idx = np.arange(0, len(lat_arr), 10)
            result['all_lat'].extend(lat_arr[sample_idx].tolist())
            result['all_lon'].extend(lon_arr[sample_idx].tolist())

            result['mmsi_set'].add(file_info['mmsi'])
            result['start_area_set'].add(intp_df["start_area"].iloc[0])
            result['end_area_set'].add(intp_df["end_area"].iloc[0])

            result['segment_infos'].append({
                'file_idx': file_info['fid'],
                'seg_idx': seg_idx,
                'length': n,
                'fid': file_info['fid'],
                'mmsi': file_info['mmsi'],
                'start_area': intp_df["start_area"].iloc[0],
                'end_area': intp_df["end_area"].iloc[0],
            })

        return result if result['n_total'] > 0 else None

    except Exception as e:
        return None


def process_single_file_for_data(args):
    """
    단일 파일 처리 - 데이터 생성용 (Pass 2)
    """
    file_info, seg_infos, seq_len, stride, cog_mirror, x_mean, x_std, \
        mmsi_vocab, start_vocab, end_vocab, grid_info = args

    try:
        trj = pd.read_csv(file_info['filepath'], encoding='cp949')
        trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
        trj['fid'] = file_info['fid']

        parsed_mmsi, parsed_start, parsed_end = parse_filename(file_info['filename'])
        trj['start_area'] = parsed_start if parsed_start else file_info['start_area']
        trj['end_area'] = parsed_end if parsed_end else file_info['end_area']

        trj["datetime"] = pd.to_datetime(trj["datetime"], errors="coerce")
        for c in ["lat", "lon", "sog", "cog", "mmsi"]:
            if c in trj.columns:
                trj[c] = pd.to_numeric(trj[c], errors="coerce")
        trj = trj.dropna(subset=["datetime", "lat", "lon", "sog", "cog"])

        if trj.empty:
            return []

        segments_raw = split_by_gap(trj, max_gap_days=1)

        segments = []
        for seg_df in segments_raw:
            intp_df = data_intp(seg_df)
            if intp_df is not None and len(intp_df) > 0:
                intp_df = intp_df.sort_values("datetime").reset_index(drop=True)

                sin_cog = np.sin(np.radians(intp_df["cog"].values))
                cos_cog = np.cos(np.radians(intp_df["cog"].values))
                if cog_mirror:
                    sin_cog = -sin_cog

                intp_df["sin_cog"] = sin_cog
                intp_df["cos_cog"] = cos_cog
                segments.append(intp_df)

        results = []

        for seg_info in seg_infos:
            seg_idx = seg_info['seg_idx']

            if seg_idx >= len(segments):
                continue

            seg_df = segments[seg_idx]
            n = len(seg_df)

            if n < seq_len + 1:
                continue

            x = np.column_stack([
                seg_df["lat"].values,
                seg_df["lon"].values,
                seg_df["sog"].values,
                seg_df["sin_cog"].values,
                seg_df["cos_cog"].values,
            ]).astype(np.float32)

            xn = (x - x_mean) / x_std
            yn = xn.copy()

            mmsi_id = mmsi_vocab.get(seg_info['mmsi'], 0)
            start_id = start_vocab.get(seg_info['start_area'], 0)
            end_id = end_vocab.get(seg_info['end_area'], 0)

            lat_arr = seg_df["lat"].values
            lon_arr = seg_df["lon"].values
            grid_rows = ((lat_arr - grid_info['lat_min']) / grid_info['grid_size']).astype(int)
            grid_cols = ((lon_arr - grid_info['lon_min']) / grid_info['grid_size']).astype(int)
            grid_ids = grid_rows * grid_info['num_cols'] + grid_cols
            grid_ids = np.clip(grid_ids, 0, grid_info['total_grids'])

            x_cat = np.column_stack([
                np.full(n, mmsi_id),
                np.full(n, start_id),
                np.full(n, end_id),
                grid_ids,
            ]).astype(np.int64)

            results.append({
                'xn': xn,
                'x_cat': x_cat,
                'yn': yn,
                'length': n,
                'seg_info': seg_info,
            })

        return results

    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="학습 데이터 전처리 및 저장 (Multiprocessing 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_folder", type=str,
                        default=r"G:\NIA_ai_project\항적데이터 추출\여수_테스트",
                        help="항적 CSV 파일이 저장된 폴더")
    parser.add_argument("--transition_folder", type=str,
                        default="area_transition_results",
                        help="전이 정보 CSV 파일이 저장된 폴더")
    parser.add_argument("--output_dir", type=str, default="prepared_data",
                        help="전처리된 데이터 저장 폴더 (기본값: prepared_data)")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="시퀀스 길이 (기본값: 50)")
    parser.add_argument("--stride", type=int, default=3,
                        help="슬라이딩 윈도우 이동 간격 (기본값: 3)")
    parser.add_argument("--grid_size", type=float, default=0.05,
                        help="격자 크기 (도 단위, 기본값: 0.05)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="병렬 처리 워커 수 (기본값: 4)")
    parser.add_argument("--mask_grid_size", type=float, default=0.01,
                        help="해역 마스크 격자 크기 (도, 기본값: 0.01)")
    parser.add_argument("--route_grid_size", type=float, default=0.005,
                        help="항로 밀도 격자 크기 (도, 기본값: 0.005)")
    parser.add_argument("--route_percentile", type=float, default=80,
                        help="기준 항로 밀도 백분위 (기본값: 80)")

    args = parser.parse_args()

    seq_len = args.seq_len
    stride = args.stride
    grid_size = args.grid_size
    num_workers = min(args.num_workers, cpu_count())
    cog_mirror = True

    print("=" * 60)
    print("학습 데이터 전처리 (Multiprocessing 버전)")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_folder}")
    print(f"전이 정보 폴더: {args.transition_folder}")
    print(f"출력 폴더: {args.output_dir}")
    print(f"시퀀스 길이: {seq_len}")
    print(f"Stride: {stride}")
    print(f"격자 크기: {grid_size}도")
    print(f"워커 수: {num_workers}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 전이 정보 로드
    print("\n[STEP 1] 파일 목록 생성")
    transition_df = load_transition_data(args.transition_folder)
    file_list = get_file_list(transition_df, args.data_folder)
    print(f"[INFO] 유효한 파일 수: {len(file_list)}")

    # ==============================================
    # PASS 1: 통계 수집 (멀티프로세싱)
    # ==============================================
    print(f"\n[STEP 2] 통계 수집 (Pass 1) - {num_workers} workers")

    process_func = partial(process_single_file_for_stats, seq_len=seq_len, cog_mirror=cog_mirror)

    with Pool(num_workers) as pool:
        results = list(pool.imap(process_func, file_list, chunksize=10))

    # 결과 병합
    n_total = 0
    sum_x = np.zeros(5, dtype=np.float64)
    sum_x2 = np.zeros(5, dtype=np.float64)
    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')
    mmsi_set = set()
    start_area_set = set()
    end_area_set = set()
    segment_info_list = []

    all_lat_list = []
    all_lon_list = []

    for r in results:
        if r is None:
            continue
        n_total += r['n_total']
        sum_x += r['sum_x']
        sum_x2 += r['sum_x2']
        lat_min = min(lat_min, r['lat_min'])
        lat_max = max(lat_max, r['lat_max'])
        lon_min = min(lon_min, r['lon_min'])
        lon_max = max(lon_max, r['lon_max'])
        mmsi_set.update(r['mmsi_set'])
        start_area_set.update(r['start_area_set'])
        end_area_set.update(r['end_area_set'])
        segment_info_list.extend(r['segment_infos'])
        all_lat_list.extend(r['all_lat'])
        all_lon_list.extend(r['all_lon'])

    del results
    gc.collect()

    # 해역 경계 및 기준 항로 생성
    print(f"\n[STEP 2.5] 해역 경계 및 기준 항로 생성")
    all_lat_arr = np.array(all_lat_list)
    all_lon_arr = np.array(all_lon_list)
    print(f"[INFO] 수집된 좌표 수: {len(all_lat_arr):,}")

    sea_mask_info = create_valid_sea_mask(
        all_lat_arr, all_lon_arr,
        grid_size=args.mask_grid_size,
        min_count=1,
        expand_pixels=1
    )

    route_info = create_density_routes(
        all_lat_arr, all_lon_arr,
        grid_size=args.route_grid_size,
        top_percentile=args.route_percentile
    )

    # 해역 경계 저장
    boundary_path = os.path.join(args.output_dir, "sea_boundary.npz")
    save_boundary_data(sea_mask_info, route_info, boundary_path)

    del all_lat_list, all_lon_list, all_lat_arr, all_lon_arr
    gc.collect()

    print(f"[INFO] 유효 segment 수: {len(segment_info_list)}")
    print(f"[INFO] 총 데이터 포인트: {n_total}")

    if n_total == 0:
        raise RuntimeError("처리된 데이터가 없습니다.")

    # 평균/표준편차 계산
    x_mean = (sum_x / n_total).astype(np.float32).reshape(1, -1)
    x_var = (sum_x2 / n_total) - (sum_x / n_total) ** 2
    x_std = (np.sqrt(np.maximum(x_var, 1e-12)) + 1e-6).astype(np.float32).reshape(1, -1)

    y_mean = x_mean.copy()
    y_std = x_std.copy()

    print(f"[INFO] x_mean: {x_mean}")
    print(f"[INFO] x_std: {x_std}")

    # 격자 정보
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    lat_bounds = (lat_min - lat_margin, lat_max + lat_margin)
    lon_bounds = (lon_min - lon_margin, lon_max + lon_margin)

    grid_info = create_grid_mapping(
        lat_bounds[0], lat_bounds[1],
        lon_bounds[0], lon_bounds[1],
        grid_size
    )

    print(f"[INFO] 항로 범위: lat={lat_bounds[0]:.4f}~{lat_bounds[1]:.4f}, lon={lon_bounds[0]:.4f}~{lon_bounds[1]:.4f}")
    print(f"[INFO] 격자: {grid_info['num_rows']}x{grid_info['num_cols']} = {grid_info['total_grids']}")

    # Vocab 생성
    mmsi_vocab = {v: i+1 for i, v in enumerate(sorted(mmsi_set))}
    start_vocab = {v: i+1 for i, v in enumerate(sorted(start_area_set))}
    end_vocab = {v: i+1 for i, v in enumerate(sorted(end_area_set))}

    print(f"[INFO] MMSI: {len(mmsi_vocab)} 종류")
    print(f"[INFO] Start Area: {len(start_vocab)} 종류")
    print(f"[INFO] End Area: {len(end_vocab)} 종류")

    # ==============================================
    # PASS 2: 정규화 및 저장 (멀티프로세싱)
    # ==============================================
    print(f"\n[STEP 3] 정규화 및 저장 (Pass 2) - {num_workers} workers")

    # 파일별로 segment 정보 그룹화
    file_to_segments = defaultdict(list)
    for seg_info in segment_info_list:
        file_to_segments[seg_info['file_idx']].append(seg_info)

    # 작업 목록 생성
    tasks = []
    for file_info in file_list:
        if file_info['fid'] in file_to_segments:
            seg_infos = file_to_segments[file_info['fid']]
            tasks.append((
                file_info, seg_infos, seq_len, stride, cog_mirror,
                x_mean, x_std, mmsi_vocab, start_vocab, end_vocab, grid_info
            ))

    # 병렬 처리
    with Pool(num_workers) as pool:
        results = list(pool.imap(process_single_file_for_data, tasks, chunksize=10))

    # 결과 병합
    all_Xn_num = []
    all_X_cat = []
    all_Yn = []
    segment_bounds = []
    segment_starts = []
    current_offset = 0

    for file_results in results:
        for r in file_results:
            xn = r['xn']
            x_cat = r['x_cat']
            yn = r['yn']
            n = r['length']

            all_Xn_num.append(xn)
            all_X_cat.append(x_cat)
            all_Yn.append(yn)

            start_idx = current_offset
            end_idx = current_offset + n
            segment_bounds.append((start_idx, end_idx))

            starts = []
            max_start = end_idx - 1 - seq_len
            if max_start >= start_idx:
                for i in range(start_idx, max_start + 1, stride):
                    starts.append(i)
            segment_starts.append(starts)

            current_offset = end_idx

    del results
    gc.collect()

    print(f"[INFO] 처리된 segment 수: {len(segment_bounds)}")

    # 병합
    print("\n[STEP 4] 데이터 병합 및 저장")

    Xn_num = np.vstack(all_Xn_num).astype(np.float32)
    X_cat = np.vstack(all_X_cat).astype(np.int64)
    Yn = np.vstack(all_Yn).astype(np.float32)

    del all_Xn_num, all_X_cat, all_Yn
    gc.collect()

    all_indices = [i for starts in segment_starts for i in starts]

    print(f"[INFO] 최종 데이터 shape: Xn_num={Xn_num.shape}, X_cat={X_cat.shape}")
    print(f"[INFO] 총 시퀀스 수: {len(all_indices)}")

    # 저장
    data_path = os.path.join(args.output_dir, "training_data.npz")
    np.savez_compressed(
        data_path,
        Xn_num=Xn_num,
        X_cat=X_cat,
        Yn=Yn,
        segment_bounds=np.array(segment_bounds),
        all_indices=np.array(all_indices),
    )
    print(f"[SAVED] 학습 데이터: {data_path}")

    # 메타 정보
    meta = {
        'seq_len': seq_len,
        'stride': stride,
        'grid_size': grid_size,
        'cog_mirror': cog_mirror,
        'numeric_cols': np.array(['lat', 'lon', 'sog', 'sin_cog', 'cos_cog']),
        'cat_cols': np.array(['mmsi_id', 'start_area_id', 'end_area_id', 'grid_id']),
        'target_cols': np.array(['lat', 'lon', 'sog', 'sin_cog', 'cos_cog']),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'mmsi_vocab': str(mmsi_vocab),
        'start_vocab': str(start_vocab),
        'end_vocab': str(end_vocab),
        'num_mmsi': len(mmsi_vocab) + 1,
        'num_start_area': len(start_vocab) + 1,
        'num_end_area': len(end_vocab) + 1,
        'grid_info_lat_min': grid_info['lat_min'],
        'grid_info_lon_min': grid_info['lon_min'],
        'grid_info_lat_max': grid_info['lat_max'],
        'grid_info_lon_max': grid_info['lon_max'],
        'num_rows': grid_info['num_rows'],
        'num_cols': grid_info['num_cols'],
        'total_grids': grid_info['total_grids'],
        'lat_bounds': np.array(lat_bounds),
        'lon_bounds': np.array(lon_bounds),
    }

    meta_path = os.path.join(args.output_dir, "meta.npz")
    np.savez(meta_path, **meta)
    print(f"[SAVED] 메타 정보: {meta_path}")

    seg_starts_path = os.path.join(args.output_dir, "segment_starts.npy")
    np.save(seg_starts_path, np.array(segment_starts, dtype=object), allow_pickle=True)
    print(f"[SAVED] Segment starts: {seg_starts_path}")

    # 파일 크기
    data_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"\n[INFO] 데이터 파일 크기: {data_size:.1f} MB")

    print("\n" + "=" * 60)
    print("전처리 완료!")
    print(f"저장 위치: {args.output_dir}/")
    print(f"  - training_data.npz: 학습 데이터")
    print(f"  - meta.npz: 메타 정보")
    print(f"  - sea_boundary.npz: 해역 경계 및 기준 항로")
    print("=" * 60)


if __name__ == "__main__":
    main()
