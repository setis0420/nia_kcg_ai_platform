# -*- coding: utf-8 -*-
"""
V12 학습 데이터 전처리 - 선종/길이 정보 포함
============================================
- 파일명에서 선종, 길이 정보 추출
- 선종: 5개 카테고리 (화물선, 여객선, 유조선, 예부선, 기타선)
- 길이: 9개 카테고리 (40m 단위, 최대 360m)
- 입력: 30분, 출력: 60분

파일명 형식: {mmsi}_{선종}_{길이}_{폭}_{시간}_{구역}
예: 205751000_80_180_28_20180620082849_ulsan

사용법:
    python preprocess_all_regions_v12.py
    python preprocess_all_regions_v12.py --regions 울산
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial


# ============================================
# 설정
# ============================================
BASE_INPUT_DIR = r"K:\coding_project\NIA_선박항적예측프로그램\학습데이터"
BASE_OUTPUT_DIR = r"K:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리\v12"

DEFAULT_SEQ_LEN = 30    # 입력 시퀀스 길이 (30분)
DEFAULT_PRED_LEN = 60   # 예측 시퀀스 길이 (60분)
DEFAULT_STEP = 15       # 슬라이딩 윈도우 스텝
MIN_POINTS = 30         # 보간 후 최소 포인트 수


# ============================================
# 선종 카테고리 (5개)
# ============================================
def get_shiptype_category(shiptype_code):
    """
    선종 코드 → 카테고리 (5개)

    0: 화물선 (70-79)
    1: 여객선 (60-69)
    2: 유조선 (80-89)
    3: 예부선 (31-32)
    4: 기타선 (나머지)
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


SHIPTYPE_NAMES = ['화물선', '여객선', '유조선', '예부선', '기타선']


# ============================================
# 선박 길이 카테고리 (9개)
# ============================================
def get_length_category(length_m):
    """
    선박 길이 → 카테고리 (9개, 40m 단위)

    0: 0-40m
    1: 40-80m
    2: 80-120m
    3: 120-160m
    4: 160-200m
    5: 200-240m
    6: 240-280m
    7: 280-320m
    8: 320m+ (360m 이상 포함)
    """
    try:
        length = int(length_m)
    except (ValueError, TypeError):
        return 0  # 기본값

    if length <= 0:
        return 0

    category = length // 40
    return min(category, 8)  # 최대 8 (320m+)


LENGTH_NAMES = ['0-40m', '40-80m', '80-120m', '120-160m', '160-200m',
                '200-240m', '240-280m', '280-320m', '320m+']


# ============================================
# 파일명 파싱
# ============================================
def parse_filename(filename):
    """
    파일명에서 선종, 길이 추출
    형식: {mmsi}_{선종}_{길이}_{폭}_{시간}_{구역}.parquet
    예: 205751000_80_180_28_20180620082849_ulsan.parquet

    Returns:
        dict: {'shiptype': int, 'length': int} 또는 None
    """
    try:
        # 확장자 제거
        basename = os.path.basename(filename)
        name_parts = basename.replace('.parquet', '').replace('.csv', '').split('_')

        if len(name_parts) >= 4:
            shiptype_code = name_parts[1]  # 선종 코드
            length_m = name_parts[2]       # 길이 (m)

            return {
                'shiptype_code': int(shiptype_code),
                'length_m': int(length_m),
                'shiptype_cat': get_shiptype_category(shiptype_code),
                'length_cat': get_length_category(length_m)
            }
    except Exception:
        pass

    return {
        'shiptype_code': 0,
        'length_m': 0,
        'shiptype_cat': 4,  # 기타선
        'length_cat': 0     # 0-40m
    }


# ============================================
# 보간 함수
# ============================================
def interpolate_1min(df):
    """
    1분 간격으로 보간
    - lat, lon, sog: 선형 보간
    - cog: 각도 특성 고려 (sin/cos 변환 후 보간)
    """
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

    # COG 보간 (각도 특성 고려)
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


# ============================================
# 전처리 함수
# ============================================
def preprocess_trajectory(df, file_info, min_points=MIN_POINTS):
    """단일 항적 전처리 (선종/길이 정보 포함)"""
    required_cols = ['datetime', 'lat', 'lon', 'sog', 'cog']

    for col in required_cols:
        if col not in df.columns:
            return None

    df = df[required_cols].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.dropna(subset=['lat', 'lon', 'sog', 'cog'])

    if len(df) < 10:
        return None

    # SOG 필터 (0-30 knots)
    df = df[(df['sog'] >= 0) & (df['sog'] <= 30)]

    if len(df) < 10:
        return None

    df['cog'] = df['cog'] % 360

    # 1분 간격 보간
    df_interp = interpolate_1min(df)

    if df_interp is None or len(df_interp) < min_points:
        return None

    return {
        'lat': df_interp['lat'].values,
        'lon': df_interp['lon'].values,
        'sog': df_interp['sog'].values,
        'cog': df_interp['cog'].values,
        'shiptype_cat': file_info['shiptype_cat'],
        'length_cat': file_info['length_cat'],
        'shiptype_code': file_info['shiptype_code'],
        'length_m': file_info['length_m'],
        'datetime': df_interp['datetime'].values
    }


def create_sequences(data, seq_len=30, pred_len=60, step=15):
    """시퀀스 데이터 생성 (선종/길이 포함)"""
    sequences = []
    n = len(data['lat'])

    if n < seq_len + pred_len:
        return sequences

    for i in range(0, n - seq_len - pred_len + 1, step):
        # 입력: (seq_len, 4) - [lat, lon, sog, cog]
        input_seq = np.column_stack([
            data['lat'][i:i+seq_len],
            data['lon'][i:i+seq_len],
            data['sog'][i:i+seq_len],
            data['cog'][i:i+seq_len]
        ])

        # 타겟: (pred_len, 2) - [lat, lon]
        target_seq = np.column_stack([
            data['lat'][i+seq_len:i+seq_len+pred_len],
            data['lon'][i+seq_len:i+seq_len+pred_len]
        ])

        # 메타 정보: [shiptype_cat, length_cat]
        meta = np.array([data['shiptype_cat'], data['length_cat']])

        sequences.append({
            'input': input_seq.astype(np.float32),
            'target': target_seq.astype(np.float32),
            'meta': meta.astype(np.int32),
            'shiptype_cat': data['shiptype_cat'],
            'length_cat': data['length_cat']
        })

    return sequences


# ============================================
# 단일 파일 처리
# ============================================
def process_single_file(fpath, seq_len, pred_len, step, min_points):
    """단일 파일 처리 (병렬용)"""
    try:
        # 파일명에서 선종/길이 추출
        file_info = parse_filename(fpath)

        if fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath)
        else:
            try:
                df = pd.read_csv(fpath, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(fpath, encoding='cp949')
                except:
                    df = pd.read_csv(fpath, encoding='euc-kr')

        data = preprocess_trajectory(df, file_info, min_points=min_points)

        if data is None:
            return {'status': 'skipped_short', 'sequences': []}

        if len(data['lat']) < seq_len + pred_len:
            return {'status': 'skipped_short', 'sequences': []}

        sequences = create_sequences(data, seq_len, pred_len, step)

        if len(sequences) > 0:
            return {
                'status': 'success',
                'sequences': sequences,
                'shiptype_cat': data['shiptype_cat'],
                'length_cat': data['length_cat']
            }
        else:
            return {'status': 'skipped_short', 'sequences': []}

    except Exception as e:
        return {'status': 'error', 'sequences': [], 'error': str(e)}


# ============================================
# 지역별 처리
# ============================================
def process_region(region_name, input_dir, output_dir, seq_len=30, pred_len=60, step=15, min_points=30, n_workers=4):
    """단일 지역 데이터 병렬 처리"""
    print(f"\n{'='*60}")
    print(f"지역: {region_name} (워커: {n_workers}개)")
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"설정: {seq_len}분 입력 → {pred_len}분 예측, 최소 {min_points}개")
    print(f"선종 카테고리: {SHIPTYPE_NAMES}")
    print(f"길이 카테고리: {LENGTH_NAMES}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 파일 목록 수집
    all_files = []
    month_dirs = sorted(glob.glob(os.path.join(input_dir, '*')))

    for month_dir in month_dirs:
        if os.path.isdir(month_dir):
            parquet_files = glob.glob(os.path.join(month_dir, '*.parquet'))
            csv_files = glob.glob(os.path.join(month_dir, '*.csv'))
            all_files.extend(parquet_files + csv_files)
        elif month_dir.endswith('.parquet') or month_dir.endswith('.csv'):
            all_files.append(month_dir)

    print(f"  총 파일 수: {len(all_files):,}")

    if not all_files:
        return None

    stats = {
        'region': region_name,
        'total_files': len(all_files),
        'processed_files': 0,
        'total_sequences': 0,
        'skipped_short': 0,
        'skipped_error': 0,
        'shiptype_dist': {i: 0 for i in range(5)},
        'length_dist': {i: 0 for i in range(9)}
    }

    chunk_idx = 0
    current_chunk = []
    chunk_size = 300000

    # 병렬 처리
    process_func = partial(process_single_file, seq_len=seq_len, pred_len=pred_len, step=step, min_points=min_points)

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, all_files),
            total=len(all_files),
            desc=f"{region_name} 처리"
        ))

    # 결과 집계
    for result in results:
        if result['status'] == 'success':
            current_chunk.extend(result['sequences'])
            stats['processed_files'] += 1
            stats['total_sequences'] += len(result['sequences'])

            st = result['shiptype_cat']
            lc = result['length_cat']
            stats['shiptype_dist'][st] = stats['shiptype_dist'].get(st, 0) + 1
            stats['length_dist'][lc] = stats['length_dist'].get(lc, 0) + 1

            # 청크 저장
            if len(current_chunk) >= chunk_size:
                chunk_file = os.path.join(output_dir, f'sequences_chunk_{chunk_idx:03d}.pkl')
                with open(chunk_file, 'wb') as f:
                    pickle.dump(current_chunk, f)
                print(f"  -> 청크 {chunk_idx} 저장: {len(current_chunk):,} 시퀀스")
                current_chunk = []
                chunk_idx += 1

        elif result['status'] == 'skipped_short':
            stats['skipped_short'] += 1
        else:
            stats['skipped_error'] += 1

    # 남은 데이터 저장
    if len(current_chunk) > 0:
        chunk_file = os.path.join(output_dir, f'sequences_chunk_{chunk_idx:03d}.pkl')
        with open(chunk_file, 'wb') as f:
            pickle.dump(current_chunk, f)
        print(f"  -> 청크 {chunk_idx} 저장: {len(current_chunk):,} 시퀀스")
        chunk_idx += 1

    stats['num_chunks'] = chunk_idx
    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    # 결과 출력
    print(f"\n--- {region_name} 완료 ---")
    print(f"  처리: {stats['processed_files']:,} / {stats['total_files']:,}")
    print(f"  시퀀스: {stats['total_sequences']:,}")
    print(f"  청크: {chunk_idx}개")

    print(f"\n  선종 분포:")
    for i, name in enumerate(SHIPTYPE_NAMES):
        count = stats['shiptype_dist'].get(i, 0)
        print(f"    {name}: {count:,}개")

    print(f"\n  길이 분포:")
    for i, name in enumerate(LENGTH_NAMES):
        count = stats['length_dist'].get(i, 0)
        print(f"    {name}: {count:,}개")

    return stats


# ============================================
# 메인
# ============================================
def main():
    parser = argparse.ArgumentParser(description='V12 전처리 - 선종/길이 포함')
    parser.add_argument('--regions', nargs='+', default=None,
                        help='처리할 지역 목록 (기본: 전체)')
    parser.add_argument('--list', action='store_true',
                        help='사용 가능한 지역 목록 출력')
    parser.add_argument('--seq_len', type=int, default=DEFAULT_SEQ_LEN,
                        help=f'입력 시퀀스 길이 (기본: {DEFAULT_SEQ_LEN})')
    parser.add_argument('--pred_len', type=int, default=DEFAULT_PRED_LEN,
                        help=f'예측 시퀀스 길이 (기본: {DEFAULT_PRED_LEN})')
    parser.add_argument('--step', type=int, default=DEFAULT_STEP,
                        help=f'슬라이딩 윈도우 스텝 (기본: {DEFAULT_STEP})')
    parser.add_argument('--min_points', type=int, default=MIN_POINTS,
                        help=f'최소 포인트 수 (기본: {MIN_POINTS})')
    parser.add_argument('--workers', type=int, default=4,
                        help='병렬 처리 워커 수 (기본: 4)')
    args = parser.parse_args()

    # 지역 목록 확인
    available_regions = []
    for d in os.listdir(BASE_INPUT_DIR):
        if os.path.isdir(os.path.join(BASE_INPUT_DIR, d)):
            available_regions.append(d)

    if args.list:
        print("사용 가능한 지역:")
        for r in sorted(available_regions):
            input_dir = os.path.join(BASE_INPUT_DIR, r)
            file_count = sum(1 for _ in glob.glob(os.path.join(input_dir, '**', '*.parquet'), recursive=True))
            file_count += sum(1 for _ in glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True))
            print(f"  - {r}: {file_count:,}개 파일")
        return

    # 처리할 지역
    if args.regions:
        regions = args.regions
    else:
        regions = available_regions

    print("=" * 60)
    print("V12 학습 데이터 전처리 - 선종/길이 포함")
    print("=" * 60)
    print(f"입력: {BASE_INPUT_DIR}")
    print(f"출력: {BASE_OUTPUT_DIR}")
    print(f"설정: {args.seq_len}분 입력 → {args.pred_len}분 예측")
    print(f"선종: {SHIPTYPE_NAMES}")
    print(f"길이: {LENGTH_NAMES}")
    print(f"지역: {regions}")
    print("=" * 60)

    # 각 지역 처리
    all_stats = []
    for region in regions:
        input_dir = os.path.join(BASE_INPUT_DIR, region)
        output_dir = os.path.join(BASE_OUTPUT_DIR, region)

        if not os.path.exists(input_dir):
            print(f"\n[경고] {region} 폴더가 없습니다: {input_dir}")
            continue

        stats = process_region(
            region, input_dir, output_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            step=args.step,
            min_points=args.min_points,
            n_workers=args.workers
        )
        if stats:
            all_stats.append(stats)

    # 최종 요약
    if all_stats:
        print("\n" + "=" * 60)
        print("전체 요약")
        print("=" * 60)
        total_files = sum(s['total_files'] for s in all_stats)
        total_processed = sum(s['processed_files'] for s in all_stats)
        total_sequences = sum(s['total_sequences'] for s in all_stats)

        print(f"{'지역':<15} {'총 파일':>12} {'처리됨':>12} {'시퀀스':>15}")
        print("-" * 55)
        for s in all_stats:
            print(f"{s['region']:<15} {s['total_files']:>12,} {s['processed_files']:>12,} {s['total_sequences']:>15,}")
        print("-" * 55)
        print(f"{'합계':<15} {total_files:>12,} {total_processed:>12,} {total_sequences:>15,}")

    print("\n완료!")


if __name__ == "__main__":
    main()
