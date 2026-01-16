# -*- coding: utf-8 -*-
"""
V10 학습 데이터 전처리 - 모든 지역 병렬 처리
============================================
- 학습데이터 폴더 내 모든 지역(울산, 부산, 군산, ...) 자동 탐지
- 각 지역별로 100분 예측용 데이터 생성
- 멀티프로세싱으로 병렬 처리

사용법:
    python preprocess_all_regions_v10.py
    python preprocess_all_regions_v10.py --regions 부산 울산  # 특정 지역만
    python preprocess_all_regions_v10.py --workers 4  # 워커 수 지정
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count, Process, Queue
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================
# 설정
# ============================================
BASE_INPUT_DIR = r"K:\coding_project\NIA_선박항적예측프로그램\학습데이터"
BASE_OUTPUT_DIR = r"K:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리\v10"

DEFAULT_SEQ_LEN = 50    # 입력 시퀀스 길이 (50분)
DEFAULT_PRED_LEN = 100  # 예측 시퀀스 길이 (100분)
DEFAULT_STEP = 20       # 슬라이딩 윈도우 스텝


# ============================================
# 유틸리티 함수
# ============================================

def get_length_category(length):
    """길이를 20m 단위로 카테고리화"""
    if pd.isna(length) or length <= 0:
        return 0  # unknown
    return min(int(length // 20), 15)  # 0~15 (300m 이상은 15)


def get_shiptype_category(shiptype):
    """선종 카테고리화 (주요 선종별)"""
    if pd.isna(shiptype):
        return 0
    shiptype = int(shiptype)
    # AIS 선종 코드 그룹화
    if shiptype in [70, 71, 72, 73, 74, 79]:  # 화물선
        return 1
    elif shiptype in [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]:  # 유조선
        return 2
    elif shiptype in [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:  # 여객선
        return 3
    elif shiptype == 30:  # 어선
        return 4
    elif shiptype in [31, 32, 33, 34, 35, 36, 37]:  # 예인선/특수선
        return 5
    elif shiptype in [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]:  # 기타
        return 6
    else:
        return 0  # unknown


def preprocess_trajectory(df):
    """단일 항적 전처리"""
    required_cols = ['datetime', 'lat', 'lon', 'sog', 'cog']
    optional_cols = ['shiptype', 'length']

    for col in required_cols:
        if col not in df.columns:
            return None

    cols_to_use = required_cols.copy()
    for col in optional_cols:
        if col in df.columns:
            cols_to_use.append(col)

    df = df[cols_to_use].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.dropna(subset=['lat', 'lon', 'sog', 'cog'])

    if len(df) < 50:
        return None

    df = df[(df['sog'] >= 0) & (df['sog'] <= 30)]

    if len(df) < 50:
        return None

    df['cog'] = df['cog'] % 360

    shiptype_cat = 0
    length_cat = 0
    if 'shiptype' in df.columns:
        shiptype_cat = get_shiptype_category(df['shiptype'].iloc[0])
    if 'length' in df.columns:
        length_cat = get_length_category(df['length'].iloc[0])

    return {
        'lat': df['lat'].values,
        'lon': df['lon'].values,
        'sog': df['sog'].values,
        'cog': df['cog'].values,
        'shiptype_cat': shiptype_cat,
        'length_cat': length_cat,
        'datetime': df['datetime'].values
    }


def create_sequences(data, seq_len=50, pred_len=100, step=20):
    """시퀀스 데이터 생성"""
    sequences = []
    n = len(data['lat'])

    if n < seq_len + pred_len:
        return sequences

    for i in range(0, n - seq_len - pred_len + 1, step):
        input_seq = np.column_stack([
            data['lat'][i:i+seq_len],
            data['lon'][i:i+seq_len],
            data['sog'][i:i+seq_len],
            data['cog'][i:i+seq_len]
        ])

        target_seq = np.column_stack([
            data['lat'][i+seq_len:i+seq_len+pred_len],
            data['lon'][i+seq_len:i+seq_len+pred_len]
        ])

        meta = np.array([data['shiptype_cat'], data['length_cat']])

        sequences.append({
            'input': input_seq.astype(np.float32),
            'target': target_seq.astype(np.float32),
            'meta': meta.astype(np.int32)
        })

    return sequences


def process_single_file(fpath, seq_len, pred_len, step):
    """단일 파일 처리 (병렬용)"""
    try:
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

        data = preprocess_trajectory(df)

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


def process_region(region_name, input_dir, output_dir, seq_len=50, pred_len=100, step=20, n_workers=4):
    """단일 지역 데이터 병렬 처리"""
    print(f"\n{'='*60}")
    print(f"지역: {region_name} (워커: {n_workers}개)")
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
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
        return {
            'region': region_name,
            'total_files': 0,
            'processed_files': 0,
            'total_sequences': 0,
            'skipped_short': 0,
            'skipped_error': 0,
            'shiptype_dist': {},
            'length_dist': {},
            'num_chunks': 0
        }

    stats = {
        'region': region_name,
        'total_files': len(all_files),
        'processed_files': 0,
        'total_sequences': 0,
        'skipped_short': 0,
        'skipped_error': 0,
        'shiptype_dist': {},
        'length_dist': {}
    }

    chunk_idx = 0
    current_chunk = []
    chunk_size = 300000

    # 병렬 처리
    process_func = partial(process_single_file, seq_len=seq_len, pred_len=pred_len, step=step)

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

    print(f"\n--- {region_name} 완료 ---")
    print(f"  처리: {stats['processed_files']:,} / {stats['total_files']:,}")
    print(f"  시퀀스: {stats['total_sequences']:,}")
    print(f"  청크: {chunk_idx}개")

    return stats


def get_available_regions(base_input_dir):
    """사용 가능한 지역 목록 반환"""
    regions = []
    for item in os.listdir(base_input_dir):
        item_path = os.path.join(base_input_dir, item)
        if os.path.isdir(item_path):
            regions.append(item)
    return sorted(regions)


def process_region_wrapper(args_tuple):
    """지역 병렬 처리용 래퍼"""
    region, input_dir, output_dir, seq_len, pred_len, step = args_tuple
    return process_region(
        region_name=region,
        input_dir=input_dir,
        output_dir=output_dir,
        seq_len=seq_len,
        pred_len=pred_len,
        step=step,
        n_workers=2  # 지역당 워커 수 (지역 병렬이므로 줄임)
    )


def main():
    parser = argparse.ArgumentParser(
        description="V10 학습 데이터 전처리 - 모든 지역 병렬 처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_base", type=str, default=BASE_INPUT_DIR,
                        help="입력 데이터 기본 폴더")
    parser.add_argument("--output_base", type=str, default=BASE_OUTPUT_DIR,
                        help="출력 데이터 기본 폴더")
    parser.add_argument("--regions", nargs='*', default=None,
                        help="처리할 지역 목록 (미지정시 전체)")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN,
                        help="입력 시퀀스 길이 (기본: 50분)")
    parser.add_argument("--pred_len", type=int, default=DEFAULT_PRED_LEN,
                        help="예측 시퀀스 길이 (기본: 100분)")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help="슬라이딩 윈도우 스텝 (기본: 20)")
    parser.add_argument("--region_workers", type=int, default=4,
                        help="동시 처리할 지역 수 (기본: 4)")
    parser.add_argument("--list", action="store_true",
                        help="사용 가능한 지역 목록만 출력")

    args = parser.parse_args()

    available_regions = get_available_regions(args.input_base)

    if args.list:
        print("사용 가능한 지역 목록:")
        for region in available_regions:
            print(f"  - {region}")
        return

    if args.regions:
        regions_to_process = args.regions
        for region in regions_to_process:
            if region not in available_regions:
                print(f"[경고] '{region}' 지역이 존재하지 않습니다.")
        regions_to_process = [r for r in regions_to_process if r in available_regions]
    else:
        regions_to_process = available_regions

    if not regions_to_process:
        print("[오류] 처리할 지역이 없습니다.")
        return

    print("=" * 60)
    print("V10 학습 데이터 전처리 - 지역 병렬 처리")
    print("=" * 60)
    print(f"입력: {args.input_base}")
    print(f"출력: {args.output_base}")
    print(f"설정: {args.seq_len}분 입력 → {args.pred_len}분 예측")
    print(f"지역 병렬: {args.region_workers}개 동시 처리")
    print(f"지역: {', '.join(regions_to_process)}")
    print()

    # 지역별 인자 준비
    region_args = []
    for region in regions_to_process:
        input_dir = os.path.join(args.input_base, region)
        output_dir = os.path.join(args.output_base, region)
        region_args.append((region, input_dir, output_dir, args.seq_len, args.pred_len, args.step))

    # 지역 병렬 처리
    all_stats = {}
    with ProcessPoolExecutor(max_workers=args.region_workers) as executor:
        futures = {executor.submit(process_region_wrapper, arg): arg[0] for arg in region_args}

        for future in as_completed(futures):
            region = futures[future]
            try:
                stats = future.result()
                all_stats[region] = stats
                print(f"\n[완료] {region}: {stats['total_sequences']:,} 시퀀스")
            except Exception as e:
                print(f"\n[에러] {region}: {e}")
                all_stats[region] = {'total_files': 0, 'processed_files': 0, 'total_sequences': 0}

    # 전체 요약
    print("\n" + "=" * 60)
    print("전체 요약")
    print("=" * 60)

    total_files = 0
    total_processed = 0
    total_sequences = 0

    print(f"\n{'지역':<10} {'총 파일':>12} {'처리됨':>12} {'시퀀스':>15}")
    print("-" * 52)

    for region in regions_to_process:
        stats = all_stats.get(region, {'total_files': 0, 'processed_files': 0, 'total_sequences': 0})
        total_files += stats['total_files']
        total_processed += stats['processed_files']
        total_sequences += stats['total_sequences']
        print(f"{region:<10} {stats['total_files']:>12,} {stats['processed_files']:>12,} {stats['total_sequences']:>15,}")

    print("-" * 52)
    print(f"{'합계':<10} {total_files:>12,} {total_processed:>12,} {total_sequences:>15,}")
    print("=" * 60)

    summary = {
        'regions': list(all_stats.keys()),
        'total_files': total_files,
        'total_processed': total_processed,
        'total_sequences': total_sequences,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'step': args.step,
        'region_stats': all_stats
    }

    summary_file = os.path.join(args.output_base, 'all_regions_summary.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    print(f"\n요약 저장: {summary_file}")


if __name__ == "__main__":
    main()
