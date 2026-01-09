# -*- coding: utf-8 -*-
"""
울산 선박 항적 데이터 전처리
===========================
- H3 Hexagonal Grid 기반 시퀀스 생성
- 선종(shiptype), 길이(length) 임베딩 추가
- SOG, COG 임베딩 유지

입력: 학습데이터/울산/{yyyymm}/*.parquet
출력: 학습데이터_전처리완료/울산/{mmsi}.parquet
"""

import os
import sys
import pandas as pd
import numpy as np
import h3
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

# ============== 설정 ==============
INPUT_BASE = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터\울산"
OUTPUT_BASE = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"

H3_RESOLUTION = 9  # ~700m 직경
MIN_SOG = 8.0      # 최소 속력 (knots)
MAX_SOG = 21.0     # 최대 속력 (knots)
SEQ_LEN = 50       # 시퀀스 길이 (분)
MIN_POINTS = 60    # 최소 포인트 수 (1시간)

# 임베딩 구간 설정
NUM_SOG_BINS = 14       # SOG: 8~21+ knots, 1 knot 단위
NUM_COG_BINS = 24       # COG: 0~360도, 15도 단위
NUM_SHIPTYPE_BINS = 10  # 선종 카테고리 (0~9)
NUM_LENGTH_BINS = 20    # 길이: 0~400m, 20m 단위


def sog_to_bin(sog):
    """SOG를 구간 인덱스로 변환 (8-9, 9-10, ..., 21+)"""
    if sog < MIN_SOG:
        return 0
    bin_idx = int(sog - MIN_SOG)
    return min(bin_idx, NUM_SOG_BINS - 1)


def cog_to_bin(cog):
    """COG를 15도 단위 구간 인덱스로 변환"""
    cog = cog % 360
    return int(cog / 15) % NUM_COG_BINS


def shiptype_to_bin(shiptype):
    """
    선종 코드를 카테고리로 변환
    AIS 선종 코드 기준:
    - 0: Unknown
    - 1: 예약
    - 2: WIG (Wing In Ground)
    - 3: 특수 선박 (30-39)
    - 4: 고속선 (40-49)
    - 5: 특수 목적 (50-59)
    - 6: 여객선 (60-69)
    - 7: 화물선 (70-79)
    - 8: 탱커 (80-89)
    - 9: 기타 (90-99)
    """
    if pd.isna(shiptype) or shiptype <= 0:
        return 0

    shiptype = int(shiptype)
    if shiptype < 30:
        return min(shiptype // 10, 2)  # 0, 1, 2
    else:
        return shiptype // 10  # 3~9


def length_to_bin(length):
    """길이를 20m 단위 구간 인덱스로 변환 (0-20, 20-40, ..., 380-400+)"""
    if pd.isna(length) or length <= 0:
        return 0
    bin_idx = int(length / 20)
    return min(bin_idx, NUM_LENGTH_BINS - 1)


def interpolate_1min(df):
    """1분 간격 보간 (COG 포함)"""
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


def process_single_file(filepath):
    """단일 파일 처리"""
    try:
        # 파일명에서 메타데이터 추출: {mmsi}_{선종}_{길이}_{폭}_{수신일}_ulsan.parquet
        filename = os.path.basename(filepath)
        parts = filename.replace('_ulsan.parquet', '').split('_')

        if len(parts) < 5:
            return None

        mmsi = int(parts[0])
        shiptype = int(parts[1]) if parts[1].isdigit() else 0
        length = float(parts[2]) if parts[2].replace('.', '').isdigit() else 0

        # 데이터 로드
        df = pd.read_parquet(filepath)

        if len(df) < MIN_POINTS:
            return None

        # 필수 컬럼 확인
        for col in ['datetime', 'lat', 'lon', 'sog', 'cog']:
            if col not in df.columns:
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce') if col != 'datetime' else pd.to_datetime(df[col])

        df = df.dropna(subset=['datetime', 'lat', 'lon', 'sog', 'cog'])

        # 속력 필터링 (8 knots 이상)
        df_fast = df[df['sog'] >= MIN_SOG].copy()

        if len(df_fast) < MIN_POINTS:
            return None

        # 1분 간격 보간
        df_interp = interpolate_1min(df_fast)

        if df_interp is None or len(df_interp) < SEQ_LEN + 1:
            return None

        # H3 셀 변환
        df_interp['h3_cell'] = df_interp.apply(
            lambda row: h3.latlng_to_cell(row['lat'], row['lon'], H3_RESOLUTION), axis=1
        )

        # 구간(bin) 변환
        df_interp['sog_bin'] = df_interp['sog'].apply(sog_to_bin)
        df_interp['cog_bin'] = df_interp['cog'].apply(cog_to_bin)
        df_interp['shiptype_bin'] = shiptype_to_bin(shiptype)
        df_interp['length_bin'] = length_to_bin(length)

        # 메타데이터 추가
        df_interp['mmsi'] = mmsi
        df_interp['shiptype'] = shiptype
        df_interp['length'] = length

        return df_interp

    except Exception as e:
        return None


def process_folder(folder_path):
    """폴더 내 모든 파일 처리"""
    files = glob(os.path.join(folder_path, '*.parquet'))
    results = []

    for filepath in files:
        result = process_single_file(filepath)
        if result is not None:
            results.append(result)

    return results


def main():
    print("=" * 60)
    print("울산 선박 항적 데이터 전처리")
    print("=" * 60)
    print(f"입력: {INPUT_BASE}")
    print(f"출력: {OUTPUT_BASE}")
    print(f"H3 Resolution: {H3_RESOLUTION} (~700m)")
    print(f"최소 속력: {MIN_SOG} knots")
    print(f"시퀀스 길이: {SEQ_LEN}분")
    print()

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 입력 폴더 목록
    folders = sorted([f for f in os.listdir(INPUT_BASE)
                     if os.path.isdir(os.path.join(INPUT_BASE, f))])

    print(f"처리할 폴더: {len(folders)}개")
    print(f"폴더 목록: {folders[:5]}...{folders[-3:]}")
    print()

    # MMSI별 데이터 수집
    mmsi_data = {}
    total_files = 0
    processed_files = 0

    for folder in tqdm(folders, desc="폴더 처리"):
        folder_path = os.path.join(INPUT_BASE, folder)
        files = glob(os.path.join(folder_path, '*.parquet'))
        total_files += len(files)

        for filepath in tqdm(files, desc=f"  {folder}", leave=False):
            result = process_single_file(filepath)
            if result is not None:
                processed_files += 1
                mmsi = result['mmsi'].iloc[0]

                if mmsi not in mmsi_data:
                    mmsi_data[mmsi] = []
                mmsi_data[mmsi].append(result)

    print(f"\n전체 파일: {total_files}개")
    print(f"처리된 파일: {processed_files}개 ({processed_files/total_files*100:.1f}%)")
    print(f"고유 MMSI: {len(mmsi_data)}개")

    # MMSI별로 저장
    print("\nMMSI별 데이터 저장 중...")

    saved_count = 0
    total_rows = 0

    for mmsi, dfs in tqdm(mmsi_data.items(), desc="저장"):
        # 모든 데이터 병합
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('datetime').reset_index(drop=True)

        # 최소 포인트 수 확인
        if len(combined) < SEQ_LEN + 30:
            continue

        # 저장
        output_path = os.path.join(OUTPUT_BASE, f"{mmsi}.parquet")
        combined.to_parquet(output_path, index=False)

        saved_count += 1
        total_rows += len(combined)

    print()
    print("=" * 60)
    print("전처리 완료")
    print("=" * 60)
    print(f"저장된 MMSI: {saved_count}개")
    print(f"총 레코드 수: {total_rows:,}개")
    print(f"출력 경로: {OUTPUT_BASE}")

    # 통계 저장
    stats = {
        'total_files': total_files,
        'processed_files': processed_files,
        'unique_mmsi': len(mmsi_data),
        'saved_mmsi': saved_count,
        'total_rows': total_rows,
        'h3_resolution': H3_RESOLUTION,
        'min_sog': MIN_SOG,
        'seq_len': SEQ_LEN,
        'num_sog_bins': NUM_SOG_BINS,
        'num_cog_bins': NUM_COG_BINS,
        'num_shiptype_bins': NUM_SHIPTYPE_BINS,
        'num_length_bins': NUM_LENGTH_BINS,
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(OUTPUT_BASE, '_preprocessing_stats.csv'), index=False)
    print(f"\n통계 파일: {os.path.join(OUTPUT_BASE, '_preprocessing_stats.csv')}")


if __name__ == "__main__":
    main()
