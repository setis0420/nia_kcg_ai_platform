"""
울산 항적 데이터 전처리 스크립트
- 선종(shiptype), 길이(20m 단위)를 학습 피처로 포함
- 월별 청크로 저장하여 메모리 효율 개선
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path


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
    # 필요한 컬럼만 선택
    df = df[['datetime', 'lat', 'lon', 'sog', 'cog', 'shiptype', 'length']].copy()

    # 시간순 정렬
    df = df.sort_values('datetime').reset_index(drop=True)

    # 결측치 제거
    df = df.dropna(subset=['lat', 'lon', 'sog', 'cog'])

    if len(df) < 20:  # 최소 20개 포인트 필요
        return None

    # SOG 이상치 제거 (0~30 knots)
    df = df[(df['sog'] >= 0) & (df['sog'] <= 30)]

    # COG 정규화 (0~360)
    df['cog'] = df['cog'] % 360

    # 선종, 길이 카테고리화
    shiptype_cat = get_shiptype_category(df['shiptype'].iloc[0])
    length_cat = get_length_category(df['length'].iloc[0])

    # 피처 생성
    result = {
        'lat': df['lat'].values,
        'lon': df['lon'].values,
        'sog': df['sog'].values,
        'cog': df['cog'].values,
        'shiptype_cat': shiptype_cat,
        'length_cat': length_cat,
        'datetime': df['datetime'].values
    }

    return result


def create_sequences(data, seq_len=50, pred_len=20, step=10):
    """시퀀스 데이터 생성 - step을 늘려서 데이터양 줄임"""
    sequences = []

    n = len(data['lat'])
    if n < seq_len + pred_len:
        return sequences

    for i in range(0, n - seq_len - pred_len + 1, step):
        # 입력 시퀀스: lat, lon, sog, cog
        input_seq = np.column_stack([
            data['lat'][i:i+seq_len],
            data['lon'][i:i+seq_len],
            data['sog'][i:i+seq_len],
            data['cog'][i:i+seq_len]
        ])

        # 타겟 시퀀스: lat, lon
        target_seq = np.column_stack([
            data['lat'][i+seq_len:i+seq_len+pred_len],
            data['lon'][i+seq_len:i+seq_len+pred_len]
        ])

        # 메타 피처: 선종, 길이
        meta = np.array([data['shiptype_cat'], data['length_cat']])

        sequences.append({
            'input': input_seq.astype(np.float32),
            'target': target_seq.astype(np.float32),
            'meta': meta.astype(np.int32)
        })

    return sequences


def process_all_data(input_dir, output_dir, seq_len=50, pred_len=20, step=10):
    """전체 데이터 처리 - 청크 단위로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # 월별 폴더 목록
    month_dirs = sorted(glob.glob(os.path.join(input_dir, '*')))

    stats = {
        'total_files': 0,
        'processed_files': 0,
        'total_sequences': 0,
        'shiptype_dist': {},
        'length_dist': {}
    }

    chunk_idx = 0
    current_chunk = []
    chunk_size = 500000  # 50만 시퀀스마다 저장

    for month_dir in tqdm(month_dirs, desc="Processing months"):
        if not os.path.isdir(month_dir):
            continue

        month_name = os.path.basename(month_dir)
        parquet_files = glob.glob(os.path.join(month_dir, '*.parquet'))

        for pfile in tqdm(parquet_files, desc=f"  {month_name}", leave=False):
            stats['total_files'] += 1

            try:
                df = pd.read_parquet(pfile)
                data = preprocess_trajectory(df)

                if data is None:
                    continue

                sequences = create_sequences(data, seq_len, pred_len, step)

                if len(sequences) > 0:
                    current_chunk.extend(sequences)
                    stats['processed_files'] += 1
                    stats['total_sequences'] += len(sequences)

                    # 통계 업데이트
                    st = data['shiptype_cat']
                    lc = data['length_cat']
                    stats['shiptype_dist'][st] = stats['shiptype_dist'].get(st, 0) + 1
                    stats['length_dist'][lc] = stats['length_dist'].get(lc, 0) + 1

                    # 청크 크기 도달 시 저장
                    if len(current_chunk) >= chunk_size:
                        chunk_file = os.path.join(output_dir, f'sequences_chunk_{chunk_idx:03d}.pkl')
                        with open(chunk_file, 'wb') as f:
                            pickle.dump(current_chunk, f)
                        print(f"\n청크 {chunk_idx} 저장: {len(current_chunk)} 시퀀스")
                        current_chunk = []
                        chunk_idx += 1

            except Exception as e:
                continue

    # 남은 데이터 저장
    if len(current_chunk) > 0:
        chunk_file = os.path.join(output_dir, f'sequences_chunk_{chunk_idx:03d}.pkl')
        with open(chunk_file, 'wb') as f:
            pickle.dump(current_chunk, f)
        print(f"\n청크 {chunk_idx} 저장: {len(current_chunk)} 시퀀스")

    # 통계 저장
    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    # 결과 출력
    print(f"\n=== 전처리 완료 ===")
    print(f"총 파일: {stats['total_files']}")
    print(f"처리된 파일: {stats['processed_files']}")
    print(f"생성된 시퀀스: {stats['total_sequences']}")
    print(f"저장된 청크: {chunk_idx + 1}개")
    print(f"\n선종 분포: {stats['shiptype_dist']}")
    print(f"길이 분포: {stats['length_dist']}")
    print(f"\n저장 경로: {output_dir}")

    return stats


if __name__ == "__main__":
    input_dir = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터\울산"
    output_dir = r"k:\coding_project\NIA_선박항적예측프로그램\학습데이터_전처리완료\울산"

    process_all_data(
        input_dir=input_dir,
        output_dir=output_dir,
        seq_len=50,      # 입력 시퀀스 길이
        pred_len=20,     # 예측 시퀀스 길이
        step=10          # 슬라이딩 윈도우 스텝 (5->10으로 증가)
    )
