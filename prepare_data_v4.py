# -*- coding: utf-8 -*-
"""
격자 기반 학습 데이터 전처리 (V4)
=================================
- 격자 크기: 0.001도 (약 111m)
- 격자 ID: 문자열 "lat_idx_lon_idx" (예: "34587_128045")
- 입력: 과거 seq_len개 (격자ID, SOG, COG)
- 출력: 미래 pred_len개 (격자ID, SOG, COG)

사용법:
    python prepare_data_v4.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                               --transition_folder "area_transition_results" \
                               --output_dir "prepared_data_v4" \
                               --seq_len 50 --pred_len 10 --stride 3
"""

import os
import json
import gc
import argparse
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# 격자 크기 (도 단위)
GRID_SIZE = 0.002  # 약 222m


# ============================================
# 격자 변환 유틸리티
# ============================================

def latlon_to_grid_id(lat: float, lon: float) -> str:
    """위경도 → 격자 ID (문자열)

    예: (34.587, 128.045) → "34587_128045"
    """
    lat_idx = int(lat / GRID_SIZE)
    lon_idx = int(lon / GRID_SIZE)
    return f"{lat_idx}_{lon_idx}"


def grid_id_to_latlon(grid_id: str) -> tuple:
    """격자 ID → 위경도 (격자 중심점)

    예: "34587_128045" → (34.5875, 128.0455)
    """
    lat_idx, lon_idx = map(int, grid_id.split('_'))
    lat = (lat_idx + 0.5) * GRID_SIZE
    lon = (lon_idx + 0.5) * GRID_SIZE
    return lat, lon


class GridVocabulary:
    """격자 ID ↔ 정수 인덱스 변환 관리

    특수 토큰:
    - <PAD>: 0 (패딩)
    - <UNK>: 1 (미등록 격자)
    """

    def __init__(self):
        self.grid_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_grid = {0: '<PAD>', 1: '<UNK>'}
        self.next_idx = 2
        self.grid_counts = defaultdict(int)

    def add_grid(self, grid_id: str):
        """격자 추가"""
        self.grid_counts[grid_id] += 1
        if grid_id not in self.grid_to_idx:
            self.grid_to_idx[grid_id] = self.next_idx
            self.idx_to_grid[self.next_idx] = grid_id
            self.next_idx += 1

    def encode(self, grid_id: str) -> int:
        """격자 ID → 정수 인덱스"""
        return self.grid_to_idx.get(grid_id, 1)  # 미등록은 <UNK>

    def decode(self, idx: int) -> str:
        """정수 인덱스 → 격자 ID"""
        return self.idx_to_grid.get(idx, '<UNK>')

    def encode_sequence(self, grid_ids: list) -> np.ndarray:
        """격자 ID 시퀀스 → 정수 인덱스 시퀀스"""
        return np.array([self.encode(g) for g in grid_ids], dtype=np.int64)

    def truncate_to_top_k(self, top_k: int):
        """빈도 상위 top_k 격자만 유지, 나머지는 <UNK>로 처리

        Args:
            top_k: 유지할 격자 수 (특수 토큰 제외)
        """
        if top_k <= 0 or top_k >= len(self.grid_counts):
            return  # 제한 없음

        # 빈도순 정렬
        sorted_grids = sorted(
            self.grid_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 상위 top_k만 유지
        top_grids = set(g for g, _ in sorted_grids[:top_k])

        # 새 vocabulary 구축
        new_grid_to_idx = {'<PAD>': 0, '<UNK>': 1}
        new_idx_to_grid = {0: '<PAD>', 1: '<UNK>'}
        next_idx = 2

        for grid_id, count in sorted_grids[:top_k]:
            new_grid_to_idx[grid_id] = next_idx
            new_idx_to_grid[next_idx] = grid_id
            next_idx += 1

        self.grid_to_idx = new_grid_to_idx
        self.idx_to_grid = new_idx_to_grid
        self.next_idx = next_idx

        # 제거된 격자 수 계산
        removed = len(self.grid_counts) - top_k
        print(f"  - Top-K 제한: {len(self.grid_counts)} → {top_k} (제거: {removed})")

    def expand_neighbors(self, n_expand: int = 1):
        """인접 격자로 vocabulary 확장 (OOV 대응)"""
        existing_grids = [g for g in self.grid_to_idx.keys()
                         if g not in ['<PAD>', '<UNK>']]

        for grid_id in existing_grids:
            lat_idx, lon_idx = map(int, grid_id.split('_'))
            for dlat in range(-n_expand, n_expand + 1):
                for dlon in range(-n_expand, n_expand + 1):
                    if dlat == 0 and dlon == 0:
                        continue
                    new_grid = f"{lat_idx + dlat}_{lon_idx + dlon}"
                    if new_grid not in self.grid_to_idx:
                        self.grid_to_idx[new_grid] = self.next_idx
                        self.idx_to_grid[self.next_idx] = new_grid
                        self.next_idx += 1

    def __len__(self):
        return len(self.grid_to_idx)

    def save(self, path: str):
        """vocabulary 저장"""
        data = {
            'grid_to_idx': self.grid_to_idx,
            'idx_to_grid': {str(k): v for k, v in self.idx_to_grid.items()},
            'grid_counts': dict(self.grid_counts),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """vocabulary 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = cls()
        vocab.grid_to_idx = data['grid_to_idx']
        vocab.idx_to_grid = {int(k): v for k, v in data['idx_to_grid'].items()}
        vocab.grid_counts = defaultdict(int, data.get('grid_counts', {}))
        vocab.next_idx = max(vocab.idx_to_grid.keys()) + 1
        return vocab


# ============================================
# 데이터 전처리 함수
# ============================================

def interpolate_trajectory(df: pd.DataFrame, dt_minutes: int = 1) -> pd.DataFrame:
    """항적 데이터를 일정 시간 간격으로 보간

    Args:
        df: 원본 항적 (datetime, lat, lon, sog, cog 컬럼 필요)
        dt_minutes: 보간 간격 (분)

    Returns:
        보간된 항적 DataFrame
    """
    if len(df) < 2:
        return None

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 시간 범위
    start_time = df['datetime'].iloc[0]
    end_time = df['datetime'].iloc[-1]

    # 일정 간격 시간 생성
    time_range = pd.date_range(start=start_time, end=end_time, freq=f'{dt_minutes}min')

    if len(time_range) < 2:
        return None

    # 숫자형 시간 인덱스
    df['_time_num'] = (df['datetime'] - start_time).dt.total_seconds()
    target_times = (time_range - start_time).total_seconds().values

    # 선형 보간
    lat_interp = np.interp(target_times, df['_time_num'].values, df['lat'].values)
    lon_interp = np.interp(target_times, df['_time_num'].values, df['lon'].values)
    sog_interp = np.interp(target_times, df['_time_num'].values, df['sog'].values)

    # COG는 각도이므로 sin/cos로 보간 후 복원
    cog_rad = np.radians(df['cog'].values)
    sin_cog_interp = np.interp(target_times, df['_time_num'].values, np.sin(cog_rad))
    cos_cog_interp = np.interp(target_times, df['_time_num'].values, np.cos(cog_rad))
    cog_interp = np.degrees(np.arctan2(sin_cog_interp, cos_cog_interp))
    cog_interp = (cog_interp + 360) % 360

    return pd.DataFrame({
        'datetime': time_range,
        'lat': lat_interp,
        'lon': lon_interp,
        'sog': sog_interp,
        'cog': cog_interp
    })


def process_single_file(file_info: dict, seq_len: int, pred_len: int, stride: int):
    """단일 파일 처리 - 시퀀스 샘플 생성

    Args:
        file_info: {'path': str, 'start_area': str, 'end_area': str}
        seq_len: 입력 시퀀스 길이
        pred_len: 출력 시퀀스 길이
        stride: 슬라이딩 윈도우 이동 간격

    Returns:
        결과 딕셔너리 또는 None
    """
    try:
        df = pd.read_csv(file_info['path'])

        # 필수 컬럼 확인
        required = ['datetime', 'lat', 'lon', 'sog', 'cog']
        if not all(c in df.columns for c in required):
            return None

        # 보간
        intp_df = interpolate_trajectory(df, dt_minutes=1)
        if intp_df is None or len(intp_df) < seq_len + pred_len:
            return None

        # 격자 ID 생성
        intp_df['grid_id'] = intp_df.apply(
            lambda r: latlon_to_grid_id(r['lat'], r['lon']), axis=1
        )

        # COG sin/cos 인코딩
        cog_rad = np.radians(intp_df['cog'].values)
        intp_df['cog_sin'] = np.sin(cog_rad)
        intp_df['cog_cos'] = np.cos(cog_rad)

        # 시퀀스 샘플 생성
        total_len = seq_len + pred_len
        samples = []
        grid_ids_all = []

        for i in range(0, len(intp_df) - total_len + 1, stride):
            input_slice = intp_df.iloc[i:i + seq_len]
            target_slice = intp_df.iloc[i + seq_len:i + total_len]

            # 격자 ID
            input_grids = input_slice['grid_id'].tolist()
            target_grids = target_slice['grid_id'].tolist()

            grid_ids_all.extend(input_grids)
            grid_ids_all.extend(target_grids)

            samples.append({
                'input_grid_ids': input_grids,
                'input_sog': input_slice['sog'].values.astype(np.float32),
                'input_cog_sin': input_slice['cog_sin'].values.astype(np.float32),
                'input_cog_cos': input_slice['cog_cos'].values.astype(np.float32),
                'target_grid_ids': target_grids,
                'target_sog': target_slice['sog'].values.astype(np.float32),
                'target_cog_sin': target_slice['cog_sin'].values.astype(np.float32),
                'target_cog_cos': target_slice['cog_cos'].values.astype(np.float32),
                'start_area': file_info['start_area'],
                'end_area': file_info['end_area'],
            })

        if not samples:
            return None

        return {
            'samples': samples,
            'grid_ids': grid_ids_all,
            'sog_values': intp_df['sog'].values,
        }

    except Exception as e:
        print(f"[ERROR] {file_info['path']}: {e}")
        return None


def collect_files(data_folder: str, transition_folder: str = None) -> list:
    """파일 목록 수집 (파일명에서 start_area, end_area 추출)

    파일명 형식: {mmsi}_{start_area}_{end_area}_{start_time}_{end_time}.csv

    Args:
        data_folder: 데이터 폴더
        transition_folder: (미사용, 호환성 유지)

    Returns:
        파일 정보 리스트 [{'path': ..., 'start_area': ..., 'end_area': ...}, ...]
    """
    files = []

    # 전이 폴더가 있으면 그 안에서, 없으면 data_folder에서 직접 검색
    if transition_folder:
        search_path = os.path.join(data_folder, transition_folder)
        if os.path.exists(search_path):
            csv_files = glob(os.path.join(search_path, "**", "*.csv"), recursive=True)
        else:
            csv_files = glob(os.path.join(data_folder, "*.csv"))
    else:
        csv_files = glob(os.path.join(data_folder, "*.csv"))

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        # 파일명 파싱: {mmsi}_{start_area}_{end_area}_{start_time}_{end_time}.csv
        parts = filename.replace('.csv', '').split('_')

        if len(parts) >= 4:
            # mmsi_startarea_endarea_starttime_endtime 형식
            # parts[0] = mmsi
            # parts[1] = start_area
            # parts[2] = end_area
            # 나머지 = timestamps
            start_area = parts[1]
            end_area = parts[2]

            files.append({
                'path': csv_path,
                'start_area': start_area,
                'end_area': end_area,
            })

    return files


def main():
    parser = argparse.ArgumentParser(
        description="격자 기반 학습 데이터 전처리 (V4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_folder", type=str,
                        default="G:/NIA_ai_project/항적데이터 추출/여수",
                        help="항적 데이터 기본 폴더")
    parser.add_argument("--transition_folder", type=str,
                        default="area_transition_results",
                        help="전이 결과 폴더명")
    parser.add_argument("--output_dir", type=str, default="prepared_data_v4",
                        help="출력 폴더")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="입력 시퀀스 길이")
    parser.add_argument("--pred_len", type=int, default=10,
                        help="예측 시퀀스 길이")
    parser.add_argument("--stride", type=int, default=3,
                        help="슬라이딩 윈도우 이동 간격")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="병렬 처리 워커 수")
    parser.add_argument("--neighbor_expand", type=int, default=2,
                        help="인접 격자 확장 범위 (OOV 대응)")
    parser.add_argument("--top_k", type=int, default=50000,
                        help="상위 K개 격자만 유지 (0=제한없음)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("격자 기반 학습 데이터 전처리 (V4)")
    print("=" * 60)
    print(f"격자 크기: {GRID_SIZE}도 (약 {GRID_SIZE * 111:.0f}m)")
    print(f"입력 시퀀스: {args.seq_len}분")
    print(f"예측 시퀀스: {args.pred_len}분")
    print(f"출력 폴더: {args.output_dir}")
    print()

    # ============================================
    # 1. 파일 목록 수집
    # ============================================
    print("[STEP 1] 파일 목록 수집")
    files = collect_files(args.data_folder, args.transition_folder)
    print(f"  - 총 파일 수: {len(files)}")

    if not files:
        print("[ERROR] 처리할 파일이 없습니다.")
        return

    # start/end area 수집
    start_areas = sorted(set(f['start_area'] for f in files))
    end_areas = sorted(set(f['end_area'] for f in files))

    start_vocab = {area: idx + 1 for idx, area in enumerate(start_areas)}
    end_vocab = {area: idx + 1 for idx, area in enumerate(end_areas)}

    print(f"  - Start areas: {start_areas}")
    print(f"  - End areas: {end_areas}")
    print()

    # ============================================
    # 2. 병렬 처리로 시퀀스 샘플 생성
    # ============================================
    print("[STEP 2] 시퀀스 샘플 생성 (병렬 처리)")

    process_func = partial(
        process_single_file,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride
    )

    all_samples = []
    all_grid_ids = []
    all_sog_values = []

    total_files = len(files)
    print(f"  - 처리 중... (총 {total_files} 파일)", flush=True)

    if args.num_workers <= 1:
        # 순차 처리 (Windows 호환)
        for i, file_info in enumerate(files):
            result = process_func(file_info)
            if result is not None:
                all_samples.extend(result['samples'])
                all_grid_ids.extend(result['grid_ids'])
                all_sog_values.extend(result['sog_values'])
            if (i + 1) % 1000 == 0:
                print(f"    진행: {i+1}/{total_files} ({100*(i+1)/total_files:.1f}%)", flush=True)
    else:
        # 병렬 처리
        num_workers = min(args.num_workers, cpu_count())
        processed = 0
        with Pool(num_workers) as pool:
            for result in pool.imap(process_func, files, chunksize=100):
                if result is not None:
                    all_samples.extend(result['samples'])
                    all_grid_ids.extend(result['grid_ids'])
                    all_sog_values.extend(result['sog_values'])
                processed += 1
                if processed % 1000 == 0:
                    print(f"    진행: {processed}/{total_files} ({100*processed/total_files:.1f}%)", flush=True)

    # 결과 처리 (순차 처리에서는 이미 위에서 병합됨)
    dummy_results = []  # 호환성 유지
    for result in dummy_results:
        if result is not None:
            all_samples.extend(result['samples'])
            all_grid_ids.extend(result['grid_ids'])
            all_sog_values.extend(result['sog_values'])

    print(f"  - 총 샘플 수: {len(all_samples)}")
    print(f"  - 수집된 격자 ID 수: {len(all_grid_ids)}")
    print()

    if not all_samples:
        print("[ERROR] 생성된 샘플이 없습니다.")
        return

    # ============================================
    # 3. 격자 Vocabulary 구축
    # ============================================
    print("[STEP 3] 격자 Vocabulary 구축")

    grid_vocab = GridVocabulary()
    for grid_id in all_grid_ids:
        grid_vocab.add_grid(grid_id)

    print(f"  - 원본 격자 수: {len(grid_vocab)}")

    # Top-K 제한 적용
    if args.top_k > 0:
        grid_vocab.truncate_to_top_k(args.top_k)
        print(f"  - Top-K 적용 후: {len(grid_vocab)}")

    # 인접 격자 확장
    grid_vocab.expand_neighbors(n_expand=args.neighbor_expand)
    print(f"  - 확장 후 격자 수: {len(grid_vocab)}")

    # Vocabulary 저장
    vocab_path = os.path.join(args.output_dir, "grid_vocab.json")
    grid_vocab.save(vocab_path)
    print(f"  - 저장: {vocab_path}")
    print()

    # ============================================
    # 4. SOG 정규화 파라미터 계산
    # ============================================
    print("[STEP 4] 정규화 파라미터 계산")

    sog_array = np.array(all_sog_values, dtype=np.float32)
    sog_mean = float(np.mean(sog_array))
    sog_std = float(np.std(sog_array))

    print(f"  - SOG mean: {sog_mean:.4f}")
    print(f"  - SOG std: {sog_std:.4f}")
    print()

    del all_grid_ids, all_sog_values, sog_array
    gc.collect()

    # ============================================
    # 5. 데이터 변환 및 저장
    # ============================================
    print("[STEP 5] 데이터 변환 및 저장")

    n_samples = len(all_samples)

    # 배열 초기화
    input_grid_ids = np.zeros((n_samples, args.seq_len), dtype=np.int64)
    input_sog = np.zeros((n_samples, args.seq_len), dtype=np.float32)
    input_cog_sin = np.zeros((n_samples, args.seq_len), dtype=np.float32)
    input_cog_cos = np.zeros((n_samples, args.seq_len), dtype=np.float32)

    target_grid_ids = np.zeros((n_samples, args.pred_len), dtype=np.int64)
    target_sog = np.zeros((n_samples, args.pred_len), dtype=np.float32)
    target_cog_sin = np.zeros((n_samples, args.pred_len), dtype=np.float32)
    target_cog_cos = np.zeros((n_samples, args.pred_len), dtype=np.float32)

    start_area_ids = np.zeros(n_samples, dtype=np.int64)
    end_area_ids = np.zeros(n_samples, dtype=np.int64)

    for i, sample in enumerate(all_samples):
        # 입력 데이터
        input_grid_ids[i] = grid_vocab.encode_sequence(sample['input_grid_ids'])
        input_sog[i] = (sample['input_sog'] - sog_mean) / sog_std  # 정규화
        input_cog_sin[i] = sample['input_cog_sin']
        input_cog_cos[i] = sample['input_cog_cos']

        # 타겟 데이터
        target_grid_ids[i] = grid_vocab.encode_sequence(sample['target_grid_ids'])
        target_sog[i] = (sample['target_sog'] - sog_mean) / sog_std  # 정규화
        target_cog_sin[i] = sample['target_cog_sin']
        target_cog_cos[i] = sample['target_cog_cos']

        # 범주형
        start_area_ids[i] = start_vocab.get(sample['start_area'], 0)
        end_area_ids[i] = end_vocab.get(sample['end_area'], 0)

    # 저장
    data_path = os.path.join(args.output_dir, "training_data.npz")
    np.savez_compressed(
        data_path,
        input_grid_ids=input_grid_ids,
        input_sog=input_sog,
        input_cog_sin=input_cog_sin,
        input_cog_cos=input_cog_cos,
        target_grid_ids=target_grid_ids,
        target_sog=target_sog,
        target_cog_sin=target_cog_sin,
        target_cog_cos=target_cog_cos,
        start_area_ids=start_area_ids,
        end_area_ids=end_area_ids,
    )
    print(f"  - 학습 데이터 저장: {data_path}")
    print(f"    - input_grid_ids: {input_grid_ids.shape}")
    print(f"    - target_grid_ids: {target_grid_ids.shape}")

    # 메타 정보 저장
    meta_path = os.path.join(args.output_dir, "meta.npz")
    np.savez(
        meta_path,
        version='v4',
        grid_size=GRID_SIZE,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        vocab_size=len(grid_vocab),
        sog_mean=sog_mean,
        sog_std=sog_std,
        start_vocab=str(start_vocab),
        end_vocab=str(end_vocab),
        num_start_area=len(start_vocab) + 1,
        num_end_area=len(end_vocab) + 1,
        n_samples=n_samples,
    )
    print(f"  - 메타 정보 저장: {meta_path}")

    print()
    print("=" * 60)
    print("전처리 완료!")
    print(f"  - 총 샘플 수: {n_samples}")
    print(f"  - 격자 Vocabulary 크기: {len(grid_vocab)}")
    print(f"  - 출력 폴더: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
