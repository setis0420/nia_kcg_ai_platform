# -*- coding: utf-8 -*-
"""
LLM 스타일 격자 ID 시퀀스 학습 데이터 전처리 (V5)
=================================================
핵심 특징:
- 격자 ID 시퀀스만 사용 (속력/침로 벡터 삭제)
- 속력 8노트 이상만 필터링
- MMSI별로 데이터 분리 저장

사용법:
    python prepare_data_v5.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                               --transition_folder "area_transition_results" \
                               --output_dir "prepared_data_v5" \
                               --seq_len 50 --stride 1 --min_sog 8.0
"""

import os
import json
import gc
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

# 격자 크기 (도 단위) - 약 1.1km
GRID_SIZE = 0.01

# 최소 속력 (노트)
MIN_SOG = 8.0


# ============================================
# 격자 변환 유틸리티
# ============================================

def latlon_to_grid_id(lat: float, lon: float, lat_min: float, lon_min: float,
                       grid_size: float, num_cols: int) -> int:
    """위경도 → 격자 ID (정수)"""
    grid_row = int((lat - lat_min) / grid_size)
    grid_col = int((lon - lon_min) / grid_size)
    return grid_row * num_cols + grid_col


def grid_id_to_latlon(grid_id: int, lat_min: float, lon_min: float,
                       grid_size: float, num_cols: int) -> tuple:
    """격자 ID → 위경도 (격자 중심점)"""
    grid_row = grid_id // num_cols
    grid_col = grid_id % num_cols
    lat = lat_min + (grid_row + 0.5) * grid_size
    lon = lon_min + (grid_col + 0.5) * grid_size
    return lat, lon


# ============================================
# 데이터 전처리 함수
# ============================================

def interpolate_trajectory(df: pd.DataFrame, dt_minutes: int = 1) -> pd.DataFrame:
    """항적 데이터를 일정 시간 간격으로 보간"""
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

    return pd.DataFrame({
        'datetime': time_range,
        'lat': lat_interp,
        'lon': lon_interp,
        'sog': sog_interp
    })


def split_by_gap(df: pd.DataFrame, max_gap_minutes: int = 10) -> list:
    """시간 간격이 큰 경우 세그먼트 분리"""
    if df is None or df.empty:
        return []

    df = df.sort_values('datetime').copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    if df.empty:
        return []

    diff = df['datetime'].diff()
    new_seg = (diff >= pd.Timedelta(minutes=max_gap_minutes)) | diff.isna()
    seg_id = new_seg.cumsum()

    return [g.copy() for _, g in df.groupby(seg_id) if len(g) > 0]


def load_transition_data(transition_folder: str) -> pd.DataFrame:
    """전이 정보 CSV 파일들을 로드하여 병합"""
    if not os.path.exists(transition_folder):
        raise FileNotFoundError(f"전이 정보 폴더가 존재하지 않습니다: {transition_folder}")

    file_list = [f for f in os.listdir(transition_folder) if f.lower().endswith('.csv')]

    if len(file_list) == 0:
        raise ValueError(f"전이 정보 폴더에 CSV 파일이 없습니다: {transition_folder}")

    df_list = []
    for f in file_list:
        file_path = os.path.join(transition_folder, f)
        df = pd.read_csv(file_path)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] 전이 정보 로드 완료: {len(merged_df)} 건, {len(file_list)} 파일")
    return merged_df


def load_trajectory_files(transition_df: pd.DataFrame, data_folder: str) -> list:
    """전이 정보 기반 항적 파일 목록 생성"""
    files = []

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
            files.append({
                'path': filepath,
                'mmsi': int(mmsi),
                'start_area': s_area,
                'end_area': e_area,
            })

    return files


def main():
    parser = argparse.ArgumentParser(
        description="LLM 스타일 격자 ID 시퀀스 학습 데이터 전처리 (V5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_folder", type=str,
                        default="G:/NIA_ai_project/항적데이터 추출/여수",
                        help="항적 데이터 폴더")
    parser.add_argument("--transition_folder", type=str,
                        default="area_transition_results",
                        help="전이 정보 폴더")
    parser.add_argument("--output_dir", type=str, default="prepared_data_v5",
                        help="출력 폴더")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="입력 시퀀스 길이")
    parser.add_argument("--stride", type=int, default=1,
                        help="슬라이딩 윈도우 이동 간격")
    parser.add_argument("--grid_size", type=float, default=0.01,
                        help="격자 크기 (도 단위, 기본값: 0.01 = 약 1.1km)")
    parser.add_argument("--min_sog", type=float, default=8.0,
                        help="최소 속력 필터 (노트)")

    args = parser.parse_args()

    global GRID_SIZE, MIN_SOG
    GRID_SIZE = args.grid_size
    MIN_SOG = args.min_sog

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LLM 스타일 격자 ID 시퀀스 전처리 (V5)")
    print("=" * 60)
    print(f"격자 크기: {GRID_SIZE}도 (약 {GRID_SIZE * 111:.1f}km)")
    print(f"최소 속력: {MIN_SOG} 노트")
    print(f"입력 시퀀스: {args.seq_len}분")
    print(f"출력 폴더: {args.output_dir}")
    print()

    # ============================================
    # 1. 전이 정보 및 파일 목록 로드
    # ============================================
    print("[STEP 1] 파일 목록 수집")
    transition_df = load_transition_data(args.transition_folder)
    files = load_trajectory_files(transition_df, args.data_folder)
    print(f"  - 총 파일 수: {len(files)}")

    if not files:
        print("[ERROR] 처리할 파일이 없습니다.")
        return

    # MMSI별 파일 그룹화
    mmsi_files = defaultdict(list)
    for f in files:
        mmsi_files[f['mmsi']].append(f)

    print(f"  - MMSI 수: {len(mmsi_files)}")
    print()

    # ============================================
    # 2. 전체 데이터에서 위경도 범위 계산
    # ============================================
    print("[STEP 2] 위경도 범위 계산")

    all_lats = []
    all_lons = []

    for file_info in files[:min(100, len(files))]:  # 샘플링
        try:
            df = pd.read_csv(file_info['path'], encoding='cp949')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            if 'lat' in df.columns and 'lon' in df.columns:
                all_lats.extend(df['lat'].dropna().tolist())
                all_lons.extend(df['lon'].dropna().tolist())
        except:
            continue

    if not all_lats:
        print("[ERROR] 위경도 데이터를 읽을 수 없습니다.")
        return

    lat_min = min(all_lats) - 0.05
    lat_max = max(all_lats) + 0.05
    lon_min = min(all_lons) - 0.05
    lon_max = max(all_lons) + 0.05

    num_rows = int(np.ceil((lat_max - lat_min) / GRID_SIZE)) + 1
    num_cols = int(np.ceil((lon_max - lon_min) / GRID_SIZE)) + 1
    total_grids = num_rows * num_cols

    grid_info = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'grid_size': GRID_SIZE,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'total_grids': total_grids,
    }

    print(f"  - 위도 범위: {lat_min:.4f} ~ {lat_max:.4f}")
    print(f"  - 경도 범위: {lon_min:.4f} ~ {lon_max:.4f}")
    print(f"  - 격자: {num_rows} x {num_cols} = {total_grids}")
    print()

    # ============================================
    # 3. MMSI별 데이터 전처리 및 저장
    # ============================================
    print("[STEP 3] MMSI별 데이터 전처리")

    success_count = 0
    fail_count = 0
    total_samples = 0

    for mmsi, mmsi_file_list in mmsi_files.items():
        print(f"\n  MMSI: {mmsi} ({len(mmsi_file_list)} 파일)")

        all_grid_sequences = []

        for file_info in mmsi_file_list:
            try:
                df = pd.read_csv(file_info['path'], encoding='cp949')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                # 필수 컬럼 확인
                required = ['datetime', 'lat', 'lon', 'sog']
                if not all(c in df.columns for c in required):
                    continue

                # 데이터 정제
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                for c in ['lat', 'lon', 'sog']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna(subset=required)

                # 속력 필터링
                df = df[df['sog'] >= MIN_SOG].reset_index(drop=True)

                if len(df) < args.seq_len + 1:
                    continue

                # 세그먼트 분리
                segments = split_by_gap(df, max_gap_minutes=10)

                for seg_df in segments:
                    # 보간
                    intp_df = interpolate_trajectory(seg_df, dt_minutes=1)
                    if intp_df is None or len(intp_df) < args.seq_len + 1:
                        continue

                    # 속력 필터 재적용 (보간 후)
                    intp_df = intp_df[intp_df['sog'] >= MIN_SOG].reset_index(drop=True)
                    if len(intp_df) < args.seq_len + 1:
                        continue

                    # 격자 ID 계산
                    grid_ids = []
                    for _, row in intp_df.iterrows():
                        gid = latlon_to_grid_id(
                            row['lat'], row['lon'],
                            lat_min, lon_min,
                            GRID_SIZE, num_cols
                        )
                        gid = max(0, min(gid, total_grids - 1))
                        grid_ids.append(gid)

                    all_grid_sequences.append(np.array(grid_ids, dtype=np.int64))

            except Exception as e:
                continue

        if not all_grid_sequences:
            print(f"    -> 스킵 (유효 데이터 없음)")
            fail_count += 1
            continue

        # 시퀀스 샘플 생성
        samples_x = []
        samples_y = []

        for grid_seq in all_grid_sequences:
            for i in range(0, len(grid_seq) - args.seq_len, args.stride):
                x = grid_seq[i:i + args.seq_len]
                y = grid_seq[i + args.seq_len]  # 다음 격자 ID (단일 값)
                samples_x.append(x)
                samples_y.append(y)

        if len(samples_x) < 10:
            print(f"    -> 스킵 (샘플 부족: {len(samples_x)})")
            fail_count += 1
            continue

        # MMSI별 폴더 생성 및 저장
        mmsi_dir = os.path.join(args.output_dir, str(mmsi))
        os.makedirs(mmsi_dir, exist_ok=True)

        X = np.array(samples_x, dtype=np.int64)
        Y = np.array(samples_y, dtype=np.int64)

        np.savez_compressed(
            os.path.join(mmsi_dir, 'data.npz'),
            X=X,
            Y=Y,
        )

        # 메타 정보 저장
        np.savez(
            os.path.join(mmsi_dir, 'meta.npz'),
            mmsi=int(mmsi),
            seq_len=args.seq_len,
            n_samples=len(samples_x),
            **grid_info
        )

        print(f"    -> 저장 완료: {len(samples_x)} 샘플")
        success_count += 1
        total_samples += len(samples_x)

    # ============================================
    # 4. 전역 메타 정보 저장
    # ============================================
    print("\n[STEP 4] 전역 메타 정보 저장")

    np.savez(
        os.path.join(args.output_dir, 'global_meta.npz'),
        version='v5',
        grid_size=GRID_SIZE,
        min_sog=MIN_SOG,
        seq_len=args.seq_len,
        stride=args.stride,
        total_grids=total_grids,
        num_rows=num_rows,
        num_cols=num_cols,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        n_mmsi_success=success_count,
        n_mmsi_fail=fail_count,
        total_samples=total_samples,
    )

    # 성공한 MMSI 목록 저장
    success_mmsi = [str(mmsi) for mmsi in mmsi_files.keys()
                    if os.path.exists(os.path.join(args.output_dir, str(mmsi), 'data.npz'))]

    with open(os.path.join(args.output_dir, 'mmsi_list.json'), 'w') as f:
        json.dump(success_mmsi, f, indent=2)

    print()
    print("=" * 60)
    print("전처리 완료!")
    print(f"  - 성공 MMSI: {success_count}")
    print(f"  - 실패/스킵 MMSI: {fail_count}")
    print(f"  - 총 샘플 수: {total_samples}")
    print(f"  - 출력 폴더: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
