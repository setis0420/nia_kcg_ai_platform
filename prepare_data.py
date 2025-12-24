# -*- coding: utf-8 -*-
"""
학습 데이터 전처리 및 저장 (메모리 효율 버전)
=============================================
파일을 하나씩 처리하여 메모리 사용량 최소화

사용법:
    python prepare_data.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                           --transition_folder "area_transition_results" \
                           --output_dir "prepared_data" \
                           --seq_len 50 --stride 3
"""

import os
import gc
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings

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
        df = pd.read_csv(file_path)
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


def process_single_file(file_info, cog_mirror=True):
    """단일 파일 처리 (보간)"""
    try:
        trj = pd.read_csv(file_info['filepath'], encoding='cp949')
        trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
        trj['fid'] = file_info['fid']

        parsed_mmsi, parsed_start, parsed_end = parse_filename(file_info['filename'])
        trj['start_area'] = parsed_start if parsed_start else file_info['start_area']
        trj['end_area'] = parsed_end if parsed_end else file_info['end_area']

        # datetime 변환
        trj["datetime"] = pd.to_datetime(trj["datetime"], errors="coerce")
        for c in ["lat", "lon", "sog", "cog", "mmsi"]:
            if c in trj.columns:
                trj[c] = pd.to_numeric(trj[c], errors="coerce")
        trj = trj.dropna(subset=["datetime", "lat", "lon", "sog", "cog"])

        if trj.empty:
            return []

        # 시간 간격으로 segment 분리 및 보간
        segments = split_by_gap(trj, max_gap_days=1)

        results = []
        for seg_df in segments:
            intp_df = data_intp(seg_df)
            if intp_df is not None and len(intp_df) > 0:
                intp_df = intp_df.sort_values("datetime").reset_index(drop=True)

                # sin/cos 계산
                sin_cog = np.sin(np.radians(intp_df["cog"].values))
                cos_cog = np.cos(np.radians(intp_df["cog"].values))
                if cog_mirror:
                    sin_cog = -sin_cog

                intp_df["sin_cog"] = sin_cog
                intp_df["cos_cog"] = cos_cog

                results.append(intp_df)

        return results

    except Exception as e:
        print(f"[WARN] 파일 처리 실패: {file_info['filepath']}, {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="학습 데이터 전처리 및 저장 (메모리 효율 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_folder", type=str, required=True,
                        help="항적 CSV 파일이 저장된 폴더")
    parser.add_argument("--transition_folder", type=str, required=True,
                        help="전이 정보 CSV 파일이 저장된 폴더")
    parser.add_argument("--output_dir", type=str, default="prepared_data",
                        help="전처리된 데이터 저장 폴더 (기본값: prepared_data)")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="시퀀스 길이 (기본값: 50)")
    parser.add_argument("--stride", type=int, default=3,
                        help="슬라이딩 윈도우 이동 간격 (기본값: 3)")
    parser.add_argument("--grid_size", type=float, default=0.05,
                        help="격자 크기 (도 단위, 기본값: 0.05)")

    args = parser.parse_args()

    seq_len = args.seq_len
    stride = args.stride
    grid_size = args.grid_size
    cog_mirror = True

    print("=" * 60)
    print("학습 데이터 전처리 (메모리 효율 버전)")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_folder}")
    print(f"전이 정보 폴더: {args.transition_folder}")
    print(f"출력 폴더: {args.output_dir}")
    print(f"시퀀스 길이: {seq_len}")
    print(f"Stride: {stride}")
    print(f"격자 크기: {grid_size}도")
    print("=" * 60)

    # 출력 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)
    segments_dir = os.path.join(args.output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    # 1. 전이 정보 로드
    print("\n[STEP 1] 파일 목록 생성")
    transition_df = load_transition_data(args.transition_folder)
    file_list = get_file_list(transition_df, args.data_folder)
    print(f"[INFO] 유효한 파일 수: {len(file_list)}")

    # ==============================================
    # PASS 1: 통계 수집 (파일 하나씩 처리)
    # ==============================================
    print("\n[STEP 2] 통계 수집 (Pass 1)")

    # 온라인 통계 계산을 위한 변수
    n_total = 0
    sum_x = np.zeros(5, dtype=np.float64)  # lat, lon, sog, sin_cog, cos_cog
    sum_x2 = np.zeros(5, dtype=np.float64)

    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')

    mmsi_set = set()
    start_area_set = set()
    end_area_set = set()

    segment_info_list = []  # (file_idx, seg_idx, length, fid, mmsi, start_area, end_area)

    for file_idx, file_info in enumerate(file_list):
        if file_idx % 100 == 0:
            print(f"  처리 중: {file_idx}/{len(file_list)}")

        segments = process_single_file(file_info, cog_mirror=cog_mirror)

        for seg_idx, seg_df in enumerate(segments):
            n = len(seg_df)
            if n < seq_len + 1:
                continue

            # 수치 데이터
            x = np.column_stack([
                seg_df["lat"].values,
                seg_df["lon"].values,
                seg_df["sog"].values,
                seg_df["sin_cog"].values,
                seg_df["cos_cog"].values,
            ]).astype(np.float64)

            # 온라인 통계 업데이트
            n_total += n
            sum_x += x.sum(axis=0)
            sum_x2 += (x ** 2).sum(axis=0)

            # 범위 업데이트
            lat_min = min(lat_min, seg_df["lat"].min())
            lat_max = max(lat_max, seg_df["lat"].max())
            lon_min = min(lon_min, seg_df["lon"].min())
            lon_max = max(lon_max, seg_df["lon"].max())

            # Categorical 수집
            mmsi_set.add(file_info['mmsi'])
            start_area_set.add(seg_df["start_area"].iloc[0])
            end_area_set.add(seg_df["end_area"].iloc[0])

            # segment 정보 저장
            segment_info_list.append({
                'file_idx': file_idx,
                'seg_idx': seg_idx,
                'length': n,
                'fid': file_info['fid'],
                'mmsi': file_info['mmsi'],
                'start_area': seg_df["start_area"].iloc[0],
                'end_area': seg_df["end_area"].iloc[0],
            })

        del segments
        gc.collect()

    print(f"[INFO] 유효 segment 수: {len(segment_info_list)}")
    print(f"[INFO] 총 데이터 포인트: {n_total}")

    if n_total == 0:
        raise RuntimeError("처리된 데이터가 없습니다.")

    # 평균/표준편차 계산
    x_mean = (sum_x / n_total).astype(np.float32).reshape(1, -1)
    x_var = (sum_x2 / n_total) - (sum_x / n_total) ** 2
    x_std = (np.sqrt(np.maximum(x_var, 1e-12)) + 1e-6).astype(np.float32).reshape(1, -1)

    # y도 동일 (다음 스텝 예측)
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
    # PASS 2: 정규화 및 저장 (파일 단위로 처리)
    # ==============================================
    print("\n[STEP 3] 정규화 및 저장 (Pass 2)")

    segment_starts = []
    current_offset = 0

    all_Xn_num = []
    all_X_cat = []
    all_Yn = []
    segment_bounds = []

    # 파일별로 segment 정보 그룹화
    file_to_segments = defaultdict(list)
    for i, seg_info in enumerate(segment_info_list):
        file_to_segments[seg_info['file_idx']].append((i, seg_info))

    # 파일 단위로 처리 (같은 파일을 여러 번 읽지 않음)
    processed_count = 0
    for file_idx in sorted(file_to_segments.keys()):
        if processed_count % 100 == 0:
            print(f"  처리 중: {processed_count}/{len(segment_info_list)} segments")

        file_info = file_list[file_idx]
        segments = process_single_file(file_info, cog_mirror=cog_mirror)

        for orig_idx, seg_info in file_to_segments[file_idx]:
            seg_idx = seg_info['seg_idx']

            if seg_idx >= len(segments):
                continue

            seg_df = segments[seg_idx]
            n = len(seg_df)

            if n < seq_len + 1:
                continue

            # 수치 데이터
            x = np.column_stack([
                seg_df["lat"].values,
                seg_df["lon"].values,
                seg_df["sog"].values,
                seg_df["sin_cog"].values,
                seg_df["cos_cog"].values,
            ]).astype(np.float32)

            # 정규화
            xn = (x - x_mean) / x_std
            yn = xn.copy()  # 타겟도 동일

            # Categorical
            mmsi_id = mmsi_vocab.get(seg_info['mmsi'], 0)
            start_id = start_vocab.get(seg_info['start_area'], 0)
            end_id = end_vocab.get(seg_info['end_area'], 0)

            # 격자 ID (벡터화)
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

            # 저장
            all_Xn_num.append(xn)
            all_X_cat.append(x_cat)
            all_Yn.append(yn)

            # segment 경계
            start_idx = current_offset
            end_idx = current_offset + n
            segment_bounds.append((start_idx, end_idx))

            # 시퀀스 시작 인덱스
            starts = []
            max_start = end_idx - 1 - seq_len
            if max_start >= start_idx:
                for i in range(start_idx, max_start + 1, stride):
                    starts.append(i)
            segment_starts.append(starts)

            current_offset = end_idx
            processed_count += 1

        del segments
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
    print("=" * 60)


if __name__ == "__main__":
    main()
