# -*- coding: utf-8 -*-
"""
학습 데이터 전처리 및 저장
==========================
원본 CSV 데이터를 읽어서 보간, 정규화, 시퀀스 생성 후 npz 파일로 저장

사용법:
    python prepare_data.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                           --transition_folder "area_transition_results" \
                           --output_dir "prepared_data" \
                           --seq_len 50 --stride 3
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 격자 크기 (도)
GRID_SIZE = 0.05


def compute_grid_id(lat, lon, lat_min, lon_min, grid_size=0.05):
    """위경도를 격자 ID로 변환"""
    grid_row = ((lat - lat_min) / grid_size).astype(int)
    grid_col = ((lon - lon_min) / grid_size).astype(int)
    return grid_row, grid_col


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

    keep_cols = [c for c in df.columns if c in ["datetime","mmsi","lat","lon","sog","cog","fid","start_area","end_area","mmsi_id","start_area_id","end_area_id"]]
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

    for cat_col in ["mmsi_id", "start_area_id", "end_area_id"]:
        if cat_col in df.columns:
            range_df[cat_col] = df[cat_col].iloc[0]

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


def encode_categorical(df, col_name, vocab=None):
    """문자열을 정수 ID로 인코딩"""
    if vocab is None:
        unique_vals = df[col_name].unique()
        vocab = {v: i+1 for i, v in enumerate(unique_vals)}  # 0은 unknown용

    df[f"{col_name}_id"] = df[col_name].map(vocab).fillna(0).astype(int)
    return vocab


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


def load_trajectory_data(transition_df, data_folder):
    """모든 구간의 항적 데이터 로드"""
    filtered_df = transition_df.reset_index(drop=True)

    if len(filtered_df) == 0:
        raise ValueError("전이 정보 데이터가 없습니다.")

    unique_routes = filtered_df.groupby(['start_area', 'end_area']).size()
    print(f"[INFO] 전체 구간 데이터: {len(filtered_df)} 건")
    print(f"[INFO] 구간 종류: {len(unique_routes)} 개")

    all_results = []
    success_count = 0
    fail_count = 0

    for i in range(len(filtered_df)):
        mmsi = filtered_df.mmsi.iloc[i]
        s_area = filtered_df.start_area.iloc[i]
        e_area = filtered_df.end_area.iloc[i]
        start_time = pd.to_datetime(filtered_df.start_time.iloc[i]) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(filtered_df.end_time.iloc[i]) + pd.Timedelta('1 hour')

        start_time_str = start_time.strftime("%Y%m%d%H%M%S")
        end_time_str = end_time.strftime("%Y%m%d%H%M%S")

        filename = f'{mmsi}_{s_area}_{e_area}_{start_time_str}_{end_time_str}.csv'
        filepath = os.path.join(data_folder, filename)

        if not os.path.exists(filepath):
            fail_count += 1
            continue

        try:
            trj = pd.read_csv(filepath, encoding='cp949')
            trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
            trj['fid'] = i

            parsed_mmsi, parsed_start, parsed_end = parse_filename(filename)
            trj['start_area'] = parsed_start if parsed_start else s_area
            trj['end_area'] = parsed_end if parsed_end else e_area

            all_results.append(trj)
            success_count += 1
        except Exception as e:
            print(f"[WARN] 파일 로드 실패: {filepath}, {e}")
            fail_count += 1

    if len(all_results) == 0:
        raise ValueError("로드된 항적 데이터가 없습니다.")

    result_df = pd.concat(all_results, ignore_index=True)
    print(f"[INFO] 항적 데이터 로드 완료: {success_count} 성공, {fail_count} 실패")
    print(f"[INFO] 총 데이터 행: {len(result_df)}")

    return result_df


def prepare_training_data(
    df_all,
    seq_len=50,
    stride=3,
    grid_size=0.05,
    cog_mirror=True,
    seed=42,
):
    """
    학습 데이터 전처리 및 시퀀스 생성

    Returns:
        dict: 학습에 필요한 모든 데이터와 메타정보
    """
    print("=" * 60)
    print("[STEP 1] 데이터 전처리 시작")
    print("=" * 60)

    # 필수 컬럼 확인
    required = ["datetime", "mmsi", "lat", "lon", "sog", "cog", "fid"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df_all = df_all.copy()
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    for c in ["lat", "lon", "sog", "cog", "mmsi", "fid"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all = df_all.dropna(subset=["datetime", "lat", "lon", "sog", "cog", "fid", "mmsi"])
    df_all = df_all.sort_values(["fid", "datetime"]).reset_index(drop=True)

    print(f"[INFO] 원본 rows={len(df_all)}, fid={df_all.fid.nunique()}, mmsi={df_all.mmsi.nunique()}")

    # Categorical 인코딩
    if "start_area" not in df_all.columns or "end_area" not in df_all.columns:
        print("[INFO] start_area/end_area 컬럼 없음 - 기본값 사용")
        df_all["start_area"] = "unknown"
        df_all["end_area"] = "unknown"

    mmsi_vocab = encode_categorical(df_all, "mmsi")
    start_vocab = encode_categorical(df_all, "start_area")
    end_vocab = encode_categorical(df_all, "end_area")

    print(f"[INFO] Categorical 인코딩:")
    print(f"  - MMSI: {len(mmsi_vocab)} 종류")
    print(f"  - Start Area: {len(start_vocab)} 종류")
    print(f"  - End Area: {len(end_vocab)} 종류")

    # 보간 + segment_bounds 생성
    print("\n[STEP 2] 1분 간격 보간 및 시퀀스 생성")
    intp_segments = []
    seg_lengths = []

    for fid, df_fid in df_all.groupby("fid"):
        segments_raw = split_by_gap(df_fid, max_gap_days=1)
        for seg_df in segments_raw:
            intp_df = data_intp(seg_df)
            if intp_df is None or len(intp_df) == 0:
                continue
            intp_df = intp_df.sort_values("datetime").reset_index(drop=True)
            intp_segments.append(intp_df)
            seg_lengths.append(len(intp_df))

    if len(intp_segments) == 0:
        raise RuntimeError("보간된 segment가 없습니다.")

    intp_all = pd.concat(intp_segments, ignore_index=True)
    del intp_segments  # 메모리 해제

    segment_bounds = []
    s = 0
    for L in seg_lengths:
        e = s + L
        segment_bounds.append((s, e))
        s = e

    print(f"[INFO] 보간 후 rows={len(intp_all)}, segments={len(segment_bounds)}")

    # 격자 정보 생성
    lat_min, lat_max = intp_all["lat"].min(), intp_all["lat"].max()
    lon_min, lon_max = intp_all["lon"].min(), intp_all["lon"].max()

    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    lat_bounds = (lat_min - lat_margin, lat_max + lat_margin)
    lon_bounds = (lon_min - lon_margin, lon_max + lon_margin)

    grid_info = create_grid_mapping(
        lat_min - lat_margin, lat_max + lat_margin,
        lon_min - lon_margin, lon_max + lon_margin,
        grid_size
    )

    print(f"[INFO] 항로 범위: lat={lat_bounds[0]:.4f}~{lat_bounds[1]:.4f}, lon={lon_bounds[0]:.4f}~{lon_bounds[1]:.4f}")
    print(f"[INFO] 격자: {grid_info['num_rows']}x{grid_info['num_cols']} = {grid_info['total_grids']} (크기: {grid_size}도)")

    # sin/cos 변환
    sin_cog = np.sin(np.radians(intp_all["cog"].values))
    cos_cog = np.cos(np.radians(intp_all["cog"].values))
    if cog_mirror:
        sin_cog = -sin_cog

    # 격자 ID 계산
    grid_row, grid_col = compute_grid_id(
        intp_all["lat"].values, intp_all["lon"].values,
        grid_info['lat_min'], grid_info['lon_min'],
        grid_info['grid_size']
    )
    grid_row = np.clip(grid_row, 0, grid_info['num_rows'] - 1)
    grid_col = np.clip(grid_col, 0, grid_info['num_cols'] - 1)
    grid_id = grid_row * grid_info['num_cols'] + grid_col

    # 수치형 데이터 배열
    X_num = np.column_stack([
        intp_all["lat"].values,
        intp_all["lon"].values,
        intp_all["sog"].values,
        sin_cog,
        cos_cog
    ]).astype(np.float32)

    # Categorical 데이터 배열
    X_cat = np.column_stack([
        intp_all["mmsi_id"].values if "mmsi_id" in intp_all.columns else np.zeros(len(intp_all)),
        intp_all["start_area_id"].values if "start_area_id" in intp_all.columns else np.zeros(len(intp_all)),
        intp_all["end_area_id"].values if "end_area_id" in intp_all.columns else np.zeros(len(intp_all)),
        grid_id
    ]).astype(np.int64)

    # 타겟 (다음 스텝 예측)
    Y = X_num.copy()

    # 정규화 파라미터
    x_mean = X_num.mean(axis=0, keepdims=True)
    x_std = X_num.std(axis=0, keepdims=True) + 1e-6
    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True) + 1e-6

    # 정규화 적용
    Xn_num = (X_num - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    del intp_all, X_num, Y  # 메모리 해제

    # 시퀀스 인덱스 생성
    print("\n[STEP 3] 시퀀스 인덱스 생성")
    segment_starts = []
    for (s, e) in segment_bounds:
        starts = []
        max_start = e - 1 - seq_len
        if max_start >= s:
            for i in range(s, max_start + 1, stride):
                starts.append(i)
        segment_starts.append(starts)

    all_indices = [i for starts in segment_starts for i in starts]
    print(f"[INFO] 총 시퀀스 수: {len(all_indices)}")

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

    return {
        'Xn_num': Xn_num,
        'X_cat': X_cat,
        'Yn': Yn,
        'segment_bounds': np.array(segment_bounds),
        'segment_starts': segment_starts,
        'all_indices': np.array(all_indices),
        'meta': meta,
    }


def save_prepared_data(data, output_dir):
    """전처리된 데이터를 npz 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 파일
    data_path = os.path.join(output_dir, "training_data.npz")
    np.savez_compressed(
        data_path,
        Xn_num=data['Xn_num'],
        X_cat=data['X_cat'],
        Yn=data['Yn'],
        segment_bounds=data['segment_bounds'],
        all_indices=data['all_indices'],
    )
    print(f"[SAVED] 학습 데이터: {data_path}")

    # 메타 정보 파일
    meta_path = os.path.join(output_dir, "meta.npz")
    np.savez(meta_path, **data['meta'])
    print(f"[SAVED] 메타 정보: {meta_path}")

    # segment_starts는 리스트의 리스트이므로 별도 저장
    # pickle 대신 numpy object array 사용
    seg_starts_path = os.path.join(output_dir, "segment_starts.npy")
    np.save(seg_starts_path, np.array(data['segment_starts'], dtype=object), allow_pickle=True)
    print(f"[SAVED] Segment starts: {seg_starts_path}")

    # 파일 크기 출력
    data_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"\n[INFO] 데이터 파일 크기: {data_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="학습 데이터 전처리 및 저장",
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

    print("=" * 60)
    print("학습 데이터 전처리")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_folder}")
    print(f"전이 정보 폴더: {args.transition_folder}")
    print(f"출력 폴더: {args.output_dir}")
    print(f"시퀀스 길이: {args.seq_len}")
    print(f"Stride: {args.stride}")
    print(f"격자 크기: {args.grid_size}도")
    print("=" * 60)

    # 1. 전이 정보 로드
    print("\n[STEP 0] 데이터 로드")
    transition_df = load_transition_data(args.transition_folder)

    # 2. 항적 데이터 로드
    trajectory_df = load_trajectory_data(transition_df, args.data_folder)

    # 3. 전처리
    data = prepare_training_data(
        trajectory_df,
        seq_len=args.seq_len,
        stride=args.stride,
        grid_size=args.grid_size,
    )

    # 4. 저장
    print("\n[STEP 4] 데이터 저장")
    save_prepared_data(data, args.output_dir)

    print("\n" + "=" * 60)
    print("전처리 완료!")
    print(f"저장 위치: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
