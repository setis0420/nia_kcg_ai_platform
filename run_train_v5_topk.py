# -*- coding: utf-8 -*-
"""
V5 빠른 테스트: 빈도 높은 상위 K개 MMSI만 학습
===============================================
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_global_model_v5 import train_mmsi_model, create_grid_mapping, MIN_SOG

# 설정
DATA_FOLDER = "G:/NIA_ai_project/항적데이터 추출/여수"
TRANSITION_FOLDER = "area_transition_results"
SAVE_DIR = "global_model_v5"
TOP_K = 5  # 상위 몇 개 MMSI 학습할지
GRID_SIZE = 0.01
MIN_SOG_FILTER = 8.0


def main():
    print("=" * 60)
    print(f"V5 빠른 테스트: 상위 {TOP_K}개 MMSI 학습")
    print("=" * 60)

    # 1. 전이 정보 로드 및 MMSI 빈도 분석
    print("\n[STEP 1] MMSI 빈도 분석")

    transition_files = [f for f in os.listdir(TRANSITION_FOLDER) if f.endswith('.csv')]
    all_transitions = []
    for f in transition_files:
        df = pd.read_csv(os.path.join(TRANSITION_FOLDER, f))
        all_transitions.append(df)

    transition_df = pd.concat(all_transitions, ignore_index=True)
    print(f"  전이 정보: {len(transition_df)} 건")

    # MMSI별 빈도 계산
    mmsi_counts = transition_df['mmsi'].value_counts()
    print(f"\n  상위 {TOP_K}개 MMSI:")
    for i, (mmsi, count) in enumerate(mmsi_counts.head(TOP_K).items()):
        print(f"    {i+1}. MMSI {mmsi}: {count} 건")

    top_mmsi_list = mmsi_counts.head(TOP_K).index.tolist()

    # 2. 상위 MMSI 데이터만 로드
    print(f"\n[STEP 2] 상위 {TOP_K}개 MMSI 데이터 로드")

    filtered_df = transition_df[transition_df['mmsi'].isin(top_mmsi_list)].reset_index(drop=True)
    print(f"  필터링된 전이 정보: {len(filtered_df)} 건")

    all_data = []
    for i in range(len(filtered_df)):
        mmsi = filtered_df.mmsi.iloc[i]
        s_area = filtered_df.start_area.iloc[i]
        e_area = filtered_df.end_area.iloc[i]
        start_time = pd.to_datetime(filtered_df.start_time.iloc[i]) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(filtered_df.end_time.iloc[i]) + pd.Timedelta('1 hour')

        start_time_str = start_time.strftime("%Y%m%d%H%M%S")
        end_time_str = end_time.strftime("%Y%m%d%H%M%S")

        filename = f'{mmsi}_{s_area}_{e_area}_{start_time_str}_{end_time_str}.csv'
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            continue

        try:
            trj = pd.read_csv(filepath, encoding='cp949')
            trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
            trj['mmsi'] = mmsi
            all_data.append(trj)
        except:
            continue

        if (i + 1) % 100 == 0:
            print(f"    진행: {i+1}/{len(filtered_df)}", flush=True)

    if not all_data:
        print("[ERROR] 로드된 데이터 없음")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    print(f"  총 데이터: {len(df_all)} rows")

    # 3. 격자 정보 생성
    print("\n[STEP 3] 격자 정보 생성")

    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    for c in ["lat", "lon", "sog", "mmsi"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all = df_all.dropna(subset=["datetime", "lat", "lon", "sog", "mmsi"])

    # 속력 필터
    df_all = df_all[df_all["sog"] >= MIN_SOG_FILTER].reset_index(drop=True)
    print(f"  속력 >= {MIN_SOG_FILTER} 노트 필터 후: {len(df_all)} rows")

    lat_min, lat_max = df_all["lat"].min(), df_all["lat"].max()
    lon_min, lon_max = df_all["lon"].min(), df_all["lon"].max()

    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05

    grid_info = create_grid_mapping(
        lat_min - lat_margin, lat_max + lat_margin,
        lon_min - lon_margin, lon_max + lon_margin,
        GRID_SIZE
    )

    print(f"  격자: {grid_info['num_rows']}x{grid_info['num_cols']} = {grid_info['total_grids']}")

    # 격자 정보 저장
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.savez(
        os.path.join(SAVE_DIR, "grid_info.npz"),
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in grid_info.items()},
        min_sog=MIN_SOG_FILTER,
    )

    # 4. MMSI별 학습
    print(f"\n[STEP 4] MMSI별 모델 학습")

    results = {}
    for mmsi in top_mmsi_list:
        df_mmsi = df_all[df_all["mmsi"] == mmsi].copy()

        if len(df_mmsi) < 100:
            print(f"  MMSI {mmsi}: 데이터 부족 ({len(df_mmsi)} rows) - 스킵")
            continue

        model_path, config_path = train_mmsi_model(
            mmsi=mmsi,
            df_mmsi=df_mmsi,
            grid_info=grid_info,
            seq_len=50,
            stride=1,
            epochs=100,
            batch_size=64,
            lr=1e-3,
            save_dir=SAVE_DIR,
            device="cuda",
            patience=15,
            warmup_epochs=10,
            val_ratio=0.2,
        )

        if model_path:
            results[mmsi] = model_path

    # 5. 완료
    print("\n" + "=" * 60)
    print(f"학습 완료! 성공: {len(results)}/{TOP_K}")
    print(f"저장 위치: {SAVE_DIR}")
    for mmsi, path in results.items():
        print(f"  - MMSI {mmsi}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
