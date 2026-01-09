# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

DATA_FOLDER = 'G:/NIA_ai_project/항적데이터 추출/여수'
TRANSITION_FOLDER = 'area_transition_results'
TARGET_MMSI = 440103700

# 테스트 파일 찾기
transition_files = [f for f in os.listdir(TRANSITION_FOLDER) if f.endswith('.csv')][:50]
all_trans = pd.concat([pd.read_csv(os.path.join(TRANSITION_FOLDER, f)) for f in transition_files])
target_trans = all_trans[all_trans['mmsi'] == TARGET_MMSI]

print(f"MMSI {TARGET_MMSI} 전체 파일 분석")
print("=" * 60)

total_8knot = 0
total_rows = 0
max_sog_overall = 0

for idx, row in target_trans.iterrows():
    s_area = row['start_area']
    e_area = row['end_area']
    start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
    end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

    filename = f"{TARGET_MMSI}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(filepath):
        continue

    try:
        df = pd.read_csv(filepath, encoding='cp949')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        for c in ['lat', 'lon', 'sog']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['datetime', 'lat', 'lon', 'sog'])

        n_fast = (df.sog >= 8).sum()
        total_8knot += n_fast
        total_rows += len(df)
        max_sog_overall = max(max_sog_overall, df.sog.max())

        if n_fast > 0:
            print(f'{filename[:50]}...')
            print(f'  rows: {len(df)}, 8노트+: {n_fast}, max_sog: {df.sog.max():.1f}')
    except:
        continue

print()
print("=" * 60)
print(f"총 데이터: {total_rows} rows")
print(f"8노트 이상: {total_8knot} rows ({100*total_8knot/max(1,total_rows):.1f}%)")
print(f"전체 최대 속력: {max_sog_overall:.1f} 노트")

# 빈도 높은 MMSI 중 고속 선박 찾기
print("\n\n빈도 높은 MMSI 중 고속 선박 탐색")
print("=" * 60)

mmsi_counts = all_trans['mmsi'].value_counts().head(20)
for mmsi, count in mmsi_counts.items():
    mmsi_trans = all_trans[all_trans['mmsi'] == mmsi].head(5)
    max_sog = 0
    total_fast = 0

    for _, row in mmsi_trans.iterrows():
        s_area = row['start_area']
        e_area = row['end_area']
        start_time = pd.to_datetime(row['start_time']) - pd.Timedelta('1 hour')
        end_time = pd.to_datetime(row['end_time']) + pd.Timedelta('1 hour')

        filename = f"{mmsi}_{s_area}_{e_area}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}.csv"
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, encoding='cp949')
            for c in ['sog']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            max_sog = max(max_sog, df.sog.max())
            total_fast += (df.sog >= 8).sum()
        except:
            continue

    if max_sog >= 8:
        print(f"MMSI {mmsi}: {count}건, max_sog={max_sog:.1f}노트, 8노트+={total_fast}")

