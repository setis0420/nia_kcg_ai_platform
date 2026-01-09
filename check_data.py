# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

# 메타 데이터 로드
meta = np.load('prepared_data_v3/meta.npz', allow_pickle=True)

print('=== Meta 정보 ===')
print(f"num_start_area: {meta['num_start_area']}")
print(f"num_end_area: {meta['num_end_area']}")
print(f"total_grids: {meta['total_grids']}")
print(f"start_vocab: {meta['start_vocab']}")
print(f"end_vocab: {meta['end_vocab']}")

# 학습 데이터 로드
data = np.load('prepared_data_v3/training_data.npz', allow_pickle=True)
X_cat = data['X_cat']

print()
print('=== 학습 데이터 ===')
print(f'X_cat shape: {X_cat.shape}')
print(f'X_cat dtype: {X_cat.dtype}')
print(f'start_area: min={X_cat[:, 0].min()}, max={X_cat[:, 0].max()}, unique={len(np.unique(X_cat[:, 0]))}')
print(f'end_area: min={X_cat[:, 1].min()}, max={X_cat[:, 1].max()}, unique={len(np.unique(X_cat[:, 1]))}')
print(f'grid_id: min={X_cat[:, 2].min()}, max={X_cat[:, 2].max()}, unique={len(np.unique(X_cat[:, 2]))}')

print()
print('=== 고유값 분포 ===')
print(f'start_area unique values: {np.unique(X_cat[:, 0])}')
print(f'end_area unique values: {np.unique(X_cat[:, 1])}')

print()
print('=== Xn_num 확인 ===')
Xn_num = data['Xn_num']
print(f'Xn_num shape: {Xn_num.shape}')
print(f'Xn_num dtype: {Xn_num.dtype}')

print()
print('=== Embedding 테스트 ===')
# 작은 테스트
num_start = int(meta['num_start_area'])
num_end = int(meta['num_end_area'])
num_grids = int(meta['total_grids']) + 1

print(f'Embedding sizes: start={num_start}, end={num_end}, grids={num_grids}')

# CPU에서 Embedding 테스트
start_embed = nn.Embedding(num_start, 16)
end_embed = nn.Embedding(num_end, 16)
grid_embed = nn.Embedding(num_grids, 16)

# 샘플 테스트
sample_cat = torch.from_numpy(X_cat[:100])
print(f'sample_cat shape: {sample_cat.shape}')
print(f'sample start indices: {sample_cat[:, 0].min().item()} ~ {sample_cat[:, 0].max().item()}')
print(f'sample end indices: {sample_cat[:, 1].min().item()} ~ {sample_cat[:, 1].max().item()}')
print(f'sample grid indices: {sample_cat[:, 2].min().item()} ~ {sample_cat[:, 2].max().item()}')

try:
    start_out = start_embed(sample_cat[:, 0])
    print(f'start_embed OK: {start_out.shape}')
except Exception as e:
    print(f'start_embed ERROR: {e}')

try:
    end_out = end_embed(sample_cat[:, 1])
    print(f'end_embed OK: {end_out.shape}')
except Exception as e:
    print(f'end_embed ERROR: {e}')

try:
    grid_out = grid_embed(sample_cat[:, 2])
    print(f'grid_embed OK: {grid_out.shape}')
except Exception as e:
    print(f'grid_embed ERROR: {e}')
