# V7 멀티 임베딩 H3 모델

## 개요
H3 셀 + SOG 구간 + COG 구간을 멀티 임베딩으로 결합한 Transformer 기반 항적 예측 모델

## 데이터
- **전처리 데이터 위치**: `학습데이터_전처리완료/울산/`
- **청크 파일**: `sequences_chunk_000.pkl` ~ `sequences_chunk_080.pkl` (81개)
- **총 시퀀스**: 40,154,828개
- **데이터 구조**:
  - `input`: (50, 4) - [lat, lon, sog, cog] 50 타임스텝
  - `target`: (20, 2) - [lat, lon] 20 타임스텝 예측
  - `meta`: (2,) - [shiptype, length_bin]

## 모델 구조
```
H3MultiEmbedModel
├── cell_embed: H3 셀 임베딩 (num_cells, 128)
├── sog_embed: SOG 구간 임베딩 (21, 128)
├── cog_embed: COG 구간 임베딩 (24, 128)
├── pos_embed: 위치 임베딩 (512, 128)
├── transformer: TransformerEncoder (4 layers, 8 heads)
└── fc: Linear -> (num_cells * 20)
```

## 설정
| 항목 | 값 |
|------|-----|
| H3 Resolution | 9 (~700m 직경) |
| SEQ_LEN | 50 (입력 시퀀스) |
| PRED_LEN | 20 (예측 시퀀스) |
| SOG 구간 | 21개 (0-1, 1-2, ..., 20+ 노트) |
| COG 구간 | 24개 (15도 단위) |
| embed_dim | 128 |
| num_heads | 8 |
| num_layers | 4 |
| dropout | 0.1 |

## 학습 실행
```bash
# 전체 데이터 학습
"C:\Users\user\anaconda3\envs\kki_gpu2\python.exe" run_train_v7_multi.py

# 테스트 (일부 청크만)
"C:\Users\user\anaconda3\envs\kki_gpu2\python.exe" run_train_v7_multi.py --max_chunks 5 --epochs 10

# 옵션
--max_chunks N    # 최대 청크 수 (None=전체)
--epochs N        # 에포크 수 (기본: 50)
--batch_size N    # 배치 크기 (기본: 256)
--lr FLOAT        # 학습률 (기본: 0.001)
```

## 저장 위치
- **모델**: `global_model_v7_multi/울산/model.pth`
- **설정**: `global_model_v7_multi/울산/config.pkl`
- **셀 매핑**: `global_model_v7_multi/울산/cell_mapping.pkl`

## 학습 프로세스
1. 전처리된 pkl 파일에서 시퀀스 로드
2. 10% 샘플링으로 H3 셀 어휘 구축
3. Train/Val 분할 (90%/10%)
4. CrossEntropyLoss + AdamW + CosineAnnealingLR
5. Early stopping (patience=10)

## 예측 방식
- 입력: 50개 타임스텝의 (lat, lon, sog, cog) → H3 셀 + SOG bin + COG bin
- 출력: 20개 타임스텝의 H3 셀 예측 (각 스텝별 분류)
- H3 셀 중심 좌표로 변환하여 최종 위치 예측
