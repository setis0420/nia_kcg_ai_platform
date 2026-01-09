# -*- coding: utf-8 -*-
"""
선박 항적 예측 모델 V5 학습 실행 파일 (LLM 스타일)
===================================================
핵심 특징:
- 격자 ID 시퀀스만으로 다음 격자 예측 (Language Model 스타일)
- 속력/침로 벡터 완전 삭제
- 속력 8노트 이상만 필터링
- MMSI별 개별 모델 학습 및 저장

사용법:
    python run_train_v5.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                           --transition_folder "area_transition_results" \
                           --epochs 100 \
                           --device cuda
"""

import os
import sys
import argparse
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 현재 폴더를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_global_model_v5 import train_all_mmsi_models


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
        df['source_file'] = f
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] 전이 정보 로드 완료: {len(merged_df)} 건, {len(file_list)} 파일")
    return merged_df


def load_trajectory_data(
    transition_df: pd.DataFrame,
    data_folder: str,
) -> pd.DataFrame:
    """모든 구간의 항적 데이터 로드"""

    filtered_df = transition_df.reset_index(drop=True)

    if len(filtered_df) == 0:
        raise ValueError("전이 정보 데이터가 없습니다.")

    # 구간 통계 출력
    unique_routes = filtered_df.groupby(['start_area', 'end_area']).size()
    print(f"[INFO] 전체 구간 데이터: {len(filtered_df)} 건")
    print(f"[INFO] 구간 종류: {len(unique_routes)} 개")

    all_results = []
    success_count = 0
    fail_count = 0
    total = len(filtered_df)

    print(f"[INFO] 파일 로딩 시작... (총 {total} 건)")

    for i in range(total):
        # 진행률 표시 (1000건마다)
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  진행: {i+1}/{total} ({100*(i+1)/total:.1f}%)", flush=True)

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
            # Unnamed 컬럼 제거
            trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
            trj['fid'] = i
            all_results.append(trj)
            success_count += 1
        except Exception as e:
            fail_count += 1

    if len(all_results) == 0:
        raise ValueError("로드된 항적 데이터가 없습니다.")

    result_df = pd.concat(all_results, ignore_index=True)
    print(f"[INFO] 항적 데이터 로드 완료: {success_count} 성공, {fail_count} 실패")
    print(f"[INFO] 총 데이터 행: {len(result_df)}")
    print(f"[INFO] MMSI 종류: {result_df['mmsi'].nunique()}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="선박 항적 예측 모델 V5 학습 (LLM 스타일 격자 ID 시퀀스)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_train_v5.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \\
                         --transition_folder "area_transition_results" \\
                         --epochs 100

V5 핵심 특징:
  - 격자 ID 시퀀스만으로 다음 격자 예측 (Language Model 스타일)
  - 속력/침로 벡터 완전 삭제
  - 속력 8노트 이상만 필터링 (항해 중인 선박만)
  - MMSI별 개별 모델 학습 및 저장
        """
    )

    # 필수 인자
    parser.add_argument("--data_folder", type=str, required=True,
                        help="항적 CSV 파일이 저장된 폴더 경로")
    parser.add_argument("--transition_folder", type=str, required=True,
                        help="전이 정보 CSV 파일이 저장된 폴더 경로")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=100,
                        help="최대 학습 에폭 (기본값: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="배치 크기 (기본값: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률 (기본값: 0.001)")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="입력 시퀀스 길이 (기본값: 50)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sliding window 이동 간격 (기본값: 1)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (기본값: 20)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Warmup 에폭 수 (기본값: 10)")

    # 격자 설정
    parser.add_argument("--grid_size", type=float, default=0.01,
                        help="격자 크기 (도 단위, 기본값: 0.01 = 약 1.1km)")

    # 속력 필터
    parser.add_argument("--min_sog", type=float, default=8.0,
                        help="최소 속력 필터 (노트, 기본값: 8.0)")

    # 기타
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="검증 데이터 비율 (기본값: 0.2)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="학습 장치 (기본값: cuda)")
    parser.add_argument("--save_dir", type=str, default="global_model_v5",
                        help="모델 저장 폴더 (기본값: global_model_v5)")

    args = parser.parse_args()

    print("=" * 60)
    print("선박 항적 예측 모델 V5 학습 (LLM 스타일)")
    print("=" * 60)
    print(f"핵심 특징:")
    print(f"  - 격자 ID 시퀀스 → 다음 격자 ID 예측")
    print(f"  - 속력/침로 벡터 삭제")
    print(f"  - 속력 {args.min_sog}노트 이상만 필터링")
    print(f"  - MMSI별 개별 모델 저장")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_folder}")
    print(f"전이 정보 폴더: {args.transition_folder}")
    print(f"장치: {args.device}")
    print(f"격자 크기: {args.grid_size}도 (약 {args.grid_size * 111:.1f}km)")
    print("=" * 60)

    # 1. 전이 정보 로드
    print("\n[STEP 1] 전이 정보 로드")
    transition_df = load_transition_data(args.transition_folder)

    # 2. 항적 데이터 로드 (모든 구간)
    print("\n[STEP 2] 항적 데이터 로드 (모든 구간)")
    trajectory_df = load_trajectory_data(
        transition_df,
        args.data_folder,
    )

    # 3. MMSI별 모델 학습
    print("\n[STEP 3] MMSI별 모델 학습 시작")
    results = train_all_mmsi_models(
        trajectory_df,
        seq_len=args.seq_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        val_ratio=args.val_ratio,
        grid_size=args.grid_size,
        min_sog=args.min_sog,
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"학습된 MMSI 수: {len(results)}")
    print(f"모델 저장 위치: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
