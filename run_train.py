# -*- coding: utf-8 -*-
"""
선박 항적 예측 모델 학습 실행 파일
============================================
사용법:
    python run_train.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \
                        --transition_folder "area_transition_results" \
                        --start_area "남쪽 진입" \
                        --end_area "여수정박지B" \
                        --epochs 300 \
                        --device cuda

필수 인자:
    --data_folder: 항적 CSV 파일이 저장된 폴더
    --transition_folder: 전이 정보 CSV 폴더
    --start_area: 시작 구역 이름
    --end_area: 도착 구역 이름
"""

import os
import sys
import argparse
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 현재 폴더를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_global_model import train_global_model


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
    start_area: str,
    end_area: str
) -> pd.DataFrame:
    """지정된 시작/도착 구역에 해당하는 항적 데이터 로드"""

    # 시작/도착 구역 필터링
    filtered_df = transition_df[
        (transition_df.start_area == start_area) &
        (transition_df.end_area == end_area)
    ].reset_index(drop=True)

    if len(filtered_df) == 0:
        raise ValueError(f"해당 구간 데이터가 없습니다: {start_area} -> {end_area}")

    print(f"[INFO] 구간 필터링: {start_area} -> {end_area}, {len(filtered_df)} 건")

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

        filename = f'{data_folder}/{mmsi}_{s_area}_{e_area}_{start_time_str}_{end_time_str}.csv'

        if not os.path.exists(filename):
            fail_count += 1
            continue

        try:
            trj = pd.read_csv(filename, encoding='cp949')
            # Unnamed 컬럼 제거
            trj = trj.loc[:, ~trj.columns.str.contains('^Unnamed')]
            trj['fid'] = i
            all_results.append(trj)
            success_count += 1
        except Exception as e:
            print(f"[WARN] 파일 로드 실패: {filename}, {e}")
            fail_count += 1

    if len(all_results) == 0:
        raise ValueError("로드된 항적 데이터가 없습니다.")

    result_df = pd.concat(all_results, ignore_index=True)
    print(f"[INFO] 항적 데이터 로드 완료: {success_count} 성공, {fail_count} 실패")
    print(f"[INFO] 총 데이터 행: {len(result_df)}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="선박 항적 예측 모델 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_train.py --data_folder "G:/NIA_ai_project/항적데이터 추출/여수" \\
                      --transition_folder "area_transition_results" \\
                      --start_area "남쪽 진입" \\
                      --end_area "여수정박지B" \\
                      --epochs 300
        """
    )

    # 필수 인자
    parser.add_argument("--data_folder", type=str, required=True,
                        help="항적 CSV 파일이 저장된 폴더 경로")
    parser.add_argument("--transition_folder", type=str, required=True,
                        help="전이 정보 CSV 파일이 저장된 폴더 경로")
    parser.add_argument("--start_area", type=str, required=True,
                        help="시작 구역 이름")
    parser.add_argument("--end_area", type=str, required=True,
                        help="도착 구역 이름")

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=300,
                        help="최대 학습 에폭 (기본값: 300)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="배치 크기 (기본값: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률 (기본값: 0.001)")
    parser.add_argument("--seq_len", type=int, default=50,
                        help="시퀀스 길이 (기본값: 50)")
    parser.add_argument("--step_size", type=int, default=3,
                        help="스텝 크기 (기본값: 3)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (기본값: 20)")
    parser.add_argument("--warmup_epochs", type=int, default=30,
                        help="Warmup 에폭 수 (기본값: 30)")

    # Smoothness & 변침 학습
    parser.add_argument("--smooth_lambda", type=float, default=0.05,
                        help="Smoothness 정규화 계수 (기본값: 0.05)")
    parser.add_argument("--heading_lambda", type=float, default=0.02,
                        help="침로 smoothness 계수 (기본값: 0.02, 작을수록 변침 허용)")
    parser.add_argument("--turn_boost", type=float, default=2.0,
                        help="변침 구간 가중치 (기본값: 2.0, 클수록 변침 학습 강화)")

    # 기타
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="검증 데이터 비율 (기본값: 0.2)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="학습 장치 (기본값: cuda)")
    parser.add_argument("--save_dir", type=str, default="global_model",
                        help="모델 저장 폴더 (기본값: global_model)")

    args = parser.parse_args()

    print("=" * 60)
    print("선박 항적 예측 모델 학습")
    print("=" * 60)
    print(f"데이터 폴더: {args.data_folder}")
    print(f"전이 정보 폴더: {args.transition_folder}")
    print(f"구간: {args.start_area} -> {args.end_area}")
    print(f"장치: {args.device}")
    print("=" * 60)

    # 1. 전이 정보 로드
    print("\n[STEP 1] 전이 정보 로드")
    transition_df = load_transition_data(args.transition_folder)

    # 2. 항적 데이터 로드
    print("\n[STEP 2] 항적 데이터 로드")
    trajectory_df = load_trajectory_data(
        transition_df,
        args.data_folder,
        args.start_area,
        args.end_area
    )

    # 3. 모델 학습
    print("\n[STEP 3] 모델 학습 시작")
    model_path, scaler_path = train_global_model(
        trajectory_df,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        lr=args.lr,
        smooth_lambda=args.smooth_lambda,
        heading_lambda=args.heading_lambda,
        turn_boost=args.turn_boost,
        val_ratio=args.val_ratio,
        seq_len=args.seq_len,
        step_size=args.step_size,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"모델 저장 위치: {model_path}")
    print(f"스케일러 저장 위치: {scaler_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
