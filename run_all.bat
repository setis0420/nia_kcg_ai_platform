@echo off
chcp 65001 > nul

echo ============================================================
echo 선박 항적 예측 모델 - 전체 파이프라인
echo ============================================================
echo.

REM Python path (kki_gpu2 conda env)
set PYTHON_EXE=C:\Users\user\anaconda3\envs\kki_gpu2\python.exe

REM ============================================================
REM STEP 1: 데이터 전처리
REM ============================================================
echo [STEP 1/2] 데이터 전처리 시작...
echo ============================================================

set DATA_FOLDER=G:/NIA_ai_project/항적데이터 추출/여수
set TRANSITION_FOLDER=area_transition_results
set OUTPUT_DIR=prepared_data
set SEQ_LEN=50
set STRIDE=3
set GRID_SIZE=0.05

"%PYTHON_EXE%" prepare_data.py ^
    --data_folder "%DATA_FOLDER%" ^
    --transition_folder "%TRANSITION_FOLDER%" ^
    --output_dir %OUTPUT_DIR% ^
    --seq_len %SEQ_LEN% ^
    --stride %STRIDE% ^
    --grid_size %GRID_SIZE%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] 데이터 전처리 실패!
    pause
    exit /b 1
)

echo.
echo [STEP 1/2] 데이터 전처리 완료!
echo.

REM ============================================================
REM STEP 2: 모델 학습 (TFT)
REM ============================================================
echo [STEP 2/2] 모델 학습 시작 (Temporal Fusion Transformer)...
echo ============================================================

set DATA_DIR=prepared_data
set SAVE_DIR=global_model_tft
set MODEL_TYPE=tft
set EPOCHS=300
set BATCH_SIZE=256
set LR=0.001
set PATIENCE=20
set WARMUP_EPOCHS=30
set SMOOTH_LAMBDA=0.05
set HEADING_LAMBDA=0.02
set TURN_BOOST=2.0
set VAL_RATIO=0.2
set DEVICE=cuda
set HIDDEN_DIM=128
set EMBED_DIM=16
set N_HEADS=4
set NUM_LSTM_LAYERS=2
set DROPOUT=0.1
set CHUNK_SIZE=100

"%PYTHON_EXE%" train_model.py ^
    --data_dir %DATA_DIR% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --patience %PATIENCE% ^
    --warmup_epochs %WARMUP_EPOCHS% ^
    --smooth_lambda %SMOOTH_LAMBDA% ^
    --heading_lambda %HEADING_LAMBDA% ^
    --turn_boost %TURN_BOOST% ^
    --val_ratio %VAL_RATIO% ^
    --device %DEVICE% ^
    --save_dir %SAVE_DIR% ^
    --embed_dim %EMBED_DIM% ^
    --chunk_size %CHUNK_SIZE% ^
    --model_type %MODEL_TYPE% ^
    --hidden_dim %HIDDEN_DIM% ^
    --n_heads %N_HEADS% ^
    --num_lstm_layers %NUM_LSTM_LAYERS% ^
    --dropout %DROPOUT%

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo 전체 파이프라인 완료!
    echo - 전처리 데이터: %OUTPUT_DIR%/
    echo - 모델: %SAVE_DIR%/model_tft.pth
    echo - 스케일러: %SAVE_DIR%/scaler_tft.npz
) else (
    echo 모델 학습 실패! (error code: %ERRORLEVEL%)
)
echo ============================================================
pause
