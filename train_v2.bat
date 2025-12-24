@echo off
chcp 65001 > nul

echo ============================================================
echo Ship Trajectory Prediction Model V2 Training
echo (Universal Model - Categorical + Grid)
echo ============================================================

REM Python path (kki_gpu2 conda env)
set PYTHON_EXE=C:\Users\user\anaconda3\envs\kki_gpu2\python.exe

REM Data settings
set DATA_FOLDER=G:/NIA_ai_project/항적데이터 추출/여수
set TRANSITION_FOLDER=area_transition_results

REM Training hyperparameters
set EPOCHS=300
set BATCH_SIZE=256
set LR=0.001
set SEQ_LEN=50
set STRIDE=3
set PATIENCE=20
set WARMUP_EPOCHS=30
set SMOOTH_LAMBDA=0.05
set HEADING_LAMBDA=0.02
set TURN_BOOST=2.0
set VAL_RATIO=0.2
set DEVICE=cuda
set SAVE_DIR=global_model_v2

REM V2 specific settings
set EMBED_DIM=16
set GRID_SIZE=0.05

echo.
echo [Settings]
echo Python: %PYTHON_EXE%
echo Data folder: %DATA_FOLDER%
echo Transition folder: %TRANSITION_FOLDER%
echo Device: %DEVICE%
echo Epochs: %EPOCHS%
echo Seq Length: %SEQ_LEN%
echo Stride: %STRIDE%
echo Embed Dim: %EMBED_DIM%
echo Grid Size: %GRID_SIZE%
echo ============================================================
echo.

"%PYTHON_EXE%" run_train_v2.py ^
    --data_folder "%DATA_FOLDER%" ^
    --transition_folder "%TRANSITION_FOLDER%" ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --seq_len %SEQ_LEN% ^
    --stride %STRIDE% ^
    --patience %PATIENCE% ^
    --warmup_epochs %WARMUP_EPOCHS% ^
    --smooth_lambda %SMOOTH_LAMBDA% ^
    --heading_lambda %HEADING_LAMBDA% ^
    --turn_boost %TURN_BOOST% ^
    --val_ratio %VAL_RATIO% ^
    --device %DEVICE% ^
    --save_dir %SAVE_DIR% ^
    --embed_dim %EMBED_DIM% ^
    --grid_size %GRID_SIZE%

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
    echo Model saved to: %SAVE_DIR%
    echo Checkpoints saved to: %SAVE_DIR%/checkpoints
) else (
    echo Training failed with error code: %ERRORLEVEL%
)
echo ============================================================
pause
