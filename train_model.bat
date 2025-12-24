@echo off
chcp 65001 > nul

echo ============================================================
echo Model Training - Ship Trajectory Prediction V2
echo ============================================================

REM Python path (kki_gpu2 conda env)
set PYTHON_EXE=C:\Users\user\anaconda3\envs\kki_gpu2\python.exe

REM Data & Model settings
set DATA_DIR=prepared_data
set SAVE_DIR=global_model_v2

REM Training hyperparameters
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

REM Embedding settings
set EMBED_DIM=16

echo.
echo [Settings]
echo Python: %PYTHON_EXE%
echo Data folder: %DATA_DIR%
echo Save folder: %SAVE_DIR%
echo Device: %DEVICE%
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo Embed Dim: %EMBED_DIM%
echo ============================================================
echo.

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
    --embed_dim %EMBED_DIM%

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
