@echo off
chcp 65001 > nul

echo ============================================================
echo Model Training - Ship Trajectory Prediction
echo ============================================================
echo.
echo Select Model:
echo   1. LSTM        - Basic LSTM (fast)
echo   2. LSTM+Attn   - LSTM + Attention (balanced)
echo   3. Transformer - Pure Transformer (parallel)
echo   4. TFT         - Temporal Fusion Transformer (best accuracy)
echo ============================================================
echo.
set /p MODEL_CHOICE=Enter model number (1-4, default=4):

if "%MODEL_CHOICE%"=="" set MODEL_TYPE=tft
if "%MODEL_CHOICE%"=="1" set MODEL_TYPE=lstm
if "%MODEL_CHOICE%"=="2" set MODEL_TYPE=lstm_attn
if "%MODEL_CHOICE%"=="3" set MODEL_TYPE=transformer
if "%MODEL_CHOICE%"=="4" set MODEL_TYPE=tft

if not defined MODEL_TYPE set MODEL_TYPE=tft

echo.
echo Selected model: %MODEL_TYPE%
echo ============================================================

REM Python path (kki_gpu2 conda env)
set PYTHON_EXE=C:\Users\user\anaconda3\envs\kki_gpu2\python.exe

REM Data and Model settings
set DATA_DIR=prepared_data
set SAVE_DIR=model/yeosu_tft

REM Training hyperparameters
set EPOCHS=30
set BATCH_SIZE=256
set LR=0.001
set PATIENCE=4
set WARMUP_EPOCHS=10
set SMOOTH_LAMBDA=0.05
set HEADING_LAMBDA=0.02
set TURN_BOOST=2.0
set VAL_RATIO=0.2
set DEVICE=cuda

REM Model architecture
set HIDDEN_DIM=128
set EMBED_DIM=16
set N_HEADS=4
set NUM_LSTM_LAYERS=2
set DROPOUT=0.1

REM Memory settings
set CHUNK_SIZE=100

echo.
echo [Settings]
echo Python: %PYTHON_EXE%
echo Data folder: %DATA_DIR%
echo Save folder: %SAVE_DIR%
echo Model: %MODEL_TYPE%
echo Device: %DEVICE%
echo Epochs: %EPOCHS%
echo ============================================================

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
    --hidden_dim %HIDDEN_DIM% ^
    --n_heads %N_HEADS% ^
    --num_lstm_layers %NUM_LSTM_LAYERS% ^
    --dropout %DROPOUT% ^
    --model_type %MODEL_TYPE%

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
    echo Model saved to: %SAVE_DIR%/model_%MODEL_TYPE%/
    echo Checkpoints saved to: %SAVE_DIR%/model_%MODEL_TYPE%/checkpoints/
) else (
    echo Training failed with error code: %ERRORLEVEL%
)
echo ============================================================
pause
