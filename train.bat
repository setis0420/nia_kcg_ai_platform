@echo off
chcp 65001 > nul

echo ============================================================
echo Ship Trajectory Prediction Model Training
echo ============================================================

REM Python path (kki_gpu2 conda env)
set PYTHON_EXE=C:\Users\user\anaconda3\envs\kki_gpu2\python.exe

REM Data settings
set DATA_FOLDER=G:/NIA_ai_project/항적데이터 추출/여수
set TRANSITION_FOLDER=area_transition_results
set START_AREA=남쪽 진입
set END_AREA=여수정박지B

REM Training hyperparameters
set EPOCHS=100
set BATCH_SIZE=256
set LR=0.001
set SEQ_LEN=50
set STEP_SIZE=3
set PATIENCE=20
set WARMUP_EPOCHS=30
set SMOOTH_LAMBDA=0.05
set VAL_RATIO=0.2
set DEVICE=cuda
set SAVE_DIR=global_model

echo.
echo [Settings]
echo Python: %PYTHON_EXE%
echo Data folder: %DATA_FOLDER%
echo Transition folder: %TRANSITION_FOLDER%
echo Route: %START_AREA% to %END_AREA%
echo Device: %DEVICE%
echo Epochs: %EPOCHS%
echo ============================================================
echo.

"%PYTHON_EXE%" run_train.py --data_folder "%DATA_FOLDER%" --transition_folder "%TRANSITION_FOLDER%" --start_area "%START_AREA%" --end_area "%END_AREA%" --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --seq_len %SEQ_LEN% --step_size %STEP_SIZE% --patience %PATIENCE% --warmup_epochs %WARMUP_EPOCHS% --smooth_lambda %SMOOTH_LAMBDA% --val_ratio %VAL_RATIO% --device %DEVICE% --save_dir %SAVE_DIR%

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
) else (
    echo Training failed with error code: %ERRORLEVEL%
)
echo ============================================================
pause
