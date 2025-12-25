@echo off
chcp 65001 > nul

echo ============================================================
echo Data Preprocessing - Ship Trajectory
echo ============================================================

REM Python path (kki_gpu2 conda env)
set PYTHON_EXE=C:\Users\user\anaconda3\envs\kki_gpu2\python.exe

REM Data settings
set DATA_FOLDER=G:/NIA_ai_project/항적데이터 추출/여수
set TRANSITION_FOLDER=area_transition_results
set OUTPUT_DIR=prepared_data/yeosu_50

REM Preprocessing parameters
set SEQ_LEN=50
set STRIDE=3
set GRID_SIZE=0.05

echo.
echo [Settings]
echo Python: %PYTHON_EXE%
echo Data folder: %DATA_FOLDER%
echo Transition folder: %TRANSITION_FOLDER%
echo Output: %OUTPUT_DIR%
echo Seq Length: %SEQ_LEN%
echo Stride: %STRIDE%
echo ============================================================
echo.

"%PYTHON_EXE%" prepare_data.py ^
    --data_folder "%DATA_FOLDER%" ^
    --transition_folder "%TRANSITION_FOLDER%" ^
    --output_dir %OUTPUT_DIR% ^
    --seq_len %SEQ_LEN% ^
    --stride %STRIDE% ^
    --grid_size %GRID_SIZE%

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo Preprocessing completed successfully!
    echo Output: %OUTPUT_DIR%
) else (
    echo Preprocessing failed with error code: %ERRORLEVEL%
)
echo ============================================================
pause
