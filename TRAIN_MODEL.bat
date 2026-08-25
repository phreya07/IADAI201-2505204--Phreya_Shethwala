@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: TRAIN_MODEL.bat "C:\path\to\PKLot"
  echo Drag the extracted PKLot folder onto this file or provide its path.
  pause
  exit /b 1
)
where py >nul 2>nul || (echo Python launcher not found. Install Python 3.11 first. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 (echo Could not create the Python 3.11 environment. & pause & exit /b 1)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
pip install -r requirements-training.txt
if errorlevel 1 goto :failed
python prepare_dataset.py "%~1" --per-class 2000
if errorlevel 1 goto :failed
python train.py
if errorlevel 1 goto :failed
echo.
echo Training complete. The calibrated model is ready.
echo Run RUN_APP.bat to open ParkVision.
pause
exit /b 0
:failed
echo.
echo Training stopped because an error occurred. Read the message above.
pause
exit /b 1
