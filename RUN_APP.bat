@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating ParkVision environment...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -c "import streamlit, cv2, onnxruntime, pandas" 2>nul
if errorlevel 1 (
  echo Installing required packages. This first launch can take several minutes...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo ERROR: Package installation failed. Check your internet connection and Python 3.10-3.12 installation.
    pause
    exit /b 1
  )
)
echo Starting ParkVision at http://localhost:8501
python -m streamlit run app.py --server.port 8501
if errorlevel 1 (
  echo Streamlit stopped with an error. Read the message above.
  pause
)
