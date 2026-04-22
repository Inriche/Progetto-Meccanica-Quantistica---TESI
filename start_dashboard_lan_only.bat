@echo off
setlocal

cd /d "%~dp0"
set PYTHONPATH=%cd%

if not exist ".venv\Scripts\python.exe" (
    echo [ERRORE] Ambiente virtuale non trovato in .venv
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m streamlit run ui/dashboard.py --server.address 0.0.0.0 --server.port 8501
