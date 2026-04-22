@echo off
setlocal

REM Vai nella cartella dove si trova questo .bat
cd /d "%~dp0"

REM Imposta il path del progetto per gli import Python interni
set PYTHONPATH=%cd%

REM Verifica che il venv esista
if not exist ".venv\Scripts\python.exe" (
    echo [ERRORE] Ambiente virtuale non trovato in .venv
    echo Crea prima il venv oppure copia questo file nella root del progetto.
    pause
    exit /b 1
)

REM Avvia il motore in una finestra separata
start "Engine - ProjectXXX" cmd /k ".venv\Scripts\python.exe main.py"

REM Piccola pausa per dare tempo al motore di inizializzarsi
TIMEOUT /T 3 /NOBREAK >nul

REM Avvia la dashboard Streamlit in LAN in una seconda finestra
start "Dashboard LAN - ProjectXXX" cmd /k "set PYTHONPATH=%cd% && .venv\Scripts\python.exe -m streamlit run ui/dashboard.py --server.address 0.0.0.0 --server.port 8501"

REM Apre il browser locale
start "" "http://localhost:8501"

echo.
echo Avvio completato.
echo Sul tuo PC: http://localhost:8501
echo In rete locale: http://192.168.0.7:8501  (sostituisci con l'IP attuale del PC se cambia)
echo.
pause
