@echo off
setlocal
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
cd /d "%~dp0app"
echo Programa de segmentacion de madera balsa.
echo La camara esta apagada al iniciar. Para cerrar, pulse Ctrl+C.
"%~dp0runtime\python.exe" -B demo_panel.py --device auto --max-size 384 --threshold 0.5 %*
set "BALSA_EXIT=%ERRORLEVEL%"
if not "%BALSA_EXIT%"=="0" pause
exit /b %BALSA_EXIT%
