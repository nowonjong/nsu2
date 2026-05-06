@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%myenv\Scripts\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" "%PROJECT_ROOT%run_fusion_with_arduino.py" %*
) else (
    python "%PROJECT_ROOT%run_fusion_with_arduino.py" %*
)
