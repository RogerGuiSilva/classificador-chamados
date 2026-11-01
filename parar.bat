@echo off
for %%P in (8000 8501) do for /f "tokens=5" %%i in ('netstat -ano ^| findstr /R /C:":%%P .*LISTENING"') do taskkill /PID %%i /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq API - Classificador*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Interface - Classificador*" /F >nul 2>&1
echo Sistema encerrado.
timeout /t 2 /nobreak >nul
