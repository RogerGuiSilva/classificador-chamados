@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Ambiente virtual .venv nao encontrado nesta pasta.
    echo Coloque este arquivo na raiz do projeto, ao lado da pasta .venv.
    pause
    exit /b 1
)
for %%P in (8000 8501) do for /f "tokens=5" %%i in ('netstat -ano ^| findstr /R /C:":%%P .*LISTENING"') do taskkill /PID %%i /F /T >nul 2>&1
echo Iniciando a API...
start "API - Classificador" cmd /k ".venv\Scripts\activate.bat && uvicorn app.main:app --reload"
timeout /t 8 /nobreak >nul
echo Iniciando a interface...
start "Interface - Classificador" cmd /k ".venv\Scripts\activate.bat && streamlit run ui/app.py"
exit
