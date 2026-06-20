@echo off
REM Setup script for Ollama and required models

echo ==============================================
echo Setting up Ollama for text2sql-agent-mcp
echo ==============================================

REM Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ollama is not installed!
    echo Please install Ollama from: https://ollama.com/download
    echo After installation, run this script again.
    pause
    exit /b 1
)

echo Ollama is installed. Checking if the Ollama service is running...

REM Try to list models to check if Ollama service is running
ollama list >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting Ollama service...
    start /b ollama serve
    echo Waiting for Ollama to start...
    timeout /t 5 /nobreak
)

echo Checking for required model: llama3:8b

REM Check if model exists
ollama list | findstr "llama3:8b" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Model llama3:8b not found. Downloading now...
    ollama pull llama3:8b
) else (
    echo Model llama3:8b is already installed.
)

echo ==============================================
echo Setup complete! You can now run trial1.py.
echo ==============================================

pause
