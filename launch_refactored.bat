@echo off
title Chatterbox Audiobook Studio - Refactored Edition
echo.
echo =================================================================
echo 🎧 Chatterbox Audiobook Studio - Refactored Edition
echo =================================================================
echo.
echo 🚀 Launching the modular refactored version...
echo 📁 Changing to refactor directory...
echo.

echo 🔧 Activating virtual environment...
call "%~dp0venv\Scripts\activate"

if %ERRORLEVEL% neq 0 (
    echo ❌ Error: Failed to activate virtual environment
    echo 💡 Make sure the venv folder exists in the project directory
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
echo 🔍 Checking PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

cd /d "%~dp0refactor"

if not exist "app.py" (
    echo ❌ Error: app.py not found in refactor directory
    echo 📂 Current directory: %CD%
    echo 💡 Make sure you're running this from the main project directory
    pause
    exit /b 1
)

echo ✅ Found app.py
echo 🐍 Starting Python application...
echo.
echo 💡 The app will open in your default web browser
echo 🌐 Usually at: http://localhost:7860
echo.
echo ⏹️  Press Ctrl+C in this window to stop the server
echo =================================================================
echo.

python app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Application encountered an error (Exit code: %ERRORLEVEL%)
    echo.
    echo 🔧 Troubleshooting tips:
    echo    • Make sure Python is installed and in your PATH
    echo    • Check that all dependencies are installed
    echo    • Try running: pip install -r requirements.txt
    echo.
) else (
    echo.
    echo ✅ Application closed successfully
    echo.
)

echo 🔚 Press any key to close this window...
pause >nul 