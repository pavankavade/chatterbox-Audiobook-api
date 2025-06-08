@echo off
title Chatterbox Audiobook Studio - GPU Fixed Edition
echo.
echo =================================================================
echo 🎧 Chatterbox Audiobook Studio - GPU Fixed Edition
echo =================================================================
echo.
echo 🚀 Launching with FORCED virtual environment Python...
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
echo 🔍 Testing PyTorch installation...
"%~dp0venv\Scripts\python.exe" -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch location:', torch.__file__)"

if %ERRORLEVEL% neq 0 (
    echo ❌ Error: PyTorch test failed
    pause
    exit /b 1
)

echo 🎯 Using SPECIFIC virtual environment Python executable...
cd /d "%~dp0refactor"

if not exist "app.py" (
    echo ❌ Error: app.py not found in refactor directory
    echo 📂 Current directory: %CD%
    echo 💡 Make sure you're running this from the main project directory
    pause
    exit /b 1
)

echo ✅ Found app.py
echo 🐍 Starting Python application with FORCED venv Python...
echo.
echo 💡 The app will open in your default web browser
echo 🌐 Usually at: http://localhost:7861 (using different port)
echo.
echo ⏹️  Press Ctrl+C in this window to stop the server
echo =================================================================
echo.

REM Force using the virtual environment's specific Python executable
"%~dp0venv\Scripts\python.exe" app.py --port 7861

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Application encountered an error (Exit code: %ERRORLEVEL%)
    echo.
    echo 🔧 Troubleshooting tips:
    echo    • Check that the virtual environment is properly set up
    echo    • Verify PyTorch CUDA installation
    echo    • Try running: pip install -r requirements.txt
    echo.
) else (
    echo.
    echo ✅ Application closed successfully
    echo.
)

echo 🔚 Press any key to close this window...
pause >nul 